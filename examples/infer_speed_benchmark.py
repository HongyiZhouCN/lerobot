from pathlib import Path
import numpy as np
import torch
from torch.cuda import Event
from tqdm import tqdm
import os
import time

from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.common.policies.act.modeling_act import ACTPolicy
import lerobot.configs.types as types

from contextlib import contextmanager

import gc

class VRAMMonitor:
    """
    A class to monitor VRAM usage during model inference.
    Subtracts baseline memory usage to focus on the model's memory footprint.
    """
    def __init__(self, device='cuda:0', sampling_interval=0.1, exclude_baseline=True):
        """
        Initialize the VRAM monitor.
        
        Args:
            device (str): The CUDA device to monitor.
            sampling_interval (float): Time between VRAM usage samples in seconds.
            exclude_baseline (bool): Whether to subtract baseline memory usage.
        """
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Cannot monitor VRAM.")
        
        self.device = device
        self.sampling_interval = sampling_interval
        self.device_idx = int(device.split(':')[1]) if ':' in device else 0
        self.exclude_baseline = exclude_baseline
        
        # Force garbage collection and empty CUDA cache before setting baseline
        gc.collect()
        torch.cuda.empty_cache()
        
        # Capture baseline memory usage (from datasets, etc.)
        self.baseline_memory = torch.cuda.memory_allocated(self.device_idx) if exclude_baseline else 0
        print(f"Baseline VRAM usage: {self.baseline_memory / (1024**2):.2f} MB (will be excluded from measurements)")
        
        self.reset()
    
    def reset(self):
        """Reset collected metrics."""
        self.vram_used = []
        self.vram_total = torch.cuda.get_device_properties(self.device_idx).total_memory
        self.timestamps = []
        self.peak_vram = 0
        self.start_time = None
    
    def sample(self):
        """Take a sample of current VRAM usage, excluding baseline if configured."""
        allocated = torch.cuda.memory_allocated(self.device_idx)
        
        # Subtract baseline memory to focus only on additional memory used by the model
        used = max(0, allocated - self.baseline_memory)
        
        self.vram_used.append(used)
        self.peak_vram = max(self.peak_vram, used)
        self.timestamps.append(time.time() - self.start_time)
    
    @contextmanager
    def monitor(self):
        """Context manager to monitor VRAM usage during a code block."""
        self.reset()
        self.start_time = time.time()
        
        # Start monitoring thread
        import threading
        stop_event = threading.Event()
        
        def sampling_loop():
            while not stop_event.is_set():
                self.sample()
                time.sleep(self.sampling_interval)
        
        # Start the sampling thread
        thread = threading.Thread(target=sampling_loop)
        thread.daemon = True
        thread.start()
        
        try:
            # Run the code in the context
            yield self
        finally:
            # Stop the sampling thread
            stop_event.set()
            thread.join()
            
            # Take a final sample
            self.sample()
    
    def get_summary(self):
        """Return a summary of the VRAM usage statistics."""
        if not self.vram_used:
            return {"error": "No samples collected"}
        
        # Convert bytes to MB for better readability
        used_mb = [u / (1024**2) for u in self.vram_used]
        peak_mb = self.peak_vram / (1024**2)
        total_mb = self.vram_total / (1024**2)
        baseline_mb = self.baseline_memory / (1024**2)
        
        summary = {
            "average_used_mb": np.mean(used_mb),
            "peak_used_mb": peak_mb,
            "total_available_mb": total_mb,
            "peak_usage_percentage": (self.peak_vram / self.vram_total) * 100,
            "number_of_samples": len(self.vram_used),
            "duration_seconds": self.timestamps[-1] if self.timestamps else 0
        }
        
        # Add baseline info if we're excluding it
        if self.exclude_baseline:
            summary["baseline_mb"] = baseline_mb
            summary["measurement_type"] = "Model-only (baseline excluded)"
        else:
            summary["measurement_type"] = "Total (baseline included)"
            
        return summary



def setup_environment(gpu_id="1"):
    """Set up the GPU environment."""
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return device


def load_policy(policy_class, pretrained_path, input_features=None, output_features=None):
    """Load a policy model and configure it."""
    policy = policy_class.from_pretrained(pretrained_path)
    
    # Update config if provided
    if input_features:
        policy.config.input_features = input_features
    
    if output_features:
        policy.config.output_features = output_features
        
    policy.reset()
    return policy


def generate_test_data(num_samples, state_shape, image_shape, device):
    """Generate random test data for benchmarking."""
    test_states = np.random.random((num_samples, *state_shape))
    test_states = torch.from_numpy(test_states).to(dtype=torch.bfloat16, device=device, non_blocking=True)
    
    test_images = np.random.random((num_samples, *image_shape))
    test_images = torch.from_numpy(test_images).to(dtype=torch.bfloat16, device=device, non_blocking=True)
    
    return test_states, test_images


def create_observation(state, image, policy, task="push_t"):
    """Create an observation dictionary for policy input."""
    # if isinstance(policy, ACTPolicy) or isinstance(policy, DiffusionPolicy):
    return {
            "observation.images.top": image,
            "observation.state": state,
            "task": [task],
        }
    # return {
        # "observation.state": state,
        # "observation.image": image,
        # "task": [task],
    # }


def benchmark_policy(policy, test_states, test_images, num_runs=1000, task="push_t", replan_every=50):
    """Benchmark a policy's inference speed."""
    # Warm-up runs
    for j in tqdm(range(replan_every), desc="Warm-up"):
        state = test_states[j].unsqueeze(0)
        image = test_images[j].unsqueeze(0)
        observation = create_observation(state, image, policy, task)
        
        with torch.inference_mode():
            _ = policy.select_action(observation)
    
    # Create CUDA events for timing
    start_event = Event(enable_timing=True)
    end_event = Event(enable_timing=True)
    
    # Benchmark runs
    policy.reset()
    times = []
    times_generate_chunk = []

    for i in tqdm(range(num_runs), desc="Benchmarking"):
        state = test_states[i % len(test_states)].unsqueeze(0)
        image = test_images[i % len(test_images)].unsqueeze(0)
        observation = create_observation(state, image, policy, task)
            
        # Record start time
        start_event.record()
            
        with torch.autocast('cuda', dtype=torch.bfloat16):
        # Run inference
            with torch.inference_mode():
                _ = policy.select_action(observation)
            
        # Record end time and calculate elapsed time
        end_event.record()
        end_event.synchronize()
        times.append(start_event.elapsed_time(end_event))
        if i % replan_every == 0:
            times_generate_chunk.append(start_event.elapsed_time(end_event))
    
    
    # Calculate statistics
    times = np.array(times)
    avg_time = np.mean(times)
    std_time = np.std(times)
    times_generate_chunk = np.array(times_generate_chunk)
    avg_time_generate_chunk = np.mean(times_generate_chunk)
    std_time_generate_chunk = np.std(times_generate_chunk)
    
    return {
        "times": times,
        "avg_time": avg_time,
        "std_time": std_time,
        "times_generate_chunk": times_generate_chunk,
        "avg_time_generate_chunk": avg_time_generate_chunk,
        "std_time_generate_chunk": std_time_generate_chunk
    }


def main():
    # Setup
    device = setup_environment(gpu_id="7")
    print(f"Using device: {device}")

    # Generate test data
    num_samples = 1000
    state_shape = (14,)
    image_shape = (3, 224, 224)
    test_states, test_images = generate_test_data(num_samples, state_shape, image_shape, device)

    # monitor = VRAMMonitor(device="cuda:0")
    
    diff_policy = load_policy(
        DiffusionPolicy, 
        "/home/zhou/Codes/flower_rss25/lerobot/outputs/train/diffusion_aloha_insertion_48/checkpoints/000002/pretrained_model"
    ).to(dtype=torch.bfloat16)

    monitor = VRAMMonitor(device="cuda:0")


    # act_policy = load_policy(
    #     ACTPolicy,
    #     "lerobot/act_aloha_sim_transfer_cube_human"
    # ).to(torch.bfloat16)
        
    
    pi0_policy = load_policy(
        PI0Policy, 
        "lerobot/pi0", 
        input_features=diff_policy.config.input_features,
        output_features=diff_policy.config.output_features
    ).to(dtype=torch.bfloat16)

    pi0_policy.config.input_features['observation.images.top'].shape = (3, 224, 224)
    pi0_policy.config.input_features['observation.state'].shape = (14,)
    pi0_policy.config.output_features['action'].shape = (14,)
    
    # diff_policy.config.input_features['observation.images.top'].shape = (3, 224, 224)
    # diff_policy.config.input_features['observation.state'].shape = (14,)
    # diff_policy.config.output_features['action'].shape = (14,)
    # diff_policy.config.n_action_steps=48
    # diff_policy.config.horizon = 48

    # act_policy.config.input_features['observation.images.top'].shape = (3, 224, 224)
    # act_policy.config.input_features['observation.state'].shape = (14,)
    # act_policy.config.output_features['action'].shape = (14,)
    # act_policy.config.chunk_size=100
    # act_policy.config.n_action_steps=50

    
    with monitor.monitor():

        # print("\nBenchmarking ACT Policy...")
        # act_results = benchmark_policy(act_policy, test_states, test_images, replan_every=50)
        # print(f"ACT Policy - Average time: {act_results['avg_time']:.4f} ms, \
        #         Average time generate chunk: {act_results['avg_time_generate_chunk']:.4f} ms")
    

        print("\nBenchmarking PI0 Policy...")
        pi0_results = benchmark_policy(pi0_policy, test_states, test_images, replan_every=50)
        print(f"PI0 Policy - Average time: {pi0_results['avg_time']:.4f} ms,\
                Average time generate chunk: {pi0_results['avg_time_generate_chunk']:.4f} ms")
    
        # print("\nBenchmarking Diffusion Policy...")
        # diff_results = benchmark_policy(diff_policy, test_states, test_images, replan_every=47)
        # print(f"Diffusion Policy - Average time: {diff_results['avg_time']:.4f} ms, Std: {diff_results['std_time']:.4f} ms, \
        #       # Average time generate chunk: {diff_results['avg_time_generate_chunk']:.4f} ms")
    
    # Print VRAM usage summary
    summary = monitor.get_summary()
    print(summary)


if __name__ == "__main__":
    main()