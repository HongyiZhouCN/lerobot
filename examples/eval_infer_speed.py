from pathlib import Path

import imageio
import numpy
import torch

from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy

import lerobot.configs.types as types

from tqdm import tqdm

from torch.cuda import Event

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

device = "cuda"
policy = PI0Policy.from_pretrained("lerobot/pi0")

pretrained_diffusion_path = "lerobot/diffusion_pusht"

diff_policy = DiffusionPolicy.from_pretrained(pretrained_diffusion_path)

policy.config.input_features = diff_policy.config.input_features
policy.config.input_features['observation.image'].shape = (3, 224, 224)
policy.config.input_features['observation.state'].shape = (14,)
policy.config.output_features = diff_policy.config.output_features
print(policy.config.input_features)

policy.config.output_features['action'].shape = (14,)
print(policy.config.output_features)

# Reset the policy and environments to prepare for rollout
policy.reset()



test_states = numpy.random.random((1000, 14))
test_states = torch.from_numpy(test_states).to(dtype=torch.float32, device=device, non_blocking=True)

test_images = numpy.random.random((1000, 3, 224, 224))
test_images = torch.from_numpy(test_images).to(dtype=torch.float32, device=device, non_blocking=True)

for j in tqdm(range(50)):
    # Create the policy input dictionary
    state = test_states[j].unsqueeze(0)
    image = test_images[j].unsqueeze(0)

    observation = {
        "observation.state": state,
        "observation.image": image,
        "task": ["push_t"],
    }

    # Predict the next action with respect to the current observation
    with torch.inference_mode():
        action = policy.select_action(observation)
    

# Create CUDA events
start_event = Event(enable_timing=True)
end_event = Event(enable_timing=True)
    
# Measure time for multiple runs
total_time = 0.0

times = []

for i in tqdm(range(1000)):
    # Add extra (empty) batch dimension, required to forward the policy
    state = test_states[i].unsqueeze(0)
    image = test_images[i].unsqueeze(0)


    # Create the policy input dictionary
    observation = {
        "observation.state": state,
        "observation.image": image,
        "task": ["push_t"],
    }

    start_event.record()

    # Predict the next action with respect to the current observation
    with torch.inference_mode():
        action = policy.select_action(observation)
    
    end_event.record()
    end_event.synchronize()
    times.append(start_event.elapsed_time(end_event))

times = numpy.array(times)
avg_time = numpy.mean(times)
std_time = numpy.std(times)

print("average time: ", avg_time)
print("std time: ", std_time)


if __name__ == "__main__":
    pass