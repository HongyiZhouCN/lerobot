from pathlib import Path

# import gym_pusht  # noqa: F401

import gym_pusht  # noqa: F401
import gym_aloha
import gymnasium as gym
import imageio
import numpy
import torch

from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy

import lerobot.configs.types as types

device = "cuda"
policy = PI0Policy.from_pretrained("lerobot/pi0")

pretrained_diffusion_path = "lerobot/diffusion_pusht"

diff_policy = DiffusionPolicy.from_pretrained(pretrained_diffusion_path)

# Initialize evaluation environment to render two observation types:
# an image of the scene and state/position of the agent. The environment
# also automatically stops running after 300 interactions/steps.
env = gym.make(
    "gym_aloha/AlohaTransferCube-v0",
    obs_type="pixels_agent_pos",
    max_episode_steps=300,
)
# env = gym.make(
#     "gym_pusht/PushT-v0",
#     obs_type="pixels_agent_pos",
#     max_episode_steps=300,
# )
# We can verify that the shapes of the features expected by the policy match the ones from the observations
# produced by the environment
policy.config.input_features = diff_policy.config.input_features
policy.config.input_features['observation.image'].shape = (3, 480, 640)
policy.config.input_features['observation.state'].shape = (14,)
policy.config.output_features = diff_policy.config.output_features
print(policy.config.input_features)
print(env.observation_space)

# Similarly, we can check that the actions produced by the policy will match the actions expected by the
# environment
policy.config.output_features['action'].shape = (14,)
print(policy.config.output_features)
print(env.action_space)

# Reset the policy and environments to prepare for rollout
policy.reset()
numpy_observation, info = env.reset(seed=42)

# Prepare to collect every rewards and all the frames of the episode,
# from initial state to final state.
rewards = []
frames = []

# Render frame of the initial state
frames.append(env.render())

step = 0
done = False
while not done:
    # Prepare observation for the policy running in Pytorch
    state = torch.from_numpy(numpy_observation["agent_pos"])
    image = torch.from_numpy(numpy_observation["pixels"]["top"])

    # Convert to float32 with image from channel first in [0,255]
    # to channel last in [0,1]
    state = state.to(torch.float32)
    image = image.to(torch.float32) / 255
    image = image.permute(2, 0, 1)

    # Send data tensors from CPU to GPU
    state = state.to(device, non_blocking=True)
    image = image.to(device, non_blocking=True)

    # Add extra (empty) batch dimension, required to forward the policy
    state = state.unsqueeze(0)
    image = image.unsqueeze(0)

    # Create the policy input dictionary
    observation = {
        "observation.state": state,
        "observation.image": image,
        "task": ["push_t"],
    }

    # Predict the next action with respect to the current observation
    with torch.inference_mode():
        action = policy.select_action(observation)

    # Prepare the action for the environment
    numpy_action = action.squeeze(0).to("cpu").numpy()

    # Step through the environment and receive a new observation
    numpy_observation, reward, terminated, truncated, info = env.step(numpy_action)
    print(f"{step=} {reward=} {terminated=}")
