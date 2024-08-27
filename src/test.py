import json
import gymnasium as gym
from stable_baselines3 import SAC, TD3, A2C, DDPG
import os
import argparse
import torch 
from stable_baselines3.sac.policies import MlpPolicy

from tqdm import tqdm

from custom_pusher_env import get_custom_pusher_env


def test(env, rl_model, num_steps, num_test_pos):
    obs, _ = env.reset()

    tests_solved = []
    tests_failed = []
    
    for num_test in tqdm(range(num_test_pos)):
        for _ in range(num_steps):
            action, _ = rl_model.predict(obs)
            obs, _, done, _, _ = env.step(action)

            if done:
                break
        
        if done:
            tests_solved.append(num_test)
        else:
            tests_failed.append(num_test)

        if num_test < num_test_pos - 1:
            obs = env.reset()[0]

    return len(tests_solved), len(tests_failed)


base_dir = "train_res_fixed_goal_no_slide_center_arm_400_noise_moving_longer_timesteps"
best_model_path = f"{base_dir}/models/best_model.zip"

test_pos_path = f"{base_dir}/test_pos.json"

with open(test_pos_path, "r") as f:
    test_pos = json.load(f)

test_env = get_custom_pusher_env(render_mode="human", test_pos=test_pos, test_mode=True)

rl_model = SAC.load(best_model_path, test_env)

test_res = test(test_env, rl_model, 400, len(test_pos))

print(test_res)