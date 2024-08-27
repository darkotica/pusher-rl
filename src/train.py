import os
import sys
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.sac.policies import MlpPolicy
from stable_baselines3.common.callbacks import EvalCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from  stable_baselines3.common.noise import NormalActionNoise
import numpy as np
import json

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from schedulers import linear_schedule, cosine_schedule
from custom_pusher_env import get_custom_pusher_env, generate_test_positions
import torch

def get_callbacks(env, frequency, model_save_dir, log_save_dir):
    eval_callback = EvalCallback(
        env, 
        best_model_save_path=model_save_dir,
        log_path=log_save_dir, 
        eval_freq=frequency,
        deterministic=True, 
        render=True)

    checkpoint_callback = CheckpointCallback(
        save_freq=frequency,
        save_path=model_save_dir,
        name_prefix="rl_model",
        verbose=2
    )

    cb_list = CallbackList([checkpoint_callback, eval_callback])

    return cb_list


base_dir = "train_res_fixed_goal_no_slide_center_arm_400_noise_moving_longer_timesteps"

os.makedirs(base_dir, exist_ok=True)


model_dir = f"{base_dir}/models"
log_dir = f"{base_dir}/logs"

TIMESTEPS = 4000000

test_pos = generate_test_positions()

with open(f"{base_dir}/test_pos.json", "w") as f:
    json.dump(test_pos, f, indent=4)


pusher_env_train = Monitor(get_custom_pusher_env(max_number_of_steps=400, render_mode="human", test_pos=test_pos), log_dir)
pusher_env_eval = Monitor(get_custom_pusher_env(max_number_of_steps=400, test_pos=test_pos), log_dir)

callbacks = get_callbacks(pusher_env_eval, 10000, model_dir, log_dir)

policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=[512, 512])

rl_model = SAC(
    MlpPolicy, 
    pusher_env_train, 
    verbose=1, 
    device='cuda', 
    tensorboard_log=log_dir, 
    learning_rate=cosine_schedule(0.0003, 0.000003),
    policy_kwargs=policy_kwargs,
    action_noise=NormalActionNoise(np.array(0), np.array(0.1))
)

rl_model.learn(total_timesteps=TIMESTEPS, callback=callbacks, reset_num_timesteps=True)