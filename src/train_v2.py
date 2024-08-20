import os
import sys
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.sac.policies import MlpPolicy
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from schedulers import linear_schedule

model_dir = "train_res/models"
log_dir = "train_res/logs"

TIMESTEPS = 3000000

pusher_env_train = Monitor(gym.make("Pusher-v4", render_mode=None), log_dir)
pusher_env_eval = Monitor(gym.make("Pusher-v4", render_mode="human"), log_dir)

eval_callback = EvalCallback(
    pusher_env_eval, 
    best_model_save_path=model_dir,
    log_path=log_dir, 
    eval_freq=10000,
    deterministic=True, 
    render=True)

rl_model = SAC(
    MlpPolicy, 
    pusher_env_train, 
    verbose=1, 
    device='cuda', 
    tensorboard_log=log_dir, 
    learning_rate=linear_schedule(0.001),
)

rl_model.learn(total_timesteps=TIMESTEPS, callback=eval_callback, reset_num_timesteps=True)