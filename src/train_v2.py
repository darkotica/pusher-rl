import os
import sys
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.sac.policies import MlpPolicy
from stable_baselines3.common.callbacks import EvalCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from schedulers import linear_schedule, cosine_schedule
from custom_pusher_env import get_custom_pusher_env

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



model_dir = "train_res_w_slide/models"
log_dir = "train_res_w_slide/logs"

TIMESTEPS = 3000000

pusher_env_train = Monitor(get_custom_pusher_env(), log_dir)
pusher_env_eval = Monitor(get_custom_pusher_env(render_mode="human"), log_dir)

callbacks = get_callbacks(pusher_env_eval, 10000, model_dir, log_dir)

rl_model = SAC(
    MlpPolicy, 
    pusher_env_train, 
    verbose=1, 
    device='cuda', 
    tensorboard_log=log_dir, 
    learning_rate=cosine_schedule(0.0003, 0.000003),
)

rl_model.learn(total_timesteps=TIMESTEPS, callback=callbacks, reset_num_timesteps=True)