import gymnasium as gym
from stable_baselines3 import SAC, TD3, A2C, DDPG
import os
import argparse
import torch 
from stable_baselines3.sac.policies import MlpPolicy

# Create directories to hold models and logs
model_dir = "models"
log_dir = "logs"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

def train(rl_model, algo_name=""):
    TIMESTEPS = 1000000
    iters = 0
    while True:
        iters += 1

        rl_model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
        rl_model.save(f"{model_dir}/{algo_name}_{TIMESTEPS*iters}")

def test(env, rl_model):
    obs = env.reset()[0]
    done = False
    extra_steps = 500
    while True:
        action, _ = rl_model.predict(obs)
        obs, _, done, _, _ = env.step(action)

        if done:
            extra_steps -= 1

            if extra_steps < 0:
                break


def linear_schedule(initial_value):
    """
    Linear learning rate schedule.

    :param initial_value: (float or str)
    :return: (function)
    """
    if isinstance(initial_value, str):
        initial_value = float(initial_value)

    def func(progress):
        """
        Progress will decrease from 1 (beginning) to 0
        :param progress: (float)
        :return: (float)
        """
        return progress * initial_value

    return func



if __name__ == '__main__':
    train_mode = True

    # policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=[128, 128])
    pusher_env = gym.make("Pusher-v4", render_mode="human")

    if train_mode:
        # rl_model = SAC('MlpPolicy', pusher_env, verbose=1, 
        #                device='cuda', 
        #                tensorboard_log=log_dir, 
        #                learning_rate=linear_schedule(0.003),
        #                #policy_kwargs=policy_kwargs
        # )
        rl_model = SAC(MlpPolicy, pusher_env, verbose=1, 
                       device='cuda', 
                       tensorboard_log=log_dir, 
                       learning_rate=linear_schedule(0.001),
                       #policy_kwargs=policy_kwargs
        )
        train(rl_model, "SAC")
    else:
        path_to_model = "models/SAC_2000000.zip"
        rl_model = SAC.load(path_to_model, env=pusher_env)
        test(pusher_env, rl_model)
