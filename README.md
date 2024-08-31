# Pusher-rl
- robot trained in simulated environment which is based on [gymnasium farama project pusher-v4](https://gymnasium.farama.org/environments/mujoco/pusher/)
- key differences: 
    - arm is centered
    - larger area of coverage
    - additional axis of rotation (for shoulder)

## Installation
- in order to run the environment you need to download and install [MuJoCo](https://github.com/google-deepmind/mujoco/releases)
- install the requirements file
```
pip install -r requirements.txt
```
- run ```src/train.py``` script for training, and ```src/test.py``` script for testing