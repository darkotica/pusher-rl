import numpy as np

from gymnasium.wrappers.env_checker import PassiveEnvChecker
from gymnasium.wrappers.order_enforcing import OrderEnforcing
from gymnasium.wrappers.time_limit import TimeLimit

import numpy as np

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box


DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": -1,
    "distance": 4.0,
}



class CustomPusherEnv(MujocoEnv, utils.EzPickle):

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 20,
    }

    def __init__(self, **kwargs):
        utils.EzPickle.__init__(self, **kwargs)
        observation_space = Box(low=-np.inf, high=np.inf, shape=(23,), dtype=np.float64)
        model_file = kwargs.pop("model_file", "pusher.xml")
        self.test_pos = kwargs.pop("test_pos")
        self.test_mode = kwargs.pop("test_mode")
        self.num_of_test_pos = len(self.test_pos)
        self.current_idx_of_test_pos = 0
        MujocoEnv.__init__(
            self,
            model_file,
            5,
            observation_space=observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )

    def step(self, a):
        self.do_simulation(a, self.frame_skip)

        vec_1 = self.get_body_com("object") - self.get_body_com("tips_arm")
        vec_2 = self.get_body_com("object") - self.get_body_com("goal")

        reward_near = -np.linalg.norm(vec_1)
        dist = -np.linalg.norm(vec_2)
        reward_ctrl = -np.square(a).sum()

        if abs(dist) < 0.1:
            reward_dist = 10
        else:
            reward_dist = dist

        reward = reward_dist + 0.1 * reward_ctrl + 0.5 * reward_near

        # self.do_simulation(a, self.frame_skip)
        # fix iz docs za v5, stavljeno je gore
        # racunat reward za prethodni step lol
        if self.render_mode == "human":
            self.render()

        ob = self._get_obs()
        return (
            ob,
            reward,
            abs(dist) < 0.1,
            abs(dist) < 0.1,
            dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl),
        )

    def reset_model(self):
        qpos = self.init_qpos

        if not self.test_mode:
            self.goal_pos = np.asarray([0, 0])
            while True:
                self.goal_pos = np.asarray([
                    self.np_random.uniform(low=-0.3, high=0.3, size=1)[0], 
                    self.np_random.uniform(low=-0.5, high=0.5, size=1)[0]])
                self.cylinder_pos = np.asarray([
                    self.np_random.uniform(low=-0.3, high=0.3, size=1)[0], 
                    self.np_random.uniform(low=-0.5, high=0.5, size=1)[0]])
                
                # we don't want them to spawn too close
                if np.linalg.norm(self.cylinder_pos - self.goal_pos) < 0.4:
                    continue
                
                valid = True
                for pos in self.test_pos:
                    if (pos[0] == self.goal_pos).all() or (pos[1] == self.cylinder_pos).all():
                        valid = False
                        break

                if valid:
                    break
        else:
            cur_goal, cur_cylinder = self.test_pos[self.current_idx_of_test_pos]
            self.goal_pos = np.array(cur_goal)
            self.cylinder_pos = np.asarray(cur_cylinder)
            self.current_idx_of_test_pos += 1

        qpos[-4:-2] = self.cylinder_pos
        qpos[-2:] = self.goal_pos
        qvel = self.init_qvel + self.np_random.uniform(
            low=-0.005, high=0.005, size=self.model.nv
        )
        qvel[-4:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate(
            [
                self.data.qpos.flat[:7],
                self.data.qvel.flat[:7],
                self.get_body_com("tips_arm"),
                self.get_body_com("object"),
                self.get_body_com("goal"),
            ]
        )
    

def generate_test_positions(num_positions=60): # ovo stavi da bude otp area 
    positions = []

    min_y = -3
    max_y = 3
    min_x = -5
    max_x = 5

    width_x = max_x - min_x

    for i in range(num_positions):
        y_val = (i // width_x - max_y)/10
        x_val = (i % width_x - max_x)/10


        while True:
            goal_pos = np.asarray([
                np.random.uniform(low=y_val, high=y_val + 0.1, size=1)[0], 
                np.random.uniform(low=x_val, high=x_val + 0.1, size=1)[0],
                ])
            cylinder_pos = np.asarray([
                np.random.uniform(low=min_y/10, high=max_y/10, size=1)[0], 
                np.random.uniform(low=min_x/10, high=max_x/10, size=1)[0],
                ])
            
            # we don't want them to spawn too close
            if np.linalg.norm(cylinder_pos - goal_pos) > 0.4:
                break
            
        
        positions.append((goal_pos.tolist(), cylinder_pos.tolist()))
    
    return positions


def get_custom_pusher_env(max_number_of_steps=100, 
                          render_mode=None, 
                          test_pos=[],
                          test_mode=False,
                          #model_file="/home/darko-tica/Documents/pusher-rl/pusher-rl/src/custom_assets/pusher_custom.xml"):
                          model_file="/home/darko-tica/Documents/pusher-rl/pusher-rl/src/custom_assets/pusher_custom_center_arm.xml"):
    env = CustomPusherEnv(render_mode=render_mode, model_file=model_file, test_pos=test_pos, test_mode=test_mode)

    env = PassiveEnvChecker(env)
    env = OrderEnforcing(env)
    env = TimeLimit(env, max_number_of_steps)

    return env