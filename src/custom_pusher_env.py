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
        observation_space = Box(low=-np.inf, high=np.inf, shape=(25,), dtype=np.float64)
        model_file = kwargs.pop("model_file", "pusher.xml")
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
        reward_dist = -np.linalg.norm(vec_2)
        reward_ctrl = -np.square(a).sum()
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
            False,
            False,
            dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl),
        )

    def reset_model(self):
        qpos = self.init_qpos

        # self.goal_pos = np.asarray([0, 0])
        # while True:
        #     self.cylinder_pos = np.concatenate(
        #         [
        #             # cylinder can now be all around the goal
        #             self.np_random.uniform(low=-0.3, high=0.3, size=1),
        #             self.np_random.uniform(low=-0.3, high=0.3, size=1),
        #         ]
        #     )
        #     if np.linalg.norm(self.cylinder_pos - self.goal_pos) > 0.17:
        #         break
        self.goal_pos = np.asarray([
            self.np_random.uniform(low=-0.4, high=0.4, size=1)[0], 
            self.np_random.uniform(low=-0.9, high=0.9, size=1)[0]])
        self.cylinder_pos = np.asarray([
            self.np_random.uniform(low=-0.4, high=0.4, size=1)[0], 
            self.np_random.uniform(low=-0.9, high=0.9, size=1)[0]])

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
                self.data.qpos.flat[:8],
                self.data.qvel.flat[:8],
                self.get_body_com("tips_arm"),
                self.get_body_com("object"),
                self.get_body_com("goal"),
            ]
        )


def get_custom_pusher_env(max_number_of_steps=100, 
                          render_mode=None, 
                          model_file="/home/darko-tica/Documents/pusher-rl/pusher-rl/src/custom_assets/pusher_custom.xml"):
    env = CustomPusherEnv(render_mode=render_mode, model_file=model_file)

    env = PassiveEnvChecker(env)
    env = OrderEnforcing(env)
    env = TimeLimit(env, max_number_of_steps)

    return env