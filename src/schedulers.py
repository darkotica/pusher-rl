

import math


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


def cosine_schedule(start_lr, end_lr, total_timesteps):
    # TODO istrazi ima restart neki
    def func(cur_timestep):
        """
        Progress will decrease from 1 (beginning) to 0
        :param progress: (float)
        :return: (float)
        """
        return end_lr + 0.5*(start_lr - end_lr)*(1 + math.cos(cur_timestep * math.pi/total_timesteps))

    return func
