

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


def cosine_schedule(start_lr, end_lr):

    def func(progress):
        """
        Progress will decrease from 1 (beginning) to 0
        :param progress: (float)
        :return: (float)
        """
        progress = 1 - progress
        # progress = cur timestep / total timesteps
        return end_lr + 0.5*(start_lr - end_lr)*(1 + math.cos(math.pi * progress))

    return func
