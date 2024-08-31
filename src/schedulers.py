

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

def cosine_schedule_WR(
        initial_learning_rate,
        first_decay_steps,
        num_steps_total,
        t_mul=2.0,
        m_mul=1.0,
        alpha=0.0,):

    def func(progress):
        """
        Progress will decrease from 1 (beginning) to 0
        :param progress: (float)
        :return: (float)
        """
        progress = 1 - progress
        current_step = progress * num_steps_total
        completed_fraction = current_step/first_decay_steps

        i_restart = math.floor(completed_fraction)
        completed_fraction -= i_restart

        m_fac = m_mul**i_restart
        cosine_decayed = (
            0.5
            * m_fac
            * (
                1.0
                + math.cos(math.pi * completed_fraction)
            )
        )
        decayed = (1 - alpha) * cosine_decayed + alpha

        return initial_learning_rate * decayed

    return func