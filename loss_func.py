from energy_manager import compute_energy_levels, get_energy_level_in_percents
from parameters import CONSTANT_PENALTY_VALUE, MAX_SPEED
from track import Track
import numpy as np


def compute_loss_func(speed_vector: list, track: Track):
    """Computes loss function according to energy levels and section speeds"""
    return sum([track.sections.loc[i].length / speed_vector[i] for i in range(len(track.sections))])


def compute_total_penalty(speed_vector: list, track: Track, func_type="original"):
    """Computes total penalty according to energy levels and section speeds"""
    energy_levels = compute_energy_levels(track, speed_vector)
    energy_level_penalty = compute_energy_level_penalty(energy_levels, func_type)

    speed_penalty = compute_speed_penalty(speed_vector, func_type)

    total_penalty = energy_level_penalty + speed_penalty
    return total_penalty


def compute_speed_penalty(speeds: list, func_type):
    """Penalizes speed values more than MAX_SPEED and less than 0"""
    penalty_func = box_penalty_func_factory(0, MAX_SPEED, func_type)
    return sum([penalty_func(x) for x in speeds])


def compute_energy_level_penalty(energy_levels: list, func_type):
    """Penalizes energy level values more than 100 and less than 0"""
    penalty_func = box_penalty_func_factory(0, 100, func_type)
    energy_levels_in_percents = list(map(get_energy_level_in_percents, energy_levels))
    # Starting from 1 due to full charge of battery at beginning of race
    return sum([penalty_func(x) for x in energy_levels_in_percents[1:]])


def box_penalty_func_factory(a, b, func_type):
    """Creates penalty function for box constraints: a < x < b"""

    if func_type is "original":
        def inner(x, deg=2):
            return np.piecewise(x, [x < a, x > b], [lambda x: (a - x) ** deg,
                                                    lambda x: (x - b) ** deg])

    elif func_type is "parabolic":
        width = b - a
        shift = a / (width / 2) + 1

        def inner(x, koef=10, deg=8):
            return koef * (1 / (width / 2) * x - shift) ** deg

    elif func_type is "constant":
        def inner(x, penalty_value=CONSTANT_PENALTY_VALUE):
            if a < x < b:
                return 0
            else:
                return penalty_value
    else:
        assert False, "No such func_type of penalty function: " + func_type

    return inner
