from functools import reduce
from energy_manager import compute_energy_levels, get_energy_level_in_percents
from track import Track


def loss_func(section_speeds: list, track: Track):
    """Computes loss function"""
    loss = 0.0

    for i in range(len(track.sections)):
        section_time = track.sections.loc[i].length / section_speeds[i]
        loss += section_time

    energy_levels = compute_energy_levels(track, section_speeds)
    energy_level_penalty = compute_energy_level_penalty(energy_levels)
    # print("Loss: Only time: %5.2f Only penalty: %5.2f" % (loss, energy_level_penalty))
    loss += energy_level_penalty  # TODO Tune: add coefficient
    return loss


def compute_energy_level_penalty(energy_levels: list):
    """Computes penalty"""
    energy_levels_in_percents = list(map(get_energy_level_in_percents, energy_levels))
    # Starting from 1 due to full charge of battery at beginning of race
    return sum([penalty_func_1(x) for x in energy_levels_in_percents[1:]])


def compute_energy_level_penalty_fast(energy_levels: list):
    """Computing penalties with discontinuous penalty function
        Works faster than compute_energy_level_penalty"""
    energy_levels_in_percents = list(map(get_energy_level_in_percents, energy_levels))

    penalty_sum = 0.0

    below_level = list(filter(lambda x: x < 0, energy_levels_in_percents))
    below_level_penalty = list(map(lambda x: abs(x) ** 3, below_level))
    penalty_sum += reduce(lambda x, y: x + y, below_level_penalty, 0)

    above_level = list(filter(lambda x: x > 100, energy_levels_in_percents))
    above_level_penalty = list(map(lambda x: abs(x - 100) ** 3, above_level))
    penalty_sum += reduce(lambda x, y: x + y, above_level_penalty, 0)
    return penalty_sum


def penalty_func_1(x):
    """Continuous function for computing penalties for violation of task conditions"""
    return 10 * (1 / 50 * x - 1) ** 20


def penalty_func_2(x):
    """Discontinuous function for computing penalties for violation of task conditions"""
    if x < 0:
        return abs(x) ** 3
    elif x > 100:
        return abs(x - 100 ** 3)
    else:
        return 0
