from energy_manager import compute_energy_levels, get_energy_level_in_percents
from parameters import PENALTY_VALUE, MAX_SPEED
from track import Track


def compute_loss_func(section_speeds: list, track: Track):
    """Computes loss function according to energy levels and section speeds"""
    return sum([track.sections.loc[i].length / section_speeds[i] for i in range(len(track.sections))])


def compute_total_penalty(section_speeds: list, track: Track, continuous=True):
    """Computes total penalty according to energy levels and section speeds"""
    energy_levels = compute_energy_levels(track, section_speeds)
    energy_level_penalty = compute_energy_level_penalty(energy_levels, continuous)

    speed_penalty = compute_speed_penalty(section_speeds, continuous)

    total_penalty = energy_level_penalty + speed_penalty
    return total_penalty


def box_penalty_func_factory(a, b, continuous=True):
    """Creates penalty function for box constraints: a < x < b"""
    if continuous:
        width = b - a
        shift = a / (width / 2) + 1

        def inner(x, koef=20, deg=30):
            return koef * (1 / (width / 2) * x - shift) ** deg
    else:
        def inner(x, penalty_value=PENALTY_VALUE):
            if a < x < b:
                return 0
            else:
                return penalty_value
    return inner


def compute_speed_penalty(speeds: list, continuous):
    """Penalizes speed values more than MAX_SPEED and less than 0"""
    penalty_func = box_penalty_func_factory(0, MAX_SPEED, continuous)
    return sum([penalty_func(x) for x in speeds])


def compute_energy_level_penalty(energy_levels: list, continuous):
    """Penalizes energy level values more than 100 and less than 0"""
    penalty_func = box_penalty_func_factory(0, 100, continuous)
    energy_levels_in_percents = list(map(get_energy_level_in_percents, energy_levels))
    # Starting from 1 due to full charge of battery at beginning of race
    return sum([penalty_func(x) for x in energy_levels_in_percents[1:]])


def compute_energy_level_penalty_v2(energy_levels: list):
    """Penalizes changes in energy level between to near sections which are greater then 5% of battery charge"""
    penalty_func = box_penalty_func_factory(-5, 5)
    energy_levels_in_percents = list(map(get_energy_level_in_percents, energy_levels))
    # Starting from 1 due to full charge of battery at beginning of race
    energy_level_changes = [energy_levels_in_percents[i] - energy_levels_in_percents[i - 1]
                            for i in range(1, len(energy_levels_in_percents))]

    return sum([penalty_func(x) for x in energy_level_changes])
