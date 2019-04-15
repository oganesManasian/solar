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
    speed_penalty = compute_speed_penalty(section_speeds)
    # print("Loss: Only time: %5.2f Only penalty: %5.2f" % (loss, energy_level_penalty + speed_penalty))
    loss += energy_level_penalty + speed_penalty
    return loss


def compute_speed_penalty(speeds: list):
    penalty_func = box_penalty_func_factory(0, 25)
    return sum([penalty_func(x) for x in speeds])


def compute_energy_level_penalty(energy_levels: list):
    penalty_func = box_penalty_func_factory(0, 100)
    energy_levels_in_percents = list(map(get_energy_level_in_percents, energy_levels))
    # Starting from 1 due to full charge of battery at beginning of race
    return sum([penalty_func(x) for x in energy_levels_in_percents[1:]])


def box_penalty_func_factory(a, b):
    """Creates continuous penalty function for box constraints: a < x < b"""
    width = b - a
    shift = a / (width / 2) + 1

    def inner(x, koef=20, deg=30):
        return koef * (1 / (width / 2) * x - shift) ** deg
    return inner


def compute_energy_level_penalty_old(energy_levels: list):
    """Computes penalty"""
    energy_levels_in_percents = list(map(get_energy_level_in_percents, energy_levels))
    # Starting from 1 due to full charge of battery at beginning of race
    return sum([penalty_func_1(x) for x in energy_levels_in_percents[1:]])


def penalty_func_1(x, koef=20, deg=30):
    """Continuous penalty function for box constraints: 0 < x < 100"""
    return koef * (1 / 50 * x - 1) ** deg


def penalty_func_2(x, deg=3):
    """Discontinuous penalty function for box constraints: 0 < x < 100"""
    return max(max(0, -x), max(0, x - 100)) ** deg

