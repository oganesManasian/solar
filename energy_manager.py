import math
from functools import reduce
import pandas as pd
import matplotlib.pyplot as plt
from track import Track
from utils import nonnegative, print_res_if_debug
from environment_data import compute_solar_radiation, get_weather_params_owm

DEBUG_FUNC_RESULTS = False

# Battery
ENERGY_LEVEL_PERCENT_MAX = 100
ENERGY_LEVEL_PERCENT_MIN = 0
BATTER_CHARGE_MAX = 5100 * 3600
BATTER_CHARGE_MIN = 0  # W * h
EFFICIENCY_BATTERY = 0.98
# Environment
SOLAR_RADIATION = 500  # W / m²  # TODO add modeling
GRAVITY_ACCELERATION = 9.81
FRICTION_RESISTANCE_RATE = 0.0025
AIR_DENSITY = 1.18
# Vehicle
VEHICLE_FRONT_AREA = 0.6  # m²
FRONTAL_DENSITY_RATE = 0.15
VEHICLE_PANEL_AREA = 4  # m²
VEHICLE_EQUIPMENT_POWER = 40  # W
VEHICLE_WEIGHT = 385  # kg
EFFICIENCY_INCOME = 0.2 * 0.985
EFFICIENCY_OUTCOME = 0.94  # TODO add modeling


def loss_func(section_speeds: list, track: Track):
    loss = 0.0

    for i in range(len(track.sections)):
        section_time = track.sections.loc[i].length / section_speeds[i]
        loss += section_time

    energy_levels = compute_energy_levels(track, section_speeds)
    energy_level_penalty = compute_energy_level_penalty(energy_levels)
    # print("Loss only time", loss)
    # print("Loss only penalty", energy_level_penalty)
    loss += energy_level_penalty  # TODO Tune: add coefficient

    return loss


def compute_energy_level_penalty(energy_levels: list):
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


def get_energy_level_in_percents(energy_level_wt):
    return energy_level_wt / BATTER_CHARGE_MAX * 100


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


def compute_energy_levels(track: Track, section_speeds: list):
    """Computes energy level of each sector"""
    energy_levels = [BATTER_CHARGE_MAX]
    hour = 15  # TODO calculate using section length and speeds
    for i in range(len(track.sections)):
        energy_income = compute_energy_income(section_speeds[i],
                                              track.sections.loc[i].solar_radiation[hour],
                                              # SOLAR_RADIATION,  # TODO pass from track.sections.loc[i].solar_radiation
                                              VEHICLE_PANEL_AREA,
                                              track.sections.loc[i].slope_angle,
                                              track.sections.loc[i].length,
                                              EFFICIENCY_INCOME)
        energy_outcome = compute_energy_outcome(section_speeds[i],
                                                track.sections.loc[i].length,
                                                AIR_DENSITY,
                                                VEHICLE_FRONT_AREA,
                                                FRONTAL_DENSITY_RATE,
                                                VEHICLE_WEIGHT,
                                                GRAVITY_ACCELERATION,
                                                track.sections.loc[i].slope_angle,
                                                FRICTION_RESISTANCE_RATE,
                                                VEHICLE_EQUIPMENT_POWER,
                                                EFFICIENCY_OUTCOME)
        energy_level = energy_levels[-1] + (energy_income - energy_outcome) * EFFICIENCY_BATTERY
        energy_levels.append(energy_level)
    return energy_levels


def compute_energy_levels_full(track: Track, section_speeds: list):
    """Computes energy level, energy income and outcome on each sector"""
    energy_levels = [BATTER_CHARGE_MAX]
    energy_incomes = [0]
    energy_outcomes = [0]

    model_params = pd.DataFrame(columns=["solar_radiation",
                                         "vehicle_panel_area",
                                         "section_slope_angle",
                                         "section_length",
                                         "efficiency_income",
                                         "air_density",
                                         "vehicle_front_area",
                                         "frontal_density_rate",
                                         "vehicle weight",
                                         "gravity",
                                         "friction_resistance_rate",
                                         "vehicle_equipment_power",
                                         "efficiency_outcome",
                                         "section_speed"])

    for i in range(len(track.sections)):
        energy_income = compute_energy_income(section_speeds[i],
                                              SOLAR_RADIATION,  # TODO pass from track.sections.loc[i].solar_radiation
                                              VEHICLE_PANEL_AREA,
                                              track.sections.loc[i].slope_angle,
                                              track.sections.loc[i].length,
                                              EFFICIENCY_INCOME)
        energy_incomes.append(energy_income)

        energy_outcome = compute_energy_outcome(section_speeds[i],
                                                track.sections.loc[i].length,
                                                AIR_DENSITY,
                                                VEHICLE_FRONT_AREA,
                                                FRONTAL_DENSITY_RATE,
                                                VEHICLE_WEIGHT,
                                                GRAVITY_ACCELERATION,
                                                track.sections.loc[i].slope_angle,
                                                FRICTION_RESISTANCE_RATE,
                                                VEHICLE_EQUIPMENT_POWER,
                                                EFFICIENCY_OUTCOME)
        energy_outcomes.append(energy_outcome)

        energy_level = energy_levels[-1] + (energy_income - energy_outcome) * EFFICIENCY_BATTERY
        energy_levels.append(energy_level)

        model_params.loc[len(model_params)] = [SOLAR_RADIATION,
                                               VEHICLE_PANEL_AREA,
                                               track.sections.loc[i].slope_angle,
                                               track.sections.loc[i].length,
                                               EFFICIENCY_INCOME,
                                               AIR_DENSITY,
                                               VEHICLE_FRONT_AREA,
                                               FRONTAL_DENSITY_RATE,
                                               VEHICLE_WEIGHT,
                                               GRAVITY_ACCELERATION,
                                               FRICTION_RESISTANCE_RATE,
                                               VEHICLE_EQUIPMENT_POWER,
                                               EFFICIENCY_OUTCOME,
                                               section_speeds[i]]

    return {"levels": energy_levels,
            "incomes": energy_incomes,
            "outcomes": energy_outcomes,
            "params": model_params}


def draw_energy_levels(energy_levels: list, energy_incomes: list, energy_outcomes: list):
    plt.title("Energy flow")
    plt.xlabel("Section number")
    plt.ylabel("Energy level")

    plt.plot(range(len(energy_levels)), energy_levels, "b", label='Energy level')

    # Starting from 1 due to full charge of battery at beginning of race
    violation_points = [(x + 1, y) for (x, y) in enumerate(energy_levels[1:])
                        if not 0 < get_energy_level_in_percents(y) < 100]
    if len(violation_points) > 0:
        X, Y = zip(*violation_points)
        plt.plot(X, Y, "ro", label="Violations")

    plt.plot(range(len(energy_incomes)), energy_incomes, "g-", label='Energy income')

    plt.plot(range(len(energy_outcomes)), energy_outcomes, "r-", label='Energy outcome')

    plt.xticks(range(len(energy_levels)))
    plt.legend()
    plt.grid()
    plt.show()


@print_res_if_debug(DEBUG_FUNC_RESULTS)
def compute_energy_income(section_speed,
                          solar_radiation,
                          panel_area,
                          slope_angle,
                          section_length,
                          efficiency_income):
    p_sun = solar_radiation * panel_area * math.cos(slope_angle)
    p_real = p_sun * efficiency_income
    t = section_length / section_speed
    energy_income = p_real * t
    return energy_income


def find_distance(start_pnt, end_pnt):
    dx = end_pnt[0] - start_pnt[0]
    dy = end_pnt[1] - start_pnt[1]
    return math.sqrt(dx ** 2 + dy ** 2)


@print_res_if_debug(DEBUG_FUNC_RESULTS)
def compute_energy_outcome(section_speed,
                           section_length,
                           air_density,
                           vehicle_front_area,
                           frontal_resistance_rate,
                           vehicle_weight,
                           gravity_acceleration,
                           slope_angle,
                           friction_resistance_rate,
                           equipment_power,
                           efficiency_outcome):
    drag_force = compute_drag_force(air_density,
                                    section_speed,
                                    vehicle_front_area,
                                    frontal_resistance_rate)
    gravity_force = compute_gravity_force(vehicle_weight,
                                          gravity_acceleration,
                                          slope_angle)
    friction_resistance_force = compute_friction_resistance_force(vehicle_weight,
                                                                  gravity_acceleration,
                                                                  slope_angle,
                                                                  friction_resistance_rate)
    total_force = drag_force + gravity_force + friction_resistance_force

    section_time = section_length / section_speed
    work_motion = total_force * section_length / efficiency_outcome
    work_equipment = equipment_power * section_time
    energy_outcome = work_motion + work_equipment

    # energy_motion = total_force * section_length
    # energy_equipment = equipment_power * section_length / section_speed
    # energy_outcome = energy_motion * efficiency_outcome + energy_equipment
    return energy_outcome


@nonnegative
def compute_drag_force(air_density,
                       section_speed,
                       vehicle_front_area,
                       frontal_resistance_rate):
    return 1 / 2 * air_density * section_speed ** 2 * vehicle_front_area * frontal_resistance_rate


def compute_gravity_force(vehicle_weight,
                          gravity_acceleration,
                          slope_angle):
    return vehicle_weight * gravity_acceleration * math.sin(slope_angle)


@nonnegative
def compute_friction_resistance_force(vehicle_weight,
                                      gravity_acceleration,
                                      slope_angle,
                                      friction_resistance_rate):
    return vehicle_weight * gravity_acceleration * friction_resistance_rate * math.cos(slope_angle)