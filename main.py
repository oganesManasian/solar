# coding=utf-8
from parameters import INIT_SPEED, START_DATETIME, DRIVE_TIME_BOUNDS
from track import Track
import energy_manager
import optimization_methods
from loss_func import compute_loss_func, compute_total_penalty
import datetime
import os
import numpy as np
from utils import draw_solution, draw_speed_solar_radiation_relation
import matplotlib.pyplot as plt

font = {'size': 22}
plt.rc('font', **font)


def main(init_precision="3dim", func_type="original"):
    if not os.path.isdir("logs"):
        os.mkdir("logs")

    # Load race track
    track = Track(max_section_length=30000, max_section_slope_angle=0.15)
    track.load_track_from_csv("data/track_Australia.csv")
    # track.preprocess_track()
    # track.draw_track_altitudes("График высот маршрута до предобработки")
    track.combine_points_to_sections()
    # track.draw_track_altitudes("График высот маршрута после предобработки")

    init_speeds = [INIT_SPEED] * len(track.sections)
    track.compute_arrival_times(START_DATETIME, DRIVE_TIME_BOUNDS, init_speeds)
    track.compute_weather_params()
    # track.sections.solar_radiation = track.sections.solar_radiation * 1.18  # Simulate October conditions
    # track.draw_track_features("Ключевые параметры маршрута")

    # track.sections = track.sections[:10]  # Taking only small part of track for speeding up tests

    # Test that optimization is possible
    test_speeds = [1] * len(track.sections)
    energy_levels_test = energy_manager.compute_energy_levels(track, test_speeds)
    assert (energy_levels_test[-1] >= 0), "Too little energy to cover the distance!"

    # Find initial approximation
    args = [compute_loss_func, compute_total_penalty, track]
    if init_precision == "1dim":
        kwargs = dict(func_type="original",
                      low_speed_range=np.linspace(start=15, stop=25, num=40))
    elif init_precision == "3dim":
        kwargs = dict(func_type="original",
                      low_speed_range=range(15, 20),
                      high_speed_range=range(25, 36),
                      n_range=range(1, int(np.ceil(len(track.sections) * 0.1))))
    else:
        assert False, "Not implemented such type of initial precision"
    base_speed_vector = optimization_methods.grid_search(*args, **kwargs)
    draw_solution(base_speed_vector, title="Начальное приближение")
    track.compute_arrival_times(START_DATETIME, DRIVE_TIME_BOUNDS, base_speed_vector)
    track.compute_weather_params()

    # Optimize speed
    print("Init loss:", compute_loss_func(base_speed_vector, track),
          "\nInit penalty:", compute_total_penalty(base_speed_vector, track, func_type="original"))

    optimal_speeds = optimization_methods.penalty_method(func=compute_loss_func,
                                                         penalty_func=compute_total_penalty,
                                                         x0=base_speed_vector,
                                                         args=track,
                                                         func_type=func_type,
                                                         eps=1,
                                                         tol=1e-3,
                                                         max_step=20,
                                                         show_info=True)

    # print("Optimization result:", optimal_speeds)
    print("Final loss:", compute_loss_func(optimal_speeds, track),
          "\nFinal penalty:", compute_total_penalty(optimal_speeds, track, func_type="original"))
    print("Total travel time:", round(compute_loss_func(optimal_speeds, track) / 3600, 2), "hours")

    # Try to find better solution in point vicinity
    optimization_methods.random_change_method(optimal_speeds, compute_loss_func, compute_total_penalty, track,
                                              change_koef=1, iter_num=100)

    # Solution visualisation
    draw_solution(optimal_speeds)

    # Find relation between solar radiation and vehicle speed
    solar_radiation_levels = track.sections.solar_radiation
    draw_speed_solar_radiation_relation(optimal_speeds, solar_radiation_levels)

    # Save params about model
    model_data = energy_manager.compute_energy_levels_full(track, optimal_speeds)
    energy_manager.draw_energy_levels(energy_levels=model_data["levels"],
                                      energy_incomes=model_data["incomes"],
                                      energy_outcomes=model_data["outcomes"])

    model_params = model_data["params"]
    model_params.to_csv("logs/model_params "
                        + str(datetime.datetime.today().strftime("%Y-%m-%d %H-%M-%S"))
                        + ".csv", sep=";")


if __name__ == "__main__":
    main()
