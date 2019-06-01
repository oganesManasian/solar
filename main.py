from parameters import INIT_SPEED, START_DATETIME, DRIVE_TIME_BOUNDS, OPTIMAL_SPEED_BOUNDS
from track import Track
import energy_manager
import optimization_methods
from loss_func import compute_loss_func, compute_total_penalty
import datetime
import os
import numpy as np
from utils import draw_solution, draw_speed_solar_radiation_relation

# font = {'family': 'normal',
#         # 'weight': 'bold',
#         'size': 14}
# plt.rc('font', **font)

if not os.path.isdir("logs"):
    os.mkdir("logs")

# Load race track
track = Track(max_section_length=30000, max_section_slope_angle=0.15)
track.load_track_from_csv("data/track_Australia.csv")
# track.preprocess_track()
# track.draw_track_altitudes("График высот маршрута до предобработки")
track.combine_points_to_sections(show_info=False)
# track.draw_track_altitudes("График высот маршрута после предобработки")

init_speeds = [INIT_SPEED] * len(track.sections)
track.compute_arrival_times(START_DATETIME, DRIVE_TIME_BOUNDS, init_speeds)
track.compute_weather_params()
# track.draw_track_features("Ключевые параметры маршрута")

# track.sections = track.sections[:10]  # Taking only small part of track for speeding up tests

# Test that optimization is possible
test_speeds = [1] * len(track.sections)
energy_levels_test = energy_manager.compute_energy_levels(track, test_speeds)
assert (energy_levels_test[-1] >= 0), "Too little energy to cover the distance!"

# Find initial approximation
base_speed_vector = optimization_methods.find_initial_approximation_grid(compute_loss_func,
                                                                         compute_total_penalty,
                                                                         track,
                                                                         # low_speed_range=speed_range
                                                                         high_speed_range=range(25, 35),
                                                                         low_speed_range=range(15, 20),
                                                                         n_range=range(1, int(
                                                                             np.ceil(len(track.sections) * 0.1))),
                                                                         )
track.compute_arrival_times(START_DATETIME, DRIVE_TIME_BOUNDS, base_speed_vector)
track.compute_weather_params()
# track.draw_track_features("Constant speed")

# Optimize speed
print("Init loss:", compute_loss_func(base_speed_vector, track),
      "\nInit penalty:", compute_total_penalty(base_speed_vector, track, continuous=False))

optimal_speeds = optimization_methods.exterior_penalty_method(func=compute_loss_func,
                                                              penalty_func=compute_total_penalty,
                                                              x0=base_speed_vector,
                                                              args=track,
                                                              eps=1,
                                                              tol=1e-3,
                                                              max_step=12,
                                                              show_info=True)

print("Optimization result:", optimal_speeds)
print("Final loss:", compute_loss_func(optimal_speeds, track),
      "\nFinal penalty:", compute_total_penalty(optimal_speeds, track, continuous=False))
print("Total travel time:", round(compute_loss_func(optimal_speeds, track) / 3600, 2), "hours")

# Try to find better solution in point viсinity
optimization_methods.random_change(optimal_speeds, compute_loss_func, compute_total_penalty, track, iter_num=100)

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
