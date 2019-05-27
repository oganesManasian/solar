from parameters import INIT_SPEED, START_DATETIME, DRIVE_TIME_BOUNDS, OPTIMAL_SPEED_BOUNDS
from track import Track
import energy_manager
import optimization_methods
import matplotlib.pyplot as plt
from loss_func import compute_loss_func, compute_total_penalty
import datetime
import os
import numpy as np

if not os.path.isdir("logs"):
    os.mkdir("logs")

# Load race track
track = Track()
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

# Solve with constant speed
base_speed = optimization_methods.bruteforce_method(compute_loss_func,
                                                    compute_total_penalty,
                                                    np.linspace(OPTIMAL_SPEED_BOUNDS[0], OPTIMAL_SPEED_BOUNDS[1],
                                                                (OPTIMAL_SPEED_BOUNDS[1] - OPTIMAL_SPEED_BOUNDS[0]) * 4),
                                                    track,
                                                    show_info=True)
print("Base speed:", base_speed)
base_speeds = [base_speed] * len(track.sections)
track.compute_arrival_times(START_DATETIME, DRIVE_TIME_BOUNDS, base_speeds)
track.compute_weather_params()
# track.draw_track_features("Constant speed")

# Optimize speed
print("Init loss:", compute_loss_func(base_speeds, track),
      "\nInit penalty:", compute_total_penalty(base_speeds, track, continuous=False))

# def func_to_minimize(section_speeds: list, track: Track):
#     return compute_loss_func(section_speeds, track) + compute_total_penalty(section_speeds, track, continuous=False)
#
#
# optimal_speeds, _ = optimization_methods.minimize_unconstrained(func=func_to_minimize,
#                                                                 x0=init_speeds,
#                                                                 args=(track),
#                                                                 method="L-BFGS-B",
#                                                                 tol=1e-5,
#                                                                 show_info=True)

optimal_speeds = optimization_methods.exterior_penalty_method(func=compute_loss_func,
                                                              penalty_func=compute_total_penalty,
                                                              x0=base_speeds,
                                                              args=track,
                                                              eps=1,
                                                              tol=1e-3,
                                                              show_info=True)

print("Optimization result:", optimal_speeds)
print("Final loss:", compute_loss_func(optimal_speeds, track),
      "\nFinal penalty:", compute_total_penalty(optimal_speeds, track, continuous=False))
print("Total travel time:", round(compute_loss_func(optimal_speeds, track) / 3600, 2), "hours")

# Try to find better solution in point viсinity
optimization_methods.random_change(optimal_speeds, compute_loss_func, compute_total_penalty, track, iter_num=100)

# Solution visualisation
speeds = list(optimal_speeds[:])
speeds.append(speeds[-1])
plt.step(range(len(speeds)), speeds, where='post')
plt.grid()
plt.title("Оптимальная скорость")
plt.xlabel("Номер секции")
plt.xticks(range(len(optimal_speeds)), rotation=90)
plt.ylabel("Скорость (м/с)")
figure = plt.gcf()
figure.set_size_inches(12, 8)
plt.savefig("logs/Optimal speed "
            + str(datetime.datetime.today().strftime("%Y-%m-%d %H-%M-%S"))
            + ".png")  # TODO refactor
plt.show()

# Save params about model
model_data = energy_manager.compute_energy_levels_full(track, optimal_speeds)
energy_manager.draw_energy_levels(energy_levels=model_data["levels"],
                                  energy_incomes=model_data["incomes"],
                                  energy_outcomes=model_data["outcomes"])

model_params = model_data["params"]
model_params.to_csv("logs/model_params "
                    + str(datetime.datetime.today().strftime("%Y-%m-%d %H-%M-%S"))
                    + ".csv", sep=";")

# Find relation between solar radiation and vehicle speed
solar_radiation_levels = model_data["params"].solar_radiation

plt.subplot(2, 1, 1)
plt.title("Оптимальная скорость")
# plt.xlabel("Номер секции")
plt.ylabel("Скорость (м/с)")
# plt.step(range(len(speeds)), speeds, where='post')
plt.plot(range(len(optimal_speeds)), optimal_speeds)
plt.grid()

plt.subplot(2, 1, 2)
# plt.title("Солнечная радиация")
plt.xlabel("Номер секции")
plt.ylabel("Уровень солнечной радиации (Вт/м^2)")
plt.plot(range(len(optimal_speeds)), solar_radiation_levels)
plt.grid()
figure = plt.gcf()
figure.set_size_inches(12, 8)
plt.savefig("logs/Speed and solar radiation relationship "
            + str(datetime.datetime.today().strftime("%Y-%m-%d %H-%M-%S"))
            + ".png")  # TODO refactor
plt.show()
