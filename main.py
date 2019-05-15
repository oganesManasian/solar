from track import Track
import energy_manager
import optimization_methods
import matplotlib.pyplot as plt
from loss_func import compute_loss_func, compute_total_penalty
import datetime
import os
import numpy as np

START_DATE = datetime.date.today()  # datetime.date(2019, 10, 13)
START_TIME = datetime.time(8, 0, 0)
START_DATETIME = datetime.datetime.combine(START_DATE, START_TIME)
DRIVE_TIME_BOUNDS = [8, 17]
INIT_SPEED = 72 / 3.6
OPTIMAL_SPEED_BOUNDS = [15, 30]

if not os.path.isdir("logs"):
    os.mkdir("logs")

# Load race track
track = Track()
track.load_track_from_mat("data_tracks.mat")
track.preprocess_track()
init_speeds = [INIT_SPEED] * len(track.sections)
track.compute_arrival_times(START_DATETIME, DRIVE_TIME_BOUNDS, init_speeds)
track.fill_weather_params()
# track.draw_track_altitudes("Track after preprocessing")
track.sections = track.sections[:10]  # Taking only small part of track for fast tests

# Test that optimization is possible
test_speeds = [1] * len(track.sections)
energy_levels_test = energy_manager.compute_energy_levels(track, test_speeds)
assert (energy_levels_test[-1] >= 0), "Too little energy to cover the distance!"

# Optimize speed
base_speed = optimization_methods.bruteforce_method(compute_loss_func,
                                                    compute_total_penalty,
                                                    np.linspace(OPTIMAL_SPEED_BOUNDS[0], OPTIMAL_SPEED_BOUNDS[1],
                                                                (OPTIMAL_SPEED_BOUNDS[1] - OPTIMAL_SPEED_BOUNDS[0]) * 3),
                                                    track)
print("Base speed:", base_speed)
base_speeds = [base_speed] * len(track.sections)

print("Init loss:", compute_loss_func(base_speeds, track),
      "\nInit penalty:", compute_total_penalty(base_speeds, track))

# def func_to_minimize(section_speeds: list, track: Track):
#     return compute_loss_func(section_speeds, track) + 1 * compute_total_penalty(section_speeds, track)
#
#
# optimal_speeds, _ = optimization_methods.minimize(func=func_to_minimize,
#                                                   x0=init_speeds,
#                                                   args=(track),
#                                                   method="L-BFGS-B",
#                                                   tol=1e-5,
#                                                   print_info=True)

optimal_speeds = optimization_methods.exterior_penalty_method(func=compute_loss_func,
                                                              penalty_func=compute_total_penalty,
                                                              x0=base_speeds,
                                                              args=track,
                                                              eps=1,
                                                              tol=1e-3,
                                                              print_info=True)

print("Optimization result:", optimal_speeds)
print("Final loss:", compute_loss_func(optimal_speeds, track),
      "\nFinal penalty:", compute_total_penalty(optimal_speeds, track))

# Solution visualisation
speeds = list(optimal_speeds[:])
speeds.append(speeds[-1])
plt.step(range(len(speeds)), speeds, where='post')
plt.grid()
plt.title("Optimal speed")
plt.xlabel("Sector №")
plt.xticks(range(len(optimal_speeds)), rotation=90)
plt.ylabel("Speed (m/s)")
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
