import track
import energy_manager
import optimization_methods
import matplotlib.pyplot as plt
from loss_func import loss_func

# Load race track
track = track.Track()
track.load_track_from_mat("data_tracks.mat")
# track.draw_track_altitudes("Track after preprocessing")
# track.sections = track.sections[:10]  # Taking only small part of track for fast tests

# Test that optimization is possible
speeds_test = [1] * len(track.sections)
energy_levels_test = energy_manager.compute_energy_levels(track, speeds_test)
assert (energy_levels_test[-1] >= 0), "Too little energy to cover the distance!"

# Optimize speed
INIT_SPEED = 72 / 3.6  # 20 m/s
init_speeds = [INIT_SPEED] * len(track.sections)
print("Init loss:", loss_func(init_speeds, track))

optimal_speeds = optimization_methods.minimize(loss_func,
                                               init_speeds,
                                               method="L-BFGS-B",
                                               args=(track),
                                               tol=1e-3,
                                               print_info=True)
print("Optimization result:", optimal_speeds)
print("Final loss:", loss_func(optimal_speeds, track))

# Solution analysis
speeds = list(optimal_speeds[:])
speeds.append(speeds[-1])
# plt.plot(range(len(speeds)), speeds, 'bo')
plt.step(range(len(speeds)), speeds, where='post')
plt.grid()
plt.title("Optimal speed")
plt.xlabel("Sector №")
plt.xticks(range(len(optimal_speeds)), rotation=90)
plt.ylabel("Speed (m/s)")
plt.savefig("Optimal speed.png")  # TODO refactor
plt.show()

model_data = energy_manager.compute_energy_levels_full(track, optimal_speeds)
energy_manager.draw_energy_levels(energy_levels=model_data["levels"],
                                  energy_incomes=model_data["incomes"],
                                  energy_outcomes=model_data["outcomes"])

# Save params log
model_params = model_data["params"]
model_params.to_csv("model_params.csv", sep=";")
