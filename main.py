import track
import energy_manager
import optimization_methods

# Load/Generate race track
track = track.Track()
track.load_track_from_mat("data_tracks.mat")
# track.draw_track_altitudes("Track after preprocessing")

# import copy
# for dim in [5 + i * 2 for i in range(23)]:
#     print("-----------{}------------".format(dim))
#     cur_track = copy.deepcopy(track)
#     cur_track.sections = cur_track.sections[0:dim]
#
#     # Optimize speed
#     INIT_SPEED = 72/3.6  # 20 m/s
#     init_speeds = [INIT_SPEED] * len(cur_track.sections)
#     print("Init loss:", energy_manager.loss_func(init_speeds, cur_track))
#
#     optimal_speeds = optimization_methods.minimize_scipy(energy_manager.loss_func,
#                                                          init_speeds,
#                                                          args=(cur_track),
#                                                          tol=1e-6)
#     print("Optimization result:", optimal_speeds)


# track.sections = track.sections[0:50]  # Test on small track # TODO delete

# Optimize speed
INIT_SPEED = 72 / 3.6  # 20 m/s
init_speeds = [INIT_SPEED] * len(track.sections)
print("Init loss:", energy_manager.loss_func(init_speeds, track))

optimal_speeds = optimization_methods.minimize(energy_manager.loss_func,
                                               init_speeds,
                                               args=(track),
                                               tol=1e-3,
                                               print_info=True)
print("Optimization result:", optimal_speeds)

# print("Final loss:", energy_manager.loss_func(optimal_speeds, track))

# Solution analysis
model_data = energy_manager.compute_energy_levels_full(track, optimal_speeds)
energy_manager.draw_energy_levels(energy_levels=model_data["levels"],
                                  energy_incomes=model_data["incomes"],
                                  energy_outcomes=model_data["outcomes"])
model_params = model_data["params"]
model_params.to_csv("model_params.csv", sep=";")

# # Start simulation
# speeds = [speed / 3.6 for speed in range(70, 74)]
# losses = []
#
# for speed in speeds:
#     print("Testing with speed = {} m/s".format(speed))
#     section_speeds = [speed] * len(track.sections)
#     loss = energy_manager.loss_func(section_speeds, track)
#     losses.append(loss)
#
# plt.title("Speed - Loss")
# plt.xlabel("Speed")
# plt.ylabel("Loss")
#
# plt.plot(speeds, losses)
#
# plt.grid()
# plt.show()
