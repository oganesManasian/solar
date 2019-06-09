import optimization_methods
from loss_func import compute_loss_func, compute_total_penalty
from parameters import START_DATETIME, DRIVE_TIME_BOUNDS
from track import Track
from utils import timeit


@timeit
def multiple_start(func, penalty_func, track, func_type="original",
                   init_points_num=5, show_info=True):
    a = 10
    b = 35
    init_speeds = [b - a + (b - a) / init_points_num * i for i in range(init_points_num + 1)]
    init_vector_pool = [[speed] * len(track.sections) for speed in init_speeds]

    track = Track(max_section_length=30000, max_section_slope_angle=0.15)
    track.load_track_from_csv("data/track_Australia.csv")
    track.combine_points_to_sections()

    results = []
    for i, init_vector in enumerate(range(len(init_vector_pool))):
        track.compute_arrival_times(START_DATETIME, DRIVE_TIME_BOUNDS, init_vector)
        track.compute_weather_params()

        optimal_speeds = optimization_methods.penalty_method(func=func,
                                                             penalty_func=penalty_func,
                                                             x0=init_vector,
                                                             args=track,
                                                             func_type=func_type,
                                                             eps=1,
                                                             tol=1e-3,
                                                             max_step=20,
                                                             show_info=True)
        loss = compute_loss_func(optimal_speeds, track)
        penalty = compute_total_penalty(optimal_speeds, track, func_type="original")
        results.append([loss, penalty, optimal_speeds])

    results.sort(key=lambda res: (res[1], res[0]))
    if show_info:
        print("Best result Loss: {} Penalty {}".format(results[0][0], results[0][1]))
    return results[0][2]


@timeit
def genetic_algorithm(func, penalty_func, track, speed_range=range(15, 20), func_type="original",
                      population_size=10, mutation_rate=5, max_epoch=30, max_epochs_with_similar_result=3,
                      show_info=False):

    def generate_initial_population():
        nonlocal cur_id
        init_candidates = []
        for speed in speed_range:
            cand_params = [speed] * len(track.sections)
            cand_loss = func(cand_params, track)
            cand_penalty = penalty_func(cand_params, track, func_type=func_type)
            cand_id = cur_id
            cur_id += 1
            init_candidates.append([cand_loss, cand_penalty, cand_params, cand_id])
        return init_candidates

    def generate_candidates(parent):
        nonlocal cur_id
        new_candidates = []
        for _ in range(2):
            for change_koef in range(1, mutation_rate):
                cand_params = optimization_methods.random_vector_change(parent[2], change_koef)
                cand_loss = func(cand_params, track)
                cand_penalty = penalty_func(cand_params, track, func_type=func_type)
                cand_id = cur_id
                cur_id += 1
                new_candidates.append([cand_loss, cand_penalty, cand_params, cand_id])
        return new_candidates

    cur_id = 0
    population = generate_initial_population()

    epoch = 1
    epochs_with_similar_result = 0
    while epoch <= max_epoch and epochs_with_similar_result < max_epochs_with_similar_result:
        cur_best_id = population[0][3]

        new_candidates = []
        for parent in population:
            new_candidates += generate_candidates(parent)
        population += new_candidates

        # population.sort(key=lambda candidate: (candidate[1], candidate[0]))
        population.sort(key=lambda candidate: candidate[0] + 100 * candidate[1])

        population = population[:population_size]

        if show_info:
            print("Step: {} Best values: \n"
                  "Loss: {} Penalty: {} Id: {}\n"
                  "Loss: {} Penalty: {} Id: {}\n"
                  "Loss: {} Penalty: {} Id: {}".format(epoch,
                                                       population[0][0], population[0][1], population[0][3],
                                                       population[1][0], population[1][1], population[1][3],
                                                       population[2][0], population[2][1], population[2][3]))
        new_best_id = population[0][3]
        if cur_best_id == new_best_id:
            epochs_with_similar_result += 1
        else:
            epochs_with_similar_result = 0

        epoch += 1

    if show_info:
        print("End of optimization. Best candidate loss: {} penalty: {}".format(population[0][0],
                                                                                population[0][1]))
    return population[0][2]
