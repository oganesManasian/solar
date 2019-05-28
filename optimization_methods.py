import datetime
import scipy.optimize
import energy_manager
from utils import timeit
from track import Track
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


@timeit
def exterior_penalty_method(func, penalty_func, x0, args=None,
                            eps=1, tol=1e-3, mu0=0.1, betta=3, max_step=20,
                            show_info=False):
    """Minimizes function with constraints using exterior penalty method"""
    optimization_description = pd.DataFrame(columns=["Step", "MU",
                                                     "Loss function", "Penalty function",
                                                     "MU * penalty function",
                                                     "Average speed", "Speed vector norm",
                                                     "Speed vector change norm"])
    x = x0
    mu = mu0
    step = 1
    if show_info:
        print("Starting minimization")

    while True:
        if show_info:
            print("Step:", step,
                  "mu:", mu)

        def func_to_minimize(section_speeds: list, track: Track):
            return func(section_speeds, track) + mu * penalty_func(section_speeds, track, continuous=True)

        if show_info:
            print("Starting internal minimization")
        new_x, success = minimize_unconstrained(func=func_to_minimize,
                                                x0=x,
                                                args=args,
                                                method="L-BFGS-B",
                                                tol=tol,
                                                show_info=show_info)

        loss_function_value = func(new_x, args)
        penalty_function_value = penalty_func(new_x, args, continuous=False)
        mu_x_penalty_function_value = mu * penalty_func(new_x, args, continuous=False)
        average_speed = sum([new_x[i] * args.sections.loc[i].length for i in range(len(new_x))]) \
                        / sum(args.sections.length)
        speed_vector_norm = np.linalg.norm(new_x, 1)
        speed_vector_change_norm = np.linalg.norm(new_x - x, 1)
        optimization_description.loc[len(optimization_description)] = (int(step), mu,
                                                                       round(loss_function_value, 3),
                                                                       round(penalty_function_value, 3),
                                                                       round(mu_x_penalty_function_value, 3),
                                                                       np.round(average_speed, 3),
                                                                       round(speed_vector_norm, 3),
                                                                       round(speed_vector_change_norm, 3))

        if not success:
            print("Internal minimization failed")
            break

        if show_info:
            print("Loss function:", loss_function_value,
                  "\nPenalty function:", penalty_function_value,
                  "\nMU * Penalty function", mu_x_penalty_function_value)

        if mu_x_penalty_function_value < eps:
            if show_info:
                print("Successful optimization")
            break
        else:
            mu *= betta
            step += 1

        if step > max_step:
            print("Exceeded maximum number of iterations")
            break

        x = new_x

    optimization_description.to_csv("logs/optimization "
                                    + str(datetime.datetime.today().strftime("%Y-%m-%d %H-%M-%S"))
                                    + ".csv", sep=";")
    return new_x


@timeit
def minimize_unconstrained(func, x0, method="L-BFGS-B", args=None, tol=1e-3, show_info=False):
    """Minimizes func without constraints using chosen method with scipy.optimize package"""
    res = scipy.optimize.minimize(func, x0, method=method, args=args, tol=tol, options={'disp': False},
                                  callback=MinimizeCallback(func, args, show_info))

    if show_info:
        print("     Optimization was successful?", res.success)
        if not res.success:
            print(res.message)
        # print("Objective function value:", res.fun)
        print("     Made iterations", res.nit)

    return res.x, res.success


class MinimizeCallback(object):
    """Prints loss value and penalty value at each iteration"""

    def __init__(self, loss_func, args, show_info):
        self.loss_func = loss_func
        self.args = args
        self.show_info = show_info
        self.iter = 1

    def __call__(self, x):
        # ...
        if self.show_info:
            print("     ", self.iter, "iteration - done")
        self.iter += 1


@timeit
def bruteforce_method(func, penalty_func, speed_range, track, show_info=False):
    from parameters import MAX_SPEED
    sections_with_max_speed = 5
    final_energy_level = []
    race_time = []
    penalty = []
    for speed in speed_range:
        speeds = [MAX_SPEED - 5] * sections_with_max_speed + [speed] * (len(track.sections) - sections_with_max_speed)
        energy_levels = energy_manager.compute_energy_levels(track, speeds)

        final_energy_level.append(energy_levels[-1])
        penalty.append(penalty_func(speeds, track, continuous=False))
        race_time.append(func(speeds, track))

    titles = ["Количество энергии", "Штраф", "Время прохождения марщрута (с)"]
    y = [final_energy_level, penalty, race_time]
    for i in range(len(y)):
        plt.subplot(1, 3, i + 1)
        plt.title(titles[i])
        plt.plot(speed_range, y[i])
        plt.grid()
    figure = plt.gcf()
    figure.set_size_inches(12, 8)
    plt.savefig("logs/Const speed method "
                + str(datetime.datetime.today().strftime("%Y-%m-%d %H-%M-%S"))
                + ".png")  # TODO refactor
    if show_info:
        plt.show()
    else:
        plt.clf()
        plt.close()

    def func_to_minimize(section_speed, track):
        section_speeds = [section_speed] * len(track.sections)
        # return func(section_speeds, track) + penalty_func(section_speeds, track, continuous=True)
        return penalty_func(section_speeds, track, continuous=True)

    possible_ind = [i for i in range(len(final_energy_level)) if final_energy_level[i] >= 0]
    if len(possible_ind) == 0:
        possible_ind = [0, 0]
    bounds = [speed_range[possible_ind[0]], speed_range[possible_ind[-1]]]
    print("Bounds for base speed:", bounds)
    optimal_speed = scipy.optimize.minimize_scalar(func_to_minimize, args=track, bounds=bounds, method="bounded")
    print("Successful minimization?", optimal_speed.success)
    return optimal_speed.x


def random_change(x, func, penalty_func, track, iter_num):
    base_loss_value = func(x, track) + penalty_func(x, track, continuous=False)
    better_x_value_pairs = []
    for i in range(iter_num):
        # rand_vec = (np.random.rand(len(x)) - 0.5) * 2  # TODO test
        rand_vec = np.random.rand(len(x)) - 0.5
        new_x = [x1 + x2 for x1, x2 in zip(x, rand_vec)]
        loss_value = func(new_x, track) + penalty_func(new_x, track, continuous=False)
        if loss_value < base_loss_value:
            better_x_value_pairs.append((new_x, loss_value))

    print("Found {} better x values".format(len(better_x_value_pairs)))
    better_x_value_pairs.sort(key=lambda x_value_pair: x_value_pair[1])
    if len(better_x_value_pairs) > 0:
        print("Best loss value = {}, Base loss value = {}".format(better_x_value_pairs[0][1], base_loss_value))
    return better_x_value_pairs
