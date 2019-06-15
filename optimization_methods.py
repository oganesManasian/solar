import datetime
import scipy.optimize
from utils import timeit
from track import Track
import pandas as pd
import numpy as np
from parameters import START_DATETIME, DRIVE_TIME_BOUNDS


@timeit
def penalty_method(func, penalty_func, x0, args=None,
                   func_type="original", eps=1, tol=1e-3, mu0=1, betta=3, max_step=20,
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

        def func_to_minimize(speed_vector: list, track: Track):
            return func(speed_vector, track) + mu * penalty_func(speed_vector, track, func_type=func_type)

        if show_info:
            print("Starting internal minimization")
        new_x, success = minimize_unconstrained(func=func_to_minimize,
                                                x0=x,
                                                args=args,
                                                method="L-BFGS-B",
                                                tol=tol,
                                                show_info=show_info)

        args.compute_arrival_times(START_DATETIME, DRIVE_TIME_BOUNDS, new_x)
        args.compute_weather_params()

        loss_function_value = func(new_x, args)
        penalty_function_value = penalty_func(new_x, args, func_type=func_type)
        mu_x_penalty_function_value = mu * penalty_func(new_x, args, func_type=func_type)
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
    res = scipy.optimize.minimize(func, x0, method=method, args=args, tol=tol,
                                  # jac=compute_loss_func_gradient,
                                  options={'disp': False}, callback=MinimizeCallback(func, args, show_info))
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
        if self.show_info:
            print("Iteration:", self.iter, "Loss function value:", round(self.loss_func(x, self.args), 3))
        self.iter += 1


@timeit
def grid_search(func, penalty_func, track, func_type="original",
                high_speed_range=range(35, 36), low_speed_range=range(15, 20), n_range=range(0, 1)):
    """Grid (1D, 2D, 2D) search of best speed vector"""

    def build_speed_vector(high_speed, low_speed, sections_with_high_speed_num):
        return [high_speed] * sections_with_high_speed_num \
               + [low_speed] * (len(track.sections) - sections_with_high_speed_num)

    results = []
    for n in n_range:
        for high_speed in high_speed_range:
            for low_speed in low_speed_range:
                speed_vector = build_speed_vector(high_speed, low_speed, n)
                value = func(speed_vector, track) + penalty_func(speed_vector, track, func_type=func_type)
                results.append([value, [high_speed, low_speed, n]])

    results.sort(key=lambda x: x[0])
    best_value, best_x = results[0]
    print("Best result:", best_value, "Best configuration:", best_x)
    return build_speed_vector(*best_x)


@timeit
def random_change_method(x, func, penalty_func, track, func_type="original",
                         change_koef=1, iter_num=100, show_info=True):
    """Random change of speed vector to gain better loss function value"""
    base_loss_value = func(x, track) + penalty_func(x, track, func_type=func_type)
    better_x_value_pairs = []
    for i in range(iter_num):
        new_x = random_vector_change(x, change_koef)
        loss_value = func(new_x, track) + penalty_func(new_x, track, func_type=func_type)
        if loss_value < base_loss_value:
            better_x_value_pairs.append((new_x, loss_value))
    if show_info:
        print("Found {} better x values".format(len(better_x_value_pairs)))
    better_x_value_pairs.sort(key=lambda x_value_pair: x_value_pair[1])
    if len(better_x_value_pairs) > 0:
        if show_info:
            print("Best loss value = {}, Base loss value = {}".format(better_x_value_pairs[0][1], base_loss_value))
        return better_x_value_pairs[0][0]
    else:
        return None


def random_vector_change(vec, change_koef):
    rand_vec = (np.random.rand(len(vec)) - 0.5) * change_koef
    return [x1 + x2 for x1, x2 in zip(vec, rand_vec)]
