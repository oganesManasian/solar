import datetime

import scipy.optimize
from utils import timeit
from track import Track
import pandas as pd


@timeit
def exterior_penalty_method(func, penalty_func, x0, args=None,
                            eps=1, tol=1e-3, mu0=1, betta=3, max_step=20,
                            print_info=False):
    """Minimizes function with constraints using exterior penalty method"""
    optimization_step_description = pd.DataFrame(columns=["Step", "MU",
                                                          "Loss function", "Penalty function",
                                                          "MU * penalty function"])
    x = x0
    mu = mu0
    step = 1
    if print_info:
        print("Starting minimization")

    while True:
        if print_info:
            print("Step:", step,
                  "mu:", mu)

        def func_to_minimize(section_speeds: list, track: Track):
                return func(section_speeds, track) + mu * penalty_func(section_speeds, track)

        if print_info:
            print("Starting internal minimization")
        x, success = minimize(func=func_to_minimize,
                              x0=x,
                              args=args,
                              method="L-BFGS-B",
                              tol=tol,
                              print_info=print_info)

        loss_function_value = func(x, args)
        penalty_function_value = penalty_func(x, args)
        mu_x_penalty_function_value = mu * penalty_func(x, args)
        optimization_step_description.loc[len(optimization_step_description)] = (step, mu,
                                                                                 loss_function_value,
                                                                                 penalty_function_value,
                                                                                 mu_x_penalty_function_value)

        if not success:
            print("Internal minimization failed")
            break

        if print_info:
            print("Loss function:", loss_function_value,
                  "\nPenalty function:", penalty_function_value,
                  "\nMU * Penalty function", mu_x_penalty_function_value)

        if mu_x_penalty_function_value < eps:
            if print_info:
                print("Successful optimization")
            break
        else:
            mu *= betta
            step += 1

        if step > max_step:
            print("Exceeded maximum number of iterations")
            break

    optimization_step_description.to_csv("logs/optimization "
                                         + str(datetime.datetime.today().strftime("%Y-%m-%d %H-%M-%S"))
                                         + ".csv", sep=";")
    return x


@timeit
def minimize(func, x0, method="L-BFGS-B", args=None, tol=1e-3, print_info=False):
    """Minimizes func without constraints using chosen method with scipy.optimize package"""
    res = scipy.optimize.minimize(func, x0, method=method, args=args, tol=tol, options={'disp': False},
                                  callback=MinimizeCallback(func, args, print_info))

    if print_info:
        print("     Optimization was successful?", res.success)
        if not res.success:
            print(res.message)
        # print("Objective function value:", res.fun)
        print("     Made iterations", res.nit)

    return res.x, res.success


class MinimizeCallback(object):
    """Prints loss value and penalty value at each iteration"""
    def __init__(self, loss_func, args, print_info):
        self.loss_func = loss_func
        self.args = args
        self.print_info = print_info
        self.iter = 1

    def __call__(self, x):
        # ...
        if self.print_info:
            print("     ", self.iter, "iteration - done")
        self.iter += 1
