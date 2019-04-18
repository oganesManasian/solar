import scipy.optimize
import skopt
from utils import timeit


@timeit
def minimize(func, x0, method="BFGS", args=None, tol=1e-3, print_info=False):
    """Minimizes func using chosen method with scipy.optimize package"""
    res = scipy.optimize.minimize(func, x0, method=method, args=args, tol=tol, options={'disp': False},
                                  callback=MinimizeCallback(func, args))
    if print_info:
        print("Optimization was successful?", res.success)
        if not res.success:
            print(res.message)
        # print("Objective function value:", res.fun)
        print("Made iterations", res.nit)
        # print("X", res.x) Because it is already printed in main
    return res.x


class MinimizeCallback(object):
    """Prints loss value and penalty value at each iteration"""
    def __init__(self, loss_func, args):
        self.loss_func = loss_func
        self.args = args
        self.iter = 0

    def __call__(self, x):
        print("Iteration: {0:3d} Penalty value: {1:8.2f} Loss: {2:8.2f}"
              .format(self.iter,
                      self.loss_func(section_speeds=x, track=self.args, compute_only_penalty=True),
                      self.loss_func(section_speeds=x, track=self.args, compute_only_penalty=False)))
        self.iter += 1
