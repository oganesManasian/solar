import scipy.optimize
import skopt
from utils import timeit


@timeit
def minimize(func, x0, lib="scipy", method="BFGS", args=None, tol=1e-3, print_info=False):
    if lib == "scipy":
        return minimize_scipy(func, x0, method=method, args=args, tol=tol, print_info=print_info)
    elif lib == "scikit":
        return minimize_scikit(func, x0)
    else:
        print("Chose unavailable minimization package")
        return None


def minimize_scipy(func, x0, method="BFGS", args=None, tol=1e-3, print_info=False):
    res = scipy.optimize.minimize(func, x0, method=method, args=args, tol=tol)
    if print_info:
        print("Optimization was successful?", res.success)
        if not res.success:
            print(res.message)
        print("Objective function value:", res.fun)
        print("Made iterations", res.nit)
        # print("X", res.x) Because it is already printed in main
    return res.x


def minimize_scikit(func, x0):
    raise NotImplementedError
    # res = skopt.gp_minimize(func, [(-2.0, 2.0)])
    # return res
