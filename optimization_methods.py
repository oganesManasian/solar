import scipy.optimize
import skopt
from utils import timeit


@timeit
def minimize(func, x0, method="scipy", args=None, tol=1e-3, print_info=False):
    if method == "scipy":
        return minimize_scipy(func, x0, args=args, tol=tol, print_info=print_info)
    elif method == "scikit":
        return minimize_scikit(func, x0)
    else:
        print("Chose unavailable minimization package")
        return None


def minimize_scipy(func, x0, args=None, tol=1e-3, print_info=False):
    res = scipy.optimize.minimize(func, x0, args=args, tol=tol)
    if print_info:
        print("Optimization was successful?", res.success)
        if not res.success:
            print(res.message)
        print("Objective function value:", res.fun)
        print("Made iterations", res.nit)
        print("X", res.x)
    return res.x


def minimize_scikit(func, x0):
    # res = skopt.gp_minimize(func, [(-2.0, 2.0)])
    # return res
    return None
