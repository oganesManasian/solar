import time
import requests


def timeit(method):
    def timed(*args, **kwards):
        start_time = time.time()
        result = method(*args, **kwards)
        elapsed_time = time.time() - start_time
        print(method.__name__, "took", elapsed_time)
        return result

    return timed


def nonnegative(method):
    color_red = '\33[31m'
    color_end = '\33[0m'

    def func(*args, **kwards):
        result = method(*args, **kwards)
        if result < 0:
            print(color_red + "WARNING" + color_end)
            print("Negative result = ", result, method.__name__)
            print("Args", args)
        return result
    return func


def print_res_if_debug(debug):
    def real_decorator(func):
        def inner(*args, **kwargs):
            res = func(*args, **kwargs)
            if debug:
                print("Function:", func.__name__, "\n Args:", args, "\n  Result:", res)
            return res
        return inner
    return real_decorator


def check_net_connection():
    url = 'http://www.google.com/'
    timeout = 5
    try:
        _ = requests.get(url, timeout=timeout)
        return True
    except requests.ConnectionError:
        print("Internet connection unavailable")
    return False
