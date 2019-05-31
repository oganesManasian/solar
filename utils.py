import datetime
import time
import requests
import matplotlib.pyplot as plt

def timeit(method):
    def timed(*args, **kwards):
        start_time = time.time()
        result = method(*args, **kwards)
        elapsed_time = time.time() - start_time
        print(method.__name__, "took", round(elapsed_time, 2), "s.")
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


def draw_solution(optimal_speeds, title=""):
    speeds = list(optimal_speeds[:])
    speeds.append(speeds[-1])
    plt.step(range(len(speeds)), speeds, where='post')
    plt.grid()
    plt.title("Оптимальная скорость" + " " + title)
    plt.xlabel("Номер секции")
    # plt.xticks(range(len(optimal_speeds)), rotation=90)
    plt.xticks(ticks=[v for v in range(len(optimal_speeds)) if v % 5 == 0],
               labels=[v for v in range(len(optimal_speeds)) if v % 5 == 0],
               rotation=90)
    plt.ylabel("Скорость (м/с)")
    figure = plt.gcf()
    figure.set_size_inches(12, 8)
    plt.tight_layout()
    plt.savefig("logs/Optimal speed "
                + str(datetime.datetime.today().strftime("%Y-%m-%d %H-%M-%S"))
                + ".png")
    plt.show()


def draw_speed_solar_radiation_relation(optimal_speeds, solar_radiation_levels):
    plt.subplot(2, 1, 1)
    plt.title("Оптимальная скорость")
    # plt.xlabel("Номер секции")
    plt.ylabel("Скорость (м/с)")
    # plt.step(range(len(speeds)), speeds, where='post')
    plt.plot(range(len(optimal_speeds)), optimal_speeds)
    plt.grid()

    plt.subplot(2, 1, 2)
    # plt.title("Солнечная радиация")
    plt.xlabel("Номер секции")
    plt.ylabel("Уровень солнечной радиации (Вт/м^2)")
    plt.plot(range(len(optimal_speeds)), solar_radiation_levels)
    plt.grid()
    figure = plt.gcf()
    figure.set_size_inches(12, 8)
    plt.savefig("logs/Speed and solar radiation relationship "
                + str(datetime.datetime.today().strftime("%Y-%m-%d %H-%M-%S"))
                + ".png")

    plt.tight_layout()
    plt.show()
