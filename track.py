import random
import math
import matplotlib.pyplot as plt


# TODO shadow TrackPoint class from export
class TrackPoint:
    def __init__(self, height, x, y):
        self.height = height
        self.x = x
        self.y = y


def get_distance_between_points(pnt1, pnt2):
    return math.sqrt((pnt1.y - pnt2.y) ** 2 + (pnt1.x - pnt2.x) ** 2)


class Track:
    """
    Class for race track.
    Consists of set of points with height and x, y coordinates
    """

    def __init__(self, points_number):
        self.__track_points = [TrackPoint(50, 0, 0)]  # Track is list of points
        for _ in range(points_number - 1):
            height = random.randint(0, 100)
            x = random.randint(self.__track_points[-1].x, self.__track_points[-1].x + 50)
            y = random.randint(self.__track_points[-1].y, self.__track_points[-1].y + 50)
            self.__track_points.append(TrackPoint(height, x, y))

    def draw_track(self):
        X = []
        Y = []
        for pnt in self.__track_points:
            X.append(pnt.x)
            Y.append(pnt.y)

        plt.figure()
        plt.title("Track")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid()
        plt.plot(X, Y)
        plt.plot(X[0], Y[0], "go", label="start")
        plt.plot(X[-1], Y[-1], "ro", label="end")
        plt.legend()
        plt.show()
