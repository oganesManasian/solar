import random
import math
from utils import timeit, check_net_connection
import matplotlib.pyplot as plt
import pandas as pd
from environment_data import compute_solar_radiation, get_weather_params_owm

# (X, Y, Z) = range(3)  # Global variables are evil


def find_distance(start_pnt, end_pnt):
    dx = end_pnt[0] - start_pnt[0]
    dy = end_pnt[1] - start_pnt[1]
    return math.sqrt(dx ** 2 + dy ** 2)


def find_slope_angle(start_pnt, end_pnt):
    dx = end_pnt[0] - start_pnt[0]
    dy = end_pnt[1] - start_pnt[1]
    dz = end_pnt[2] - start_pnt[2]
    return math.atan(dz / math.sqrt(dx ** 2 + dy ** 2))


def deg2m(deg):
    # TODO make more precise:
    # https://gis.stackexchange.com/questions/61924/python-gdal-degrees-to-meters-without-reprojecting
    return deg * 111 * 1000


def m2deg(m):
    return m / 111 / 1000


class Track:
    """
    Class for race's track.
    Consists of set of points with x, y, z coordinates
    """
    MAX_SECTION_LENGTH = 10000
    MAX_SLOPE_CHANGE = 0.05

    track_points = []  # Track is list of points
    # sections_length = []
    # sections_length_sum = [0]
    # sections_slope_angle = []
    sections = pd.DataFrame(columns=["length", "length_sum", "slope_angle", "coordinates",
                                     "solar_radiation", "cloudiness"])

    def generate_track(self, sections_number):
        points_number = sections_number + 1
        self.track_points.clear()
        self.track_points = [(0, 0, 0)]
        for _ in range(points_number - 1):
            x = random.randint(self.track_points[-1][2], self.track_points[-1][2] + 100)
            y = -1
            z = random.randint(self.track_points[-1][2] - 20, self.track_points[-1][2] + 20)
            self.track_points.append((x, y, z))

    def generate_simple_track(self, sections_number):
        points_number = sections_number + 1
        self.track_points.clear()
        self.track_points = [(0, 0, 0)]
        for _ in range(points_number - 1):
            x = self.track_points[-1][0] + 1
            y = 0
            z = self.track_points[-1][2]
            self.track_points.append((x, y, z))

    def load_track_from_mat(self, filename):
        from scipy.io import loadmat
        data_tracks = loadmat(filename)
        self.track_points.clear()
        self.track_points = data_tracks["data1"]
        print("Loaded {} points".format(len(self.track_points)))
        self.preprocess_track()

    @timeit
    def preprocess_track(self):
        self.convert2m()
        self.combine_points_to_sections(debug=False)
        self.fill_weather_params()
        assert sum(self.sections.length) == 988984.1234339202, "Lost track part"   # TODO delete

    def convert2m(self):
        for i in range(len(self.track_points)):
            self.track_points[i][0] = deg2m(self.track_points[i][0])
            self.track_points[i][1] = deg2m(self.track_points[i][1])

    @timeit
    def combine_points_to_sections(self, debug=False):
        print("Before combining {} points".format(len(self.track_points)))
        # Init with first section
        section_start = self.track_points[0]
        section_end = self.track_points[1]
        section_dist = find_distance(section_start, section_end)
        for i in range(2, len(self.track_points)):
            if debug: print("\n", i)
            cur_point = self.track_points[i]

            previous_slope_angle = find_slope_angle(section_start, section_end)
            cur_slope_angle = find_slope_angle(section_end, cur_point)
            slope_angle_diff = abs(previous_slope_angle - cur_slope_angle)
            dist = find_distance(section_end, cur_point)
            if debug: print("Slope angle diff {}, Dist {}, Section dist + dist {}".format(slope_angle_diff,
                                                                                          dist,
                                                                                          section_dist + dist))

            def add_new_section():  # TODO maybe pass arguments explicitly
                if len(self.sections) == 0:
                    section_dist_sum = 0 + section_dist
                else:
                    section_dist_sum = self.sections.loc[len(self.sections) - 1].length_sum + section_dist
                x = m2deg(section_start[0])
                y = m2deg(section_start[1])
                z = section_start[2]
                # coordinates = list(map(m2deg, section_start))  # TODO make more clear de2m and reverse translations
                self.sections.loc[len(self.sections)] = ([section_dist, section_dist_sum, previous_slope_angle,
                                                          [x, y, z], None, None])

            if slope_angle_diff < self.MAX_SLOPE_CHANGE and section_dist + dist < self.MAX_SECTION_LENGTH:
                if debug: print("Decision: combining")
                section_dist += dist
                section_end = cur_point
            else:
                if debug: print("Decision: separating")
                add_new_section()

                # Prepare for next section
                section_start = section_end
                section_end = cur_point
                section_dist = dist

            if i == len(self.track_points) - 1:  # Separating last point
                if debug: print("Separating last section")
                add_new_section()

        print("After combining {} sections".format(len(self.sections)))

    @timeit
    def fill_weather_params(self):
        day = 20  # TODO get from datetime
        net_available = check_net_connection()

        for i in range(len(self.sections)):
            latitude, longitude = self.sections.iloc[i].coordinates[0], self.sections.iloc[i].coordinates[1]

            if False: #net_available:  # TODO turn on check
                cloudiness = get_weather_params_owm(latitude, longitude)["clouds"]
            else:
                cloudiness = 0  # TODO Tune default parameter
            self.sections.at[i, "cloudiness"] = [cloudiness] * 24  # each hour of day

            # TODO compute real value using cloudiness
            # self.sections.at[i, "solar_radiation"] = compute_solar_radiation(latitude, day)
            self.sections.at[i, "solar_radiation"] = [1000] * 24  # TODO delete

    def draw_track_xy(self, title="Track XY"):
        if self.track_points is None:
            print("Track points are not defined")
            return

        x = []
        y = []
        for i in range(len(self.track_points)):
            x.append(self.track_points[i][0])
            y.append(self.track_points[i][1])

        plt.figure()
        plt.title(title)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid()
        plt.plot(x, y)
        plt.plot(x[0], y[0], "go", label="start")
        plt.plot(x[1:-1], y[1:-1], "yo")
        plt.plot(x[-1], y[-1], "ro", label="end")
        plt.legend()
        plt.show()

    def draw_track_altitudes(self, title="Track altitudes"):
        x = list()
        z = list()
        for index, row in self.sections.iterrows():
            x.append(row.length_sum)
            z.append(row.coordinates[2])

        plt.figure()
        plt.title(title)
        plt.xlabel("Distance travelled (m)")
        plt.ylabel("Altitude")
        plt.grid()
        plt.plot(x, z)
        plt.plot(x[0], z[0], "go", label="start")
        plt.plot(x[-1], z[-1], "ro", label="end")
        plt.legend()
        plt.show()
