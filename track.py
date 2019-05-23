import datetime
import math
from utils import timeit, check_net_connection
import matplotlib.pyplot as plt
import pandas as pd
from environment_data import compute_solar_radiation, get_weather_params_owm
import copy
from scipy.io import loadmat

DEFAULT_SOLAR_RADIATION = 1000
DEFAULT_CLOUDNESS = 0


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
    Consists of set of points with x, y, z coordinates and their features (solar radiation, distance to previous)
    """
    MAX_SECTION_LENGTH = 20000
    MAX_SLOPE_CHANGE = 0.15

    sections = pd.DataFrame(columns=["length", "length_sum", "slope_angle", "coordinates",
                                     "solar_radiation", "arrival_time"])

    def load_track_from_mat(self, filename):
        pass

    @timeit
    def load_track_from_csv(self, filename):
        track = pd.read_csv(filename, sep=';')
        track = track.iloc[:, 0:3]
        self.sections.coordinates = [[track.iloc[i, 0], track.iloc[i, 1], track.iloc[i, 2]] for i in range(len(track))]
        print("Loaded {} points".format(len(self.sections)))

    @timeit
    def preprocess_track(self):
        self.convert2m()
        # self.compute_length()  # Needed only for plotting
        # self.compute_slope_angle()  # Needed only for plotting

    def convert2m(self):
        self.sections.coordinates = self.sections.coordinates.apply(
            lambda coords: [deg2m(coords[0]), deg2m(coords[1]), coords[2]])

    def compute_length(self):
        length_sum = 0.0
        for i in range(len(self.sections) - 1):
            length = find_distance(
                self.sections.iloc[i].coordinates,
                self.sections.iloc[i + 1].coordinates
            )
            length_sum += length
            self.sections.iloc[i].length = length
            self.sections.iloc[i].length_sum = length_sum

    def compute_slope_angle(self):
        for i in range(len(self.sections) - 1):
            self.sections.iloc[i].slope_angle = find_slope_angle(
                self.sections.iloc[i].coordinates,
                self.sections.iloc[i + 1].coordinates
            )

    @timeit
    def combine_points_to_sections(self, show_info=False):
        print("Before combining {} points".format(len(self.sections)))
        new_sections = pd.DataFrame(columns=["length", "length_sum", "slope_angle", "coordinates",
                                             "solar_radiation", "arrival_time"])
        # Init with first section
        section_start = self.sections.iloc[0].coordinates
        section_end = self.sections.iloc[1].coordinates
        section_dist = find_distance(section_start, section_end)
        for i in range(2, len(self.sections)):
            if show_info: print("\n", i)
            cur_point = self.sections.iloc[i].coordinates

            previous_slope_angle = find_slope_angle(section_start, section_end)
            cur_slope_angle = find_slope_angle(section_end, cur_point)
            slope_angle_diff = abs(previous_slope_angle - cur_slope_angle)
            dist = find_distance(section_end, cur_point)
            if show_info: print("Slope angle diff {}, Dist {}, Section dist + dist {}".format(slope_angle_diff,
                                                                                              dist,
                                                                                              section_dist + dist))

            def add_new_section():  # TODO maybe pass arguments explicitly
                if len(new_sections) == 0:
                    section_dist_sum = 0 + section_dist
                else:
                    section_dist_sum = new_sections.loc[len(new_sections) - 1].length_sum + section_dist
                x = m2deg(section_start[0])
                y = m2deg(section_start[1])
                z = section_start[2]
                # coordinates = list(map(m2deg, section_start))  # TODO make more clear de2m and reverse translations
                new_sections.loc[len(new_sections)] = ([section_dist, section_dist_sum, previous_slope_angle,
                                                          [x, y, z], None, None])

            if slope_angle_diff < self.MAX_SLOPE_CHANGE and section_dist + dist < self.MAX_SECTION_LENGTH:
                if show_info: print("Decision: combining")
                section_dist += dist
                section_end = cur_point
            else:
                if show_info: print("Decision: separating")
                add_new_section()

                # Prepare for next section
                section_start = section_end
                section_end = cur_point
                section_dist = dist

            if i == len(self.sections) - 1:  # Separating last point
                if show_info: print("Separating last section")
                add_new_section()

        self.sections = new_sections
        print("After combining {} sections".format(len(self.sections)))
        # self.sections.to_csv("logs/sections_params "
        #                      + str(datetime.datetime.today().strftime("%Y-%m-%d %H-%M-%S"))
        #                      + ".csv", sep=";")

    @timeit
    def compute_arrival_times(self, start_datetime, drive_time_bounds, speeds):  # TODO call while optimizing
        """Computes time when solar car will arrive to each section"""
        datetime_cur = copy.deepcopy(start_datetime)
        for i in range(len(self.sections)):
            self.sections.at[i, "arrival_time"] = datetime_cur
            seconds = self.sections.iloc[i].length / speeds[i]
            datetime_cur += datetime.timedelta(seconds=seconds)

            if datetime_cur.hour >= drive_time_bounds[1]:
                seconds_exceeded = (datetime_cur.hour - drive_time_bounds[1]) * 3600 \
                                   + datetime_cur.minute * 60 + datetime_cur.second
                datetime_cur += datetime.timedelta(days=1)
                datetime_cur = datetime_cur.replace(hour=drive_time_bounds[0], minute=0, second=0)
                datetime_cur += datetime.timedelta(seconds=seconds_exceeded)

    @timeit
    def compute_weather_params(self):
        net_available = check_net_connection()

        for i in range(len(self.sections)):
            latitude, longitude = self.sections.iloc[i].coordinates[0], self.sections.iloc[i].coordinates[1]
            datetime_cur = self.sections.iloc[i].arrival_time
            if net_available:
                cloudiness = get_weather_params_owm(latitude, longitude, datetime_cur)["clouds"]
                # cloudiness = DEFAULT_CLOUDNESS  # TODO delete
            else:
                cloudiness = DEFAULT_CLOUDNESS

            solar_radiation_raw = compute_solar_radiation(latitude, datetime_cur)

            # Compute final solar radiation
            self.sections.at[i, "solar_radiation"] = solar_radiation_raw * (1 - cloudiness / 100)  # TODO tune formula

    def draw_track_features(self, title=None):
        distance_covered = []
        altitudes = []
        slope_angle = []
        solar_radiation = []
        arrival_time = []
        for index, section in self.sections.iterrows():
            distance_covered.append(section.length_sum / 1000)  # to km
            altitudes.append(section.coordinates[2])
            slope_angle.append(section.slope_angle)
            solar_radiation.append(section.solar_radiation)
            arrival_time.append(section.arrival_time)

        titles = ["Altitudes", "Slope angle", "Solar radiation", "Arrival time"]
        y = [altitudes, slope_angle, solar_radiation, arrival_time]
        fig, axs = plt.subplots(1, 4, figsize=(10, 10))
        for i in range(len(titles)):
            axs[i].grid(True)
            axs[i].set_xlabel("Distance covered (km)")
            axs[i].set_title(titles[i])
            axs[i].plot(distance_covered, y[i])

        if title is not None:
            fig.suptitle(title)

        fig.tight_layout()
        plt.savefig("logs/track_features "
                    + str(datetime.datetime.today().strftime("%Y-%m-%d %H-%M-%S"))
                    + ".png")
        plt.show()
