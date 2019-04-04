from math import sin, cos, pi, radians
import numpy as np
import requests
from weather import Weather, Unit
import forecastio


SOLAR_CONSTANT = 1367  # W/m2
NORMAL_ATMOUSPHERIC_PRESSURE = 101325
TRANSMITTANCE = 0.75  # transmittance (unitless)


def compute_solar_radiation(latitude, day):
    if -90 < latitude < 90 and 0 < day < 365:

        # considering_hours = range(4, 21)
        considering_hours = range(7, 17)
        radiation_per_hour = np.zeros((24))
        for hour in considering_hours:
            declination = radians(23.45 * sin(radians(360 * (284 + day) / 365)))
            hour_angle = radians(15 * (hour - 12))
            cosz = sin(radians(latitude)) * sin(declination) + cos(radians(latitude)) * cos(declination)\
                   * cos(hour_angle)
            airmass = NORMAL_ATMOUSPHERIC_PRESSURE / (101.3 * cosz)  # optical airmass
            if abs(airmass) > 100:
                airmass = 0

            beam_radiation = cosz * SOLAR_CONSTANT * (TRANSMITTANCE ** airmass)  # on a horizontal surface
            diffuse_radiation = 0.3 * (
                        1.0 - TRANSMITTANCE ** airmass) * SOLAR_CONSTANT * cosz  # on a horizontal surface
            total_radiation = beam_radiation + diffuse_radiation
            radiation_per_hour[hour] = total_radiation  # / 1000 # TODO check that now it returns in Wt not kWt
        return radiation_per_hour
    else:  # Not right input parameters
        return None


def get_weather_weather_module(latitude, longitude):
    weather = Weather(Unit.CELSIUS)
    lookup = weather.lookup_by_latlng(latitude, longitude)
    condition = lookup.condition

    return condition.text


def get_weather_params_owm(latitude, longitude):
    api_key = "0c42f7f6b53b244c78a418f4f181282a"
    #api_key_reserve = "b6907d289e10d714a6e88b30761fae22"
    api_address = 'http://api.openweathermap.org/data/2.5/weather?lat={}&lon={}&appid=' + api_key
    url = api_address.format(latitude, longitude)
    json_data = requests.get(url).json()
    weather_params = {'temp': json_data['main']['temp'],
                      'pressure': json_data['main']['pressure'],
                      'clouds': json_data['clouds']['all']}
    return weather_params


def get_weather_params_darksky(latitude, longitude):
    api_key = "0097352dafff637bf248d40a957b86a0"
    forecast = forecastio.load_forecast(api_key, latitude, longitude)
    return forecast.currently()
