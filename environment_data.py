import os
import pickle
from math import sin, cos, radians
import requests
import forecastio

SOLAR_CONSTANT = 1367  # W/m2
NORMAL_ATMOUSPHERIC_PRESSURE = 101325
TRANSMITTANCE = 0.75  # transmittance (unitless)


def compute_solar_radiation(latitude, datetime_cur):
    day_of_year = datetime_cur.timetuple().tm_yday
    hour = datetime_cur.hour
    if -90 <= latitude <= 90 and 0 < day_of_year <= 365 and 8 <= hour <= 17:
        declination = radians(23.45 * sin(radians(360 * (284 + day_of_year) / 365)))
        hour_angle = radians(15 * (hour - 12))
        cosz = sin(radians(latitude)) * sin(declination) + cos(radians(latitude)) * cos(declination) * cos(hour_angle)
        airmass = NORMAL_ATMOUSPHERIC_PRESSURE / (101.3 * cosz)  # optical airmass
        if abs(airmass) > 100:
            airmass = 0

        beam_radiation = cosz * SOLAR_CONSTANT * (TRANSMITTANCE ** airmass)  # on a horizontal surface
        diffuse_radiation = 0.3 * (
                1.0 - TRANSMITTANCE ** airmass) * SOLAR_CONSTANT * cosz  # on a horizontal surface
        total_radiation = beam_radiation + diffuse_radiation

        return total_radiation
    else:  # Not right input parameters
        return None


def get_weather_params_owm(latitude, longitude, datetime):
    """Get 5 day forecast of cloudiness and temperature using open weather map API"""
    if os.path.isfile("weather"):
        json_data = pickle.load("weather")
    else:
        api_key = "0c42f7f6b53b244c78a418f4f181282a"
        # api_key_reserve = "b6907d289e10d714a6e88b30761fae22"
        api_address = 'http://api.openweathermap.org/data/2.5/forecast?lat={}&lon={}&appid=' + api_key
        url = api_address.format(latitude, longitude)
        json_data = requests.get(url).json()
        pickle.dump(json_data, "weather")

    ind = 0
    while True:
        datetime_forecast_str = json_data['list'][ind]['dt_txt']
        datetime_forecast = datetime.strptime(datetime_forecast_str, '%Y-%m-%d %H:%M:%S')
        if datetime_forecast > datetime or ind == len(json_data['list']) - 1:
            break
        else:
            ind += 1

    weather_params = {'temp': json_data['list'][ind]['main']['temp'],
                      'clouds': json_data['list'][ind]['clouds']['all']}
    return weather_params
