from math import sin, cos, radians
import requests
from parameters import UTC_OFFSET, SOLAR_CONSTANT, NORMAL_ATMOUSPHERIC_PRESSURE, TRANSMITTANCE, DEFAULT_CLOUDNESS


def get_solar_radiation(latitude, longitude, datetime):
    solar_radiation = get_solar_radiation_solcast(latitude, longitude, datetime)
    if solar_radiation is None:
        solar_radiation_raw = compute_solar_radiation(latitude, datetime)
        cloudiness = get_weather_params_owm(latitude, longitude, datetime)["clouds"]
        if cloudiness is None:
            cloudiness = DEFAULT_CLOUDNESS
        solar_radiation = solar_radiation_raw * (1 - cloudiness / 100)  # TODO tune formula
    return solar_radiation


def get_solar_radiation_solcast(latitude, longitude, datetime):
    """Get forecast of solar radiation using solcast api (provides 6 day forecast)"""
    json_data = get_solar_radiation_json(latitude, longitude)
    return get_solar_radiation_from_json(json_data, datetime)


def get_solar_radiation_json(latitude, longitude):
    """Request for forecast json"""
    api_key = "-7kCHxclHSKX7bKm6dasBBSBhL-lRGwq"
    api_address = "https://api.solcast.com.au/radiation/forecasts?longitude={}&latitude={}&api_key={}&format=json"
    url = api_address.format(longitude, latitude, api_key)
    json_data = requests.get(url).json()
    return json_data


def get_solar_radiation_from_json(json_data, datetime):
    """Finds most suitable forecast"""
    datetime_cur_utc = datetime + UTC_OFFSET
    if 'forecasts' in json_data.keys():
        forecast_list = json_data['forecasts']
        ind = 0
        while True:
            datetime_forecast_str = forecast_list[ind]['period_end']
            datetime_forecast = datetime.strptime(datetime_forecast_str, "%Y-%m-%dT%H:%M:%S.%f0Z")

            if datetime_forecast > datetime_cur_utc or ind == len(forecast_list) - 1:
                break
            else:
                ind += 1
        return forecast_list[ind]['ghi90']
    else:
        return None


def compute_solar_radiation(latitude, datetime):
    day_of_year = datetime.timetuple().tm_yday
    hour = datetime.hour
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
        return 0


def get_weather_params_owm(latitude, longitude, datetime):
    """Get forecast of cloudiness and temperature using open weather map API (provides 5 day forecast)"""
    api_key = "0c42f7f6b53b244c78a418f4f181282a"
    # api_key_reserve = "b6907d289e10d714a6e88b30761fae22"
    api_address = 'http://api.openweathermap.org/data/2.5/forecast?lat={}&lon={}&appid={}'
    url = api_address.format(latitude, longitude, api_key)
    json_data = requests.get(url).json()

    datetime_cur_utc = datetime + UTC_OFFSET
    if 'list' in json_data.keys():
        forecast_list = json_data['list']
        ind = 0
        while True:
            datetime_forecast_str = forecast_list[ind]['dt_txt']
            datetime_forecast = datetime.strptime(datetime_forecast_str, '%Y-%m-%d %H:%M:%S')
            if datetime_forecast > datetime_cur_utc or ind == len(json_data['list']) - 1:
                break
            else:
                ind += 1

        weather_params = {'temp': json_data['list'][ind]['main']['temp'],
                          'clouds': json_data['list'][ind]['clouds']['all']}
        return weather_params
    else:
        return None
