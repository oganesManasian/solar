import datetime

# Schedule
# UTC_OFFSET = datetime.datetime.utcnow() - datetime.datetime.now()
UTC_OFFSET = datetime.timedelta(hours=-10.5)  # Adelaide utc offset
START_DATE = datetime.date.today() + datetime.timedelta(days=1)  # Tomorrow competition start date
# START_DATE = datetime.date(2019, 10, 13)  # real competition start date
# START_DATE = datetime.date(day=9, month=6, year=2019) + datetime.timedelta(days=1)  # test competition start date
START_TIME = datetime.time(8, 30, 0)
START_DATETIME = datetime.datetime.combine(START_DATE, START_TIME)
DRIVE_TIME_BOUNDS = [8, 17]

# Optimization
INIT_SPEED = 17
OPTIMAL_SPEED_BOUNDS = [15, 25]
MAX_SPEED = 40
CONSTANT_PENALTY_VALUE = 3600

# Track
# 104 sections
MAX_SECTION_LENGTH = 30000
MAX_SLOPE_CHANGE = 0.15
# 166 sections
# MAX_SECTION_LENGTH = 20000
# MAX_SLOPE_CHANGE = 0.1
# 318 sections
# MAX_SECTION_LENGTH = 10000
# MAX_SLOPE_CHANGE = 0.1

# Battery
ENERGY_LEVEL_PERCENT_MAX = 100
ENERGY_LEVEL_PERCENT_MIN = 0
BATTER_CHARGE_MAX = 5100 * 3600
BATTER_CHARGE_MIN = 0  # W * h
EFFICIENCY_BATTERY = 0.98

# Environment
DEFAULT_SOLAR_RADIATION = 1000
DEFAULT_CLOUDNESS = 10
GRAVITY_ACCELERATION = 9.81
FRICTION_RESISTANCE_RATE = 0.0025
AIR_DENSITY = 1.18
SOLAR_CONSTANT = 1367  # W/m2
NORMAL_ATMOUSPHERIC_PRESSURE = 101325
TRANSMITTANCE = 0.75  # transmittance (unitless)

# Vehicle
VEHICLE_FRONT_AREA = 0.6  # m²
FRONTAL_DENSITY_RATE = 0.15
VEHICLE_PANEL_AREA = 4  # m²
VEHICLE_EQUIPMENT_POWER = 40  # W
VEHICLE_WEIGHT = 385  # kg
EFFICIENCY_INCOME = 0.2 * 0.985
EFFICIENCY_OUTCOME = 0.94
