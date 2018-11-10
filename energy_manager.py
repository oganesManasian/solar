import math

MAX_BATTER_POWER = 5100  # W * h
MIN_BATTER_POWER = 0  # W * h
SOLAR_RADIATION = 1000  # W / m²
BATTERY_AREA = 4  # m²
EQUIPMENT_POWER = 40  # W


class EnergyManager:
    def __init__(self):
        self.__battery_level = MAX_BATTER_POWER

    def model_energy_income(self):
        """
        Model income of energy according to road parameters during 1 second
        """
        self.__battery_level += self.calculate_energy_income()
        if self.__battery_level > MAX_BATTER_POWER:
            self.__battery_level = MAX_BATTER_POWER

    def model_energy_outcome(self):
        """
        Model outcome of energy according to road parameters during 1 second
        """
        self.__battery_level -= self.calculate_energy_outcome()
        if self.__battery_level < MIN_BATTER_POWER:
            self.__battery_level = MIN_BATTER_POWER

    def calculate_energy_income(self):
        """
        Calculate income of energy according to road parameters during 1 second
        """
        sun_angle = 10  # TODO calculate by geo position
        energy_income_potential = SOLAR_RADIATION * BATTERY_AREA * math.sin(math.radians(sun_angle))
        ECE = 0.9  # Energy conversion efficiency TODO calculate as writen in specification
        energy_income = energy_income_potential * ECE
        return energy_income

    def calculate_energy_outcome(self):
        """
        Calculate outcome of energy according to road parameters during 1 second
        """
        distance = 1
        energy_uniform_motion = 0
        energy_equipment = 0  # EQUIPMENT_POWER * distance / V
        energy_uniform_accelerated_motion = 0
        energy_outcome = energy_uniform_motion \
                         + energy_equipment \
                         + energy_uniform_accelerated_motion
        return energy_outcome

    def get_battery_level_in_percent(self):
        return self.__battery_level / MAX_BATTER_POWER * 100

# __all__ = [EnergyManager]  # List of import
