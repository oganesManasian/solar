import matplotlib.pyplot as plt


from energy_manager import EnergyManager

manager = EnergyManager()
print("Energy level: {} %".format(manager.get_battery_level_in_percent()))


