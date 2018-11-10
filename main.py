

from track import Track
from energy_manager import EnergyManager

# Generate race track
track = Track(points_number=10)
track.draw_track()

# Create energy manager
manager = EnergyManager()
print("Energy level: {} %".format(manager.get_battery_level_in_percent()))

# Start simulation
#TODO


