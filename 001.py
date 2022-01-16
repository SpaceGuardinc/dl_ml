import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import rcParams
rcParams['figure.figsize'] = 16,8

buildings = pd.read_csv("http://video.ittensive.com/machine-learning/ashrae/building_metadata.csv.gz")
print (buildings.head())

weather = pd.read_csv("http://video.ittensive.com/machine-learning/ashrae/weather_train.csv.gz")
print (weather.head())

energy_0 = pd.read_csv("http://video.ittensive.com/machine-learning/ashrae/train.0.0.csv.gz")
print (energy_0.head())
energy_0.set_index("timestamp")["meter_reading"].plot()
plt.show()


energy_0 = pd.merge(left = energy_0, right = buildings, how = "left", left_on = "building_id", right_on = "building_id")
print (energy_0.head())

energy_0.set_index(["timestamp", "site_id"], inplace = True)
weather.set_index(["timestamp", "site_id"], inplace = True)

energy_0 = pd.merge(left = energy_0, right = weather, how = "left", left_index = True, right_index = True)
energy_0.reset_index(inplace = True)
print (energy_0.head())

for column in energy_0.columns:
    energy_nulls = energy_0[column].isnull().sum()
    if energy_nulls > 0:
        print (column + ": " + str(energy_nulls))
print (energy_0[energy_0["precip_depth_1_hr"].isnull()])
print (energy_0.info())

energy_0["air_temperature"].fillna(0, inplace = True)
energy_0["cloud_coverage"].fillna(0, inplace = True)
energy_0["dew_temperature"].fillna(0, inplace = True)
energy_0["precip_depth_1_hr"] = energy_0["precip_depth_1_hr"].apply(lambda x:x if x > 0 else 0) 
energy_0_sea_level_pressure_mean = energy_0["sea_level_pressure"].mean()
energy_0["sea_level_pressure"] = energy_0["sea_level_pressure"].apply(lambda x: energy_0_sea_level_pressure_mean if x != x else x)
energy_0_wind_direction_mean = energy_0["wind_direction"].mean()
energy_0["wind_direction"] = energy_0["wind_direction"].apply(lambda x: energy_0_wind_direction_mean if x != x else x)
energy_0.info()
