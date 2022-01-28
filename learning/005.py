

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

buildings = pd.read_csv("http://video.ittensive.com/machine-learning/ashrae/building_metadata.csv.gz")
weather = pd.read_csv("http://video.ittensive.com/machine-learning/ashrae/weather_train.csv.gz")
energy_0 = pd.read_csv("http://video.ittensive.com/machine-learning/ashrae/train.0.0.csv.gz")

energy_0 = pd.merge(left = energy_0, 
        right = buildings, 
        how = "left", 
        left_on = "building_id",
        right_on = "building_id")
energy_0.set_index(["timestamp", "site_id"], inplace = True)
weather.set_index(["timestamp", "site_id"], inplace = True)
energy_0 = pd.merge(left = energy_0,
        right = weather,
        how = "left",
        left_index = True,
        right_index = True)
energy_0.reset_index(inplace = True)
energy_0 = energy_0[energy_0["meter_reading"] > 0]
energy_0["timestamp"] = pd.to_datetime(energy_0["timestamp"])
energy_0["hour"] = energy_0["timestamp"].dt.hour

# очистка данных

energy_0["air_temperature"].fillna(0, inplace = True)
energy_0["dew_temperature"].fillna(0, inplace = True)
energy_0["cloud_coverage"].fillna(0, inplace = True)
energy_0["wind_speed"].fillna(0, inplace = True)
energy_0["precip_depth_1_hr"] = energy_0["precip_depth_1_hr"].apply(lambda x:x if x > 0 else 0)
energy_0_sea_level_pressure_mean = energy_0["sea_level_pressure"].mean()
energy_0["sea_level_pressure"] = energy_0["sea_level_pressure"].apply(lambda x:energy_0_sea_level_pressure_mean if x!=x else x)
print(energy_0.info())

energy_0_train, energy_0_test = train_test_split(energy_0, test_size = 0.2)
print(energy_0_train.head())

# Линейная регрессия по часам 

hours = range(0, 24)
energy_0_train_lr = pd.DataFrame(energy_0_train,
        columns = ["meter_reading", 
            "air_temperature", 
            "dew_temperature", 
            "sea_level_pressure",
            "wind_speed",
            "cloud_coverage",
            "hour"])

energy_0_lr = [[]]*len(hours)
for hour in hours:
    energy_0_lr_hourly = energy_0_train_lr[energy_0_train_lr["hour"] == hour]
    y = energy_0_lr_hourly["meter_reading"]
    x = energy_0_lr_hourly.drop(labels = ["meter_reading",
        "hour"], 
        axis = 1)
    model = LinearRegression().fit(x, y)
    energy_0_lr[hour] = model.coef_
    energy_0_lr[hour] = np.append(energy_0_lr[hour], model.intercept_)
print(energy_0_lr)

# предсказание и оценка

def calculate_model (x):
    model = energy_0_lr[x.hour]
    meter_reading_log = np.log(x.meter_reading + 1)
    meter_reading_lr = np.log(1 + x.air_temperature * model[0] +
            x.dew_temperature * model[1] + 
            x.sea_level_pressure * model[2] +
            x.wind_speed * model[3] + 
            x.cloud_coverage * model[4] + model[5])
    x["meter_reading_lr_q"] = (meter_reading_log - meter_reading_lr)**2
    return x

energy_0_test = energy_0_test.apply(calculate_model,
        axis = 1,
        result_type = "expand")
energy_0_test_lr_rmsle = np.sqrt(energy_0_test["meter_reading_lr_q"].sum() / len(energy_0_test))
print("Качество почасовой линейной регрессии, 5 параметров: ",
        energy_0_test_lr_rmsle, round(energy_0_test_lr_rmsle, 2))



