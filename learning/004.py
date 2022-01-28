


# линейная регрессия

import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


buildings = pd.read_csv("http://video.ittensive.com/machine-learning/ashrae/building_metadata.csv.gz")
weather = pd.read_csv("http://video.ittensive.com/machine-learning/ashrae/weather_train.csv.gz")
energy_0 = pd.read_csv("http://video.ittensive.com/machine-learning/ashrae/train.0.0.csv.gz")
print(energy_0.info())

# объединение и получение положительных значений
energy_0 = pd.merge(left = energy_0, right = buildings, how = "left", left_on = "building_id", right_on = "building_id")
energy_0.set_index(["timestamp", "site_id"], inplace = True) 
weather.set_index(["timestamp", "site_id"], inplace = True) 
energy_0 = pd.merge(left = energy_0, right = weather, how = "left", left_index = True, right_index = True)
energy_0.reset_index(inplace = True)
energy_0 = energy_0[energy_0["meter_reading"] > 0]
print(energy_0.head())

# добавление часов в данные 
energy_0["timestamp"] = pd.to_datetime(energy_0["timestamp"])
energy_0["hour"] = energy_0["timestamp"].dt.hour

# разделение данных на обучение и проверку 

energy_0_train, energy_0_test = train_test_split(energy_0, test_size = 0.2)
print(energy_0_train.head())

# модель линейной регрессии

energy_0_train_averages = energy_0_train.groupby("hour").mean()["meter_reading"]

energy_0_train_lr = pd.DataFrame(energy_0_train,
            columns=["meter_reading", "air_temperature", "dew_temperature"])
y = energy_0_train_lr["meter_reading"]
x = energy_0_train_lr.drop(labels=["meter_reading"], axis=1)
model = LinearRegression().fit(x, y)
print(model.coef_, model.intercept_)

# оценка модели
def calculate_model (x):
        meter_reading_log = np.log(x.meter_reading + 1)
        meter_reading_mean = np.log(energy_0_train_averages[x.hour] + 1)
        meter_reading_lr = np.log(1 + x.air_temperature * model.coef_[0] + x.dew_temperature * model.coef_[1] + model.intercept_)
        x["meter_reading_lr_q"] = (meter_reading_log - meter_reading_lr)**2
        x["meter_reading_mean_q"] = (meter_reading_log - meter_reading_mean)**2
        return x

energy_0_test = energy_0_test.apply(calculate_model,
        axis = 1, result_type = "expand")
energy_0_test_lr_rmsle = np.sqrt(energy_0_test["meter_reading_lr_q"].sum() / len(energy_0_test))
energy_0_test_mean_rmsle = np.sqrt(energy_0_test["meter_reading_mean_q"].sum() / len(energy_0_test))
print("Качество среднего: ", energy_0_test_mean_rmsle)
print("Качество линейной регрессии: ", energy_0_test_lr_rmsle)
