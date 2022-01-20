


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

#отсечение пустых дней и выделение часов из значения времени
energy_0 = pd.read_csv("http://video.ittensive.com/machine-learning/ashrae/train.0.0.csv.gz")
energy_0 = energy_0[energy_0["meter_reading"]>0]
energy_0["timestamp"] = pd.to_datetime(energy_0["timestamp"])
energy_0["hour"] = energy_0["timestamp"].dt.hour
print(energy_0.head())

# выделение 20% на проверку данных, остальное остается на обучение
energy_0_train, energy_0_test = train_test_split(energy_0, test_size = 0.2)
print(energy_0_train.head())

# среднее и медианное значение потребление энергии по часам
energy_0_train_hours = energy_0_train.groupby("hour")
energy_0_train_averages = pd.DataFrame(
        {"Среднее": energy_0_train_hours.mean()["meter_reading"],
        "Медианное": energy_0_train_hours.median()["meter_reading"]})
print(energy_0_train_averages)

# Для вычисления метрики создадим шесть новых столбцов в тестовом наборе данных: с логарифмом значения метрики, предсказанием по среднему и по медиане, а также с квадратом разницы предсказаний и логарифма значения. Последний столбец добавим, чтобы сравнить предсказание с его отсутствием - нулями в значениях.
def calculate_model (x):
    meter_reading_log = np.log(x.meter_reading + 1)
    meter_reading_mean = np.log(energy_0_train_averages["Среднее"][x.hour] + 1)
    meter_reading_median = np.log(energy_0_train_averages["Медианное"][x.hour] + 1)
    x["meter_reading_mean_q"] = (meter_reading_log - meter_reading_mean)**2
    x["meter_reading_median_q"] = (meter_reading_log - meter_reading_median)**2
    x["meter_reading_zero_q"] = (meter_reading_log)**2
    return x

energy_0_test = energy_0_test.apply(calculate_model, axis = 1, result_type = "expand")
print(energy_0_test.head())

# просуммируем квадраты расхождений, разделим их на количество значений и извлечем квадратный корень
energy_0_test_median_rmsle = np.sqrt(energy_0_test["meter_reading_mean_q"].sum() / len(energy_0_test))
energy_0_test_mean_rmsle = np.sqrt(energy_0_test["meter_reading_mean_q"].sum() / len(energy_0_test))
energy_0_test_zero_rmsle = np.sqrt(energy_0_test["meter_reading_zero_q"].sum() / len(energy_0_test))
print ("Качество медианы:", energy_0_test_median_rmsle)
print ("Качество среднего:", energy_0_test_mean_rmsle)
print ("Качество нуля:", energy_0_test_zero_rmsle)
