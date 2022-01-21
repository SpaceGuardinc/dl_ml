

# EDA : Исследование зависимостей

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
for type_ in ["f2", "f4"]:
        print (np.finfo(type_))
for type_ in ["i1", "i2", "i4"]:
    print (np.iinfo(type_))

buildings = pd.read_csv("http://video.ittensive.com/machine-learning/ashrae/building_metadata.csv.gz")
weather = pd.read_csv("http://video.ittensive.com/machine-learning/ashrae/weather_train.csv.gz")
energy = pd.read_csv("http://video.ittensive.com/machine-learning/ashrae/train.0.csv.gz")

# потребление памяти
print("Строения:", buildings.memory_usage().sum() / 1024**2, "MB")
print("Погода:", weather.memory_usage().sum() / 1024**2, "MB")
print("Энергия:", energy.memory_usage().sum() / 1024**2, "MB")

# функция оптимизации памяти


def reduce_mem_usage (df):
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if str(col_type)[:5] == "float":
            c_min = df[col].min()
            c_max = df[col].max()
            if c_min > np.finfo("f2").min and c_max < np.finfo("f2").max:
                df[col] = df[col].astype(np.float16)
            elif c_min > np.finfo("f4").min and c_max < np.finfo("f4").max:
                df[col] = df[col].astype(np.float32)
            else:
                df[col] = df[col].astype(np.float64)
        elif str(col_type)[:3] == "int":
            c_min = df[col].min()
            c_max = df[col].max()
            if c_min > np.iinfo("i1").min and c_max < np.iinfo("i1").max:
                df[col] = df[col].astype(np.int8)
            elif c_min > np.iinfo("i2").min and c_max < np.iinfo("i2").max:
                df[col] = df[col].astype(np.int16)
            elif c_min > np.iinfo("i4").min and c_max < np.iinfo("i4").max:
                df[col] = df[col].astype(np.int32)
            elif c_min > np.iinfo("i8").min and c_max < np.iinfo("i8").max:
                df[col] = df[col].astype(np.int64)
        elif col == "timestamp":
            df[col] = pd.to_datetime(df[col])
        elif str(col_type)[:8] != "datetime":
            df[col] = df[col].astype("category")
    end_mem = df.memory_usage().sum() / 1024**2
    print('Потребление памяти меньше на',
            round(start_mem - end_mem, 2),
            'Мб (минус',
            round(100 * (start_mem - end_mem) / start_mem, 1),
            '%)')
    return df

# оптимизация памяти : строения
buildings = reduce_mem_usage(buildings)
print(buildings.info())

# оптимизация памяти : погода
weather = reduce_mem_usage(weather)
print(weather.info())

# оптимизация памяти : энергия
energy = reduce_mem_usage(energy)
print(energy.info())


def round_fillna(df, columns):
    for col in columns:
        type_ = "int8"
        if col in ["wind_direction", "year_build", "precip_depth_1_hr"]:
            type_ = "int16"
        if col == "precip_depth_1_hr":
            df[col] = df[col].apply(lambda x:0 if x < 0 else x)
        df[col] = np.round(df[col].fillna(value = 0).astype(type_))
    return df

# объединение данных и очистка
energy = pd.merge(left = energy, 
        right = buildings,
        how = "left",
        left_on = "building_id",
        right_on = "building_id")
energy = energy.set_index(["timestamp",
        "site_id"])
weather = weather.set_index(["timestamp",
        "site_id"])
energy = pd.merge(left = energy, 
        right = weather,
        how = "left",
        left_index = True,
        right_index = True)
energy.reset_index(inplace = True)
energy = energy.drop(columns = ["meter"],
        axis = 1)
energy = round_fillna(energy, ["wind_direction", 
            "wind_speed", 
            "cloud_coverage",
            "precip_depth_1_hr",
            "year_built",
            "floor_count"])
energy = energy[energy["meter_reading"] > 0]

del buildings
del weather

# поиск зависимостей зданий
data_corr_meta = pd.DataFrame(energy.groupby("building_id").median(),
        columns = ["meter_reading", 
            "square_feet",
            "year_built",
            "site_id"])
data_corr_meta.dropna(inplace = True)
sns.pairplot(data_corr_meta, height = 6)
plt.show()
del data_corr_meta

# Поиск зависимостей погода
data_corr_weather = pd.DataFrame(energy[energy["site_id"] == 0],
        columns = ["meter_reading", 
            "air_temperature",
            "dew_temperature",
            "sea_level_pressure",
            "wind_speed",
            "wind_direction"])
data_corr_weather.dropna(inplace = True)
sns.pairplot(data_corr_weather, height = 4)
plt.show()
del data_corr_weather

# поиск зависимостей тип зданий 
data_corr_temp_primary = pd.DataFrame(energy,
        columns = ["meter_reading", 
            "air_temperature",
            "primary_use"])
data_corr_temp_primary.dropna(inplace = True)
sns.pairplot(data_corr_temp_primary,
        hue = "primary_use",
        height = 4)
plt.show()
del data_corr_temp_primary


