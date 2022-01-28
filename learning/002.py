



import numpy as np 
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib.pyplot import rcParams
rcParams['figure.figsize'] = 16, 8 


energy_0 = pd.read_csv("http://video.ittensive.com/machine-learning/ashrae/train.0.0.csv.gz")
print(energy_0.head())

# adding a series with the hour of the day to build a daily consumption model
energy_0["timestamp"] = pd.to_datetime(energy_0["timestamp"])
energy_0["hour"] = energy_0["timestamp"].dt.hour
print (energy_0.head())

# displaying the average median energy consumption per hour
energy_0_hours = energy_0.groupby("hour")
energy_0_averages = pd.DataFrame(
        
        {"Среднее": energy_0_hours.mean()["meter_reading"], 
        "Медиана": energy_0_hours.median()["meter_reading"],}
        )
energy_0_averages.plot()
plt.show()

# removing null values from statistics
energy_0_hours_filtered = energy_0[energy_0["meter_reading"]>0].groupby("hour")
energy_0_averages_filtered = pd.DataFrame(
{"Среднее": energy_0_hours_filtered.mean()["meter_reading"], 
"Медиана": energy_0_hours_filtered.median()["meter_reading"],}
)
energy_0_averages_filtered.plot()
plt.show()

# interpolation by hours

x = np.arange(0, 24)
y = interp1d(x, energy_0_hours_filtered.median()["meter_reading"], kind = "cubic")
xn = np.arange(0, 23.1, 0.1)
yn = y(xn)
plt.plot(x, energy_0_hours_filtered.median()["meter_reading"], 'o', xn, yn, '-')
plt.show()
