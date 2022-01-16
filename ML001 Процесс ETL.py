{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Постановка задачи\n",
    "Всего 3 набора данных: (1) building_metadata, (2) train и (3) weather_train\n",
    "* (1) содержит building_id, для которого есть данные (2)\n",
    "* (1) содержит site_id, для которого есть данные (3)\n",
    "\n",
    "Нужно объединить все наборы данных по building_id, site_id и timestamp\n",
    "ETL = получение + очистка + совмещение данных\n",
    "\n",
    "Данные:\n",
    "* http://video.ittensive.com/machine-learning/ashrae/building_metadata.csv.gz\n",
    "* http://video.ittensive.com/machine-learning/ashrae/weather_train.csv.gz\n",
    "* http://video.ittensive.com/machine-learning/ashrae/train.0.0.csv.gz",
    "\n",
    "Соревнование: https://www.kaggle.com/c/ashrae-energy-prediction/\n",
    "\n",
    "© ITtensive, 2020"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Подключение библиотек"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [],
   "source": [
    "%matplotlib inline\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "from matplotlib.pyplot import rcParams\n",
    "rcParams['figure.figsize'] = 16, 8"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Загрузка данных: здания\n",
    "* primary_use - назначение\n",
    "* square_feet - площадь, кв.футы\n",
    "* year_built - год постройки\n",
    "* floor_count - число этажей"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "   site_id  building_id primary_use  square_feet  year_built  floor_count\n",
      "0        0            0   Education         7432      2008.0          NaN\n",
      "1        0            1   Education         2720      2004.0          NaN\n",
      "2        0            2   Education         5376      1991.0          NaN\n",
      "3        0            3   Education        23685      2002.0          NaN\n",
      "4        0            4   Education       116607      1975.0          NaN\n"
     ]
    }
   ],
   "source": [
    "buildings = pd.read_csv(\"http://video.ittensive.com/machine-learning/ashrae/building_metadata.csv.gz\")\n",
    "print (buildings.head())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Загрузка данных: погода\n",
    "* air_temperature - температура воздуха, С\n",
    "* dew_temperature - точка росы (влажность), С\n",
    "* cloud_coverage - облачность, %\n",
    "* precip_depth_1_hr - количество осадков, мм/час\n",
    "* sea_level_pressure - давление, мбар\n",
    "* wind_direction - направление ветра, градусы\n",
    "* wind_speed - скорость ветра, м/с"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "   site_id            timestamp  air_temperature  cloud_coverage  \\\n",
      "0        0  2016-01-01 00:00:00             25.0             6.0   \n",
      "1        0  2016-01-01 01:00:00             24.4             NaN   \n",
      "2        0  2016-01-01 02:00:00             22.8             2.0   \n",
      "3        0  2016-01-01 03:00:00             21.1             2.0   \n",
      "4        0  2016-01-01 04:00:00             20.0             2.0   \n",
      "\n",
      "   dew_temperature  precip_depth_1_hr  sea_level_pressure  wind_direction  \\\n",
      "0             20.0                NaN              1019.7             0.0   \n",
      "1             21.1               -1.0              1020.2            70.0   \n",
      "2             21.1                0.0              1020.2             0.0   \n",
      "3             20.6                0.0              1020.1             0.0   \n",
      "4             20.0               -1.0              1020.0           250.0   \n",
      "\n",
      "   wind_speed  \n",
      "0         0.0  \n",
      "1         1.5  \n",
      "2         0.0  \n",
      "3         0.0  \n",
      "4         2.6  \n"
     ]
    }
   ],
   "source": [
    "weather = pd.read_csv(\"http://video.ittensive.com/machine-learning/ashrae/weather_train.csv.gz\")\n",
    "print (weather.head())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Загрузка данных: потребление энергии здания 0\n",
    "* meter_reading - значение показателя (TOE, эквивалент тонн нефти)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "   building_id  meter            timestamp  meter_reading\n",
      "0            0      0  2016-01-01 00:00:00            0.0\n",
      "1            0      0  2016-01-01 01:00:00            0.0\n",
      "2            0      0  2016-01-01 02:00:00            0.0\n",
      "3            0      0  2016-01-01 03:00:00            0.0\n",
      "4            0      0  2016-01-01 04:00:00            0.0\n"
     ]
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAA78AAAHgCAYAAABtp9qRAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4xLjAsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+17YcXAAAgAElEQVR4nOzdd5wU5f0H8M/cXoOjNwsgJwKCIE2lCGLBjkaNPerPGIklRuPPxAR7jaJRUWOL0Z9RY+8FRaWjgDSlI/Xo5Y5yB9dvd35/7DyzM7PPtO2393nnZdidmZ15ttzufOf7PN9HUVUVRERERERERNksJ90NICIiIiIiIko2Br9ERERERESU9Rj8EhERERERUdZj8EtERERERERZj8EvERERERERZT0Gv0RERERERJT1ctPdgFTr0KGDWlxcnO5mEBERERERUYJ16NAB33zzzTeqqp5pXdfkgt/i4mIsWLAg3c0gIiIiIiKiJFAUpYNsObs9ExERERERUdZj8EtERERERERZj8EvERERERERZT0Gv0RERERERJT1GPwSERERERFR1mPwS0RERERERFmPwS8RERERERFlPQa/RERERERElPUY/BIREREREVHWY/BLREREREREWY/BLxEREREREWU9Br9ERERERESU9Rj8EhERERERUdZj8EtERERERERZj8EvERERERERZT0Gv0RERERERJT1GPwSERERERFR1mPwS0SUIYrHTcTfPlyS7mYQERERZSUGv0REGeS9BZvT3QQiIiKirMTgl4iIiIiIiLIeg18iIiIiIiLKegx+iYiIiIiIKOsx+CUiIiIiIqKsx+CXiIiIiIiIsh6DXyIiIiIiIsp6DH6JiIiIiIgo6zH4JSIiIiIioqzH4JeIiIiIiIiyHoNfIiIiIiIiynoMfomIiIiIiCjrMfglIiIiIiKirMfgl4iIiIiIiLIeg18iIiIiIiLKegx+iYiIiIiIKOsx+CUiIiIiIqKsx+CXiIiIiIiIsh6DXyIiIiIiIsp6DH6JiIiIiIgo6zH4JSIiIiIioqzH4JeIiIiIiIiyHoNfIiIiIiIiynoMfomIiIiIiCjrMfglIiIiIiKirMfgl4iIiIiIiLIeg18iIiIiIiLKegx+iYiIiIiIKOsx+CUiIiIiIqKsx+CXiIiIiIiIsh6DXyIiIiIiIsp6DH6JiIiIiIgo6zH4JSIiIiIioqzH4JeIiIiIiIiyHoNfIiIiIiIiynoMfomIiIiIiCjrMfglIiIiIiKirMfgl4iIiIiIiLJexgW/iqIEFEX5SVGUL7X7hyuK8qOiKGsURXlPUZR8bXmBdn+ttr44ne0mIiIiIiKizJVxwS+APwFYabj/GIAJqqr2BLAXwLXa8msB7FVVtQeACdp2RERERERERFEyKvhVFKULgDEAXtHuKwBOAfChtsnrAM7Xbp+n3Ye2frS2PREREREREZFJRgW/AJ4G8FcAIe1+ewD7VFVt0O5vAdBZu90ZwGYA0NaXa9sTERERERERmWRM8KsoyjkAdqmqutC4WLKp6mGddd/XKYqyQFGUBaWlpXG2lIiIiIiIiBqbjAl+AYwA8CtFUUoAvItwd+enAbRRFCVX26YLgG3a7S0AugKAtr41gD2yHauq+rKqqseqqnpsx44dk/cMiIiIiIiIKCNlTPCrquodqqp2UVW1GMBlAKaqqnoFgGkALtI2uxrAZ9rtz7X70NZPVVVVmvklIiIiIiKipi1jgl8HfwNwm6IoaxEe0/uqtvxVAO215bcBGJem9hEREREREVGGy3XfJPVUVZ0OYLp2ez2AIZJtagBcnNKGERERERERUaPUGDK/RERERERERHFh8EtERERERERZj8EvERERERERZT0Gv0RERERERJT1GPwSERERERFR1mPwS0RERERERFmPwS8RERERERFlPQa/RERERERElPUY/BIREREREVHWY/BLREREREREWY/BLxEREREREWU9Br9ERERERESU9Rj8EhERERERUdZj8EtERERERERZj8EvERERERERZT0Gv0RERERERJT1GPwSERERERFR1mPwS0RERERERFmPwS8RERERERFlPQa/RERERERElPUY/BIREREREVHWY/BLREREREREWY/BLxEREREREWU9Br9ERERERESU9Rj8EhERERERUdZj8EtERERERERZj8EvERERERERZT0Gv0RERERERJT1GPwSERERERFR1mPwS0RERERERFmPwS8RERERERFlPQa/RERERERElPUY/BIREREREVHWY/BLREREREREWY/BLxEREREREWU9Br9ERERERESU9Rj8EhERERERUdZj8EtERERERERZj8EvERERERERZT0Gv0RERERERJT1GPwSERERERFR1mPwS0RERERERFmPwS8RERERERFlPQa/RERERERElPUY/BIREREREVHWY/BLREREREREWY/BLxEREREREWU9Br9ERERERESU9Rj8EhERERERUdZj8EtERERERERZj8EvERERERERZT0Gv0RERERERJT1GPwSERERERFR1mPwS0RERERERFmPwS8RERERERFlPQa/RERERBmmtiGIHeU16W4GEVFWYfBLRERElGH+972fMezRKQiF1HQ3hYgoazD4JSIiokbv+zVlKK+uT3czEuarpTsAAPWhUJpbQkSUPRj8EhERUaO2r6oOV776I27878J0N8W3ab/swm9fmwdVlWd464P+M7819UGU7q+Nt2lERFmHwS8RERE1anUN4ezoml0H0twS/679z3xM/6UUdr2b67Xntnrnfrz140bTun1VdXhm8pqortHXvDYfx/19clLaS0TUmDH4JSIiopQ5+YnpuOrVHz1vv728Gk99+wsagu7df22Sp54tKNmDsgPeM6avzFqP4nETUW9p2/drylBZ2+Dr2HaZ3zpt36dPmIm7PllmWnf3p8swYfJqzFxTalo+Z/1uX8cmImoqGPwSERFRymwoq8SsNWXSdTX1QeytrAMA7KmsQ019EMMfnYpnp67FP6eujdr+1e834LFJqwAlMW276KU5uOjF2Y7b1NQHcdYzs7Bw4148O2UNAOCJb37BM5PXYEd5Dbbuq8aVr/6IP7+/WH9M6f7aqADZKqSGKzzv0Z6/ELRkdUMhFTsrwlWgRYBt3ObpyatdniURUdPF4JeIiIgywqUvz8Wgh74DAAx+6Dv89rV5+rqpq3ZFbf/Qlyvw4vR1+v2yA7WoqQ/i05+2YumW8pjaULK7ynH98m0VWLm9Ahe+OBs5OeGo+18z12PC5NUY9ugUvD67BACwetd+AEB9MITj/j4Zf/twiXR/ihLeR0hVMfb1BRisPX/BGvw+9d1qDH1kCrbtq4aq7yOy/unJa2wfS0TU1DH4JSKihBr95HT87j/z090MSgBVVTFl5c6ETLdz2/s/49iHzYHdnHW7ccDQPXjx5n2m9XPX7/F9nJdmrMOt7/2Mc5/7PraG+rCvKrq69Msz1wOA/pqt2Rkeh/zVsu3SfYi4tT4YkmbET3h8GorHTdTvPzctnAHfUVGD6b+UascKF86y2lHBeYKJ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\n",
      "text/plain": [
       "<Figure size 1152x576 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "energy_0 = pd.read_csv(\"http://video.ittensive.com/machine-learning/ashrae/train.0.0.csv.gz\")\n",
    "print (energy_0.head())\n",
    "energy_0.set_index(\"timestamp\")[\"meter_reading\"].plot()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Объединение потребления энергии и информацию о здании\n",
    "Проводим объединение по building_id"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "   building_id  meter            timestamp  meter_reading  site_id  \\\n",
      "0            0      0  2016-01-01 00:00:00            0.0        0   \n",
      "1            0      0  2016-01-01 01:00:00            0.0        0   \n",
      "2            0      0  2016-01-01 02:00:00            0.0        0   \n",
      "3            0      0  2016-01-01 03:00:00            0.0        0   \n",
      "4            0      0  2016-01-01 04:00:00            0.0        0   \n",
      "\n",
      "  primary_use  square_feet  year_built  floor_count  \n",
      "0   Education         7432      2008.0          NaN  \n",
      "1   Education         7432      2008.0          NaN  \n",
      "2   Education         7432      2008.0          NaN  \n",
      "3   Education         7432      2008.0          NaN  \n",
      "4   Education         7432      2008.0          NaN  \n"
     ]
    }
   ],
   "source": [
    "energy_0 = pd.merge(left=energy_0, right=buildings, how=\"left\",\n",
    "                   left_on=\"building_id\", right_on=\"building_id\")\n",
    "print (energy_0.head())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Объединение потребления энергии и погоды\n",
    "Выставим индексы для объединения - timestamp, site_id"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [],
   "source": [
    "energy_0.set_index([\"timestamp\", \"site_id\"], inplace=True)\n",
    "weather.set_index([\"timestamp\", \"site_id\"], inplace=True)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Проведем объединение и сбросим индексы"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "             timestamp  site_id  building_id  meter  meter_reading  \\\n",
      "0  2016-01-01 00:00:00        0            0      0            0.0   \n",
      "1  2016-01-01 01:00:00        0            0      0            0.0   \n",
      "2  2016-01-01 02:00:00        0            0      0            0.0   \n",
      "3  2016-01-01 03:00:00        0            0      0            0.0   \n",
      "4  2016-01-01 04:00:00        0            0      0            0.0   \n",
      "\n",
      "  primary_use  square_feet  year_built  floor_count  air_temperature  \\\n",
      "0   Education         7432      2008.0          NaN             25.0   \n",
      "1   Education         7432      2008.0          NaN             24.4   \n",
      "2   Education         7432      2008.0          NaN             22.8   \n",
      "3   Education         7432      2008.0          NaN             21.1   \n",
      "4   Education         7432      2008.0          NaN             20.0   \n",
      "\n",
      "   cloud_coverage  dew_temperature  precip_depth_1_hr  sea_level_pressure  \\\n",
      "0             6.0             20.0                NaN              1019.7   \n",
      "1             NaN             21.1               -1.0              1020.2   \n",
      "2             2.0             21.1                0.0              1020.2   \n",
      "3             2.0             20.6                0.0              1020.1   \n",
      "4             2.0             20.0               -1.0              1020.0   \n",
      "\n",
      "   wind_direction  wind_speed  \n",
      "0             0.0         0.0  \n",
      "1            70.0         1.5  \n",
      "2             0.0         0.0  \n",
      "3             0.0         0.0  \n",
      "4           250.0         2.6  \n"
     ]
    }
   ],
   "source": [
    "energy_0 = pd.merge(left=energy_0, right=weather, how=\"left\",\n",
    "                   left_index=True, right_index=True)\n",
    "energy_0.reset_index(inplace=True)\n",
    "print (energy_0.head())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Нахождение пропущенных данных\n",
    "Посчитаем количество пропусков данных по столбцам"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "floor_count: 8784\n",
      "air_temperature: 3\n",
      "cloud_coverage: 3830\n",
      "dew_temperature: 3\n",
      "precip_depth_1_hr: 1\n",
      "sea_level_pressure: 85\n",
      "wind_direction: 250\n",
      "             timestamp  site_id  building_id  meter  meter_reading  \\\n",
      "0  2016-01-01 00:00:00        0            0      0            0.0   \n",
      "\n",
      "  primary_use  square_feet  year_built  floor_count  air_temperature  \\\n",
      "0   Education         7432      2008.0          NaN             25.0   \n",
      "\n",
      "   cloud_coverage  dew_temperature  precip_depth_1_hr  sea_level_pressure  \\\n",
      "0             6.0             20.0                NaN              1019.7   \n",
      "\n",
      "   wind_direction  wind_speed  \n",
      "0             0.0         0.0  \n"
     ]
    }
   ],
   "source": [
    "for column in energy_0.columns:\n",
    "    energy_nulls = energy_0[column].isnull().sum()\n",
    "    if energy_nulls > 0:\n",
    "        print (column + \": \" + str(energy_nulls))\n",
    "print (energy_0[energy_0[\"precip_depth_1_hr\"].isnull()])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Заполнение пропущенных данных\n",
    "* air_temperature: NaN -> 0\n",
    "* cloud_coverage: NaN -> 0\n",
    "* dew_temperature: NaN -> 0\n",
    "* precip_depth_1_hr: NaN -> 0, -1 -> 0\n",
    "* sea_level_pressure: NaN -> среднее\n",
    "* wind_direction: NaN -> среднее (роза ветров)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 8784 entries, 0 to 8783\n",
      "Data columns (total 16 columns):\n",
      "timestamp             8784 non-null object\n",
      "site_id               8784 non-null int64\n",
      "building_id           8784 non-null int64\n",
      "meter                 8784 non-null int64\n",
      "meter_reading         8784 non-null float64\n",
      "primary_use           8784 non-null object\n",
      "square_feet           8784 non-null int64\n",
      "year_built            8784 non-null float64\n",
      "floor_count           0 non-null float64\n",
      "air_temperature       8784 non-null float64\n",
      "cloud_coverage        8784 non-null float64\n",
      "dew_temperature       8784 non-null float64\n",
      "precip_depth_1_hr     8784 non-null float64\n",
      "sea_level_pressure    8784 non-null float64\n",
      "wind_direction        8784 non-null float64\n",
      "wind_speed            8784 non-null float64\n",
      "dtypes: float64(10), int64(4), object(2)\n",
      "memory usage: 1.1+ MB\n"
     ]
    }
   ],
   "source": [
    "energy_0[\"air_temperature\"].fillna(0, inplace=True)\n",
    "energy_0[\"cloud_coverage\"].fillna(0, inplace=True)\n",
    "energy_0[\"dew_temperature\"].fillna(0, inplace=True)\n",
    "energy_0[\"precip_depth_1_hr\"] = energy_0[\"precip_depth_1_hr\"].apply(lambda x:x if x>0 else 0)\n",
    "energy_0_sea_level_pressure_mean = energy_0[\"sea_level_pressure\"].mean()\n",
    "energy_0[\"sea_level_pressure\"] = energy_0[\"sea_level_pressure\"].apply(lambda x:energy_0_sea_level_pressure_mean if x!=x else x)\n",
    "energy_0_wind_direction_mean = energy_0[\"wind_direction\"].mean()\n",
    "energy_0[\"wind_direction\"] = energy_0[\"wind_direction\"].apply(lambda x:energy_0_wind_direction_mean if x!=x else x)\n",
    "energy_0.info()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
