import csv
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import linregress

"""

# 从redis导入数据源
import redis
import json

file_path = "why.json"
redis_conn = redis.Redis(host="192.168.1.123", port=5234, password="423vbdfkn", db=1, decode_responses=True)

file_object = open(file_path, 'r', encoding="utf8")
all_data = json.load(file_object)

for key in all_data:
    redis_conn.set(key, json.dumps(all_data[key], ensure_ascii=False))
    
file_object.close()
"""


# 交易平均值
# 导入数据源
df = pd.read_csv("data/AG2112.SHF.csv", usecols=range(10))

df['datetime'] = pd.to_datetime(df['datetime'])


df = df.set_index('datetime')

df_copy = df.resample(rule = '5T').mean()


df_copy['average'] = df_copy[['highest', 'lowest']].mean(axis=1)
print(df_copy)

plt.style.use('ggplot')
fig, ax = plt.subplots()
ax.plot(df_copy.index, df_copy['average'], c='red')

ax.set_title("交易平均值", fontsize=24)
ax.set_xlabel('时间', fontsize=8)
fig.autofmt_xdate()
ax.set_ylabel("平均价格", fontsize=16)
ax.tick_params(axis='both', which='major', labelsize=16)
plt.show()

# 价格趋势
from sklearn.linear_model import LinearRegression
import numpy as np
X = df_copy.index.values.reshape(-1,1)
X = np.asarray(X, dtype='float64')
print(type(X))
df_copy['average'] = df_copy['average'].fillna(0)
y = df_copy[['average']].values.reshape(-1,1)
print(y)

regressor = LinearRegression()
regressor.fit(X, y)

y_pre = regressor.predict(X)

plt.scatter(df_copy.index, y, color = 'red')
plt.plot(df_copy.index, y_pre, color = 'blue')
plt.title('价格趋势')
plt.xlabel('时间')
plt.ylabel('价格')
plt.show()


# moving averages
windows_size = 1

numbers_series = pd.Series(df_copy['average'])

windows = numbers_series.rolling(windows_size)

moving_averages = windows.mean()

moving_averages_list = moving_averages.tolist()

final_list = moving_averages_list[windows_size - 1:]

print("Moving average: " + str(final_list))

plt.style.use('ggplot')
fig, ax = plt.subplots()
ax.plot(df_copy.index, final_list, c='red')

ax.set_title("移动均线", fontsize=24)
ax.set_xlabel('时间', fontsize=8)
fig.autofmt_xdate()
ax.set_ylabel("价格", fontsize=16)
ax.tick_params(axis='both', which='major', labelsize=16)
plt.show()
