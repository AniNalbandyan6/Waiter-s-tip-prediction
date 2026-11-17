import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

data = pd.read_csv("tips.csv")
# print(data.info())
# plt.figure(1)
# plt.scatter(data["total_bill"], data["tip"])
# plt.xlabel("bill")
# plt.ylabel("tip")
# plt.figure(2)
# sns.boxplot(data=data, x="day", y="tip")
# plt.xlabel("day")
# plt.ylabel("tip")
# plt.figure(3)
# sns.barplot(data=data, x="size", y = "tip")
# plt.xlabel("size")
# plt.ylabel("tip")
# plt.figure(4)
# sns.barplot(data=data, x="time", y = "tip")
# plt.xlabel("time")
# plt.ylabel("tip")
# plt.show()

# for i in data.columns:
#     if i not in ('tip', 'total_bill'):
#         print(data[i].value_counts())

data_wy = data.drop(columns=["tip"])
y = data['tip']
encoded_df = pd.get_dummies(
    data_wy,
    columns=['day', 'time', 'sex', 'smoker'],
    drop_first=True
)

x_train,  x_test, y_train, y_test = train_test_split(encoded_df, y, test_size=0.2, random_state=42)
# print(x_train.shape)
# print(x_test.shape)
# print(y_train.shape)
# print(y_test.shape)

model = LinearRegression()
model.fit(x_train, y_train)
y_pred=model.predict(x_test)
# print(y_pred)
# print(y_test)


mae = mean_absolute_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)
print("MAE ", mae)
print("RMSE", rmse)
