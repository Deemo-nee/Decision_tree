import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Đọc dữ liệu từ file csv
df = pd.read_csv('data/data.csv')
df.head()

# xem classificaiton
df['Classification'].value_counts()
df['Classification'] = df['Classification'] - 1

df['Classification'].value_counts()

# Tạo dữ liệu để train model
y = df['Classification'].values.reshape(-1, 1)
x = df.drop(columns=['Classification'])

print(x.shape)
print(y.shape)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix

my_tree = DecisionTreeClassifier(max_depth=3)
my_tree.fit(x_train, y_train)

# Dự đoán trên dữ liệu test
y_pred = my_tree.predict(x_test)
cm = confusion_matrix(y_test, y_pred)

print(cm)

plot_confusion_matrix(my_tree, x_test, y_test, cmap=plt.cm.Blues)

# Precision tree
df = pd.read_csv('data/datareg.csv')
df.head()

x = df['Cost'].values.reshape(-1, 1)
y = df['Profit'].values.reshape(-1, 1)

from sklearn.tree import DecisionTreeRegressor

my_tree = DecisionTreeRegressor(max_depth=3)
my_tree.fit(x, y)

x_test = [[45000]]
y_pred = my_tree.predict(x_test)
print(y_pred)

from sklearn import tree

tree.plot_tree(my_tree, future_names=['Cost'], class_names=['Profit'], filled=True)