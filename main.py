import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from pandas.plotting import scatter_matrix

filename = "C:/Users/User/PycharmProjects/pythonProject/09_irisdata.csv"
column_names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
data = pd.read_csv(filename, names=column_names)

x = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=41)

model = DecisionTreeClassifier()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
print(y_pred)

accuracy = accuracy_score(y_test, y_pred)

print("-------------------------")
print("Actual Values:", y_test)
print("Predicted Values:", y_pred)
print("-------------------------")
print("Accuracy:", accuracy)
print(data.describe())
print(data.shape)
print(data.groupby('class').size())

scatter_matrix(data, alpha=0.2, figsize=(10, 10), diagonal='hist')
plt.savefig('scatter_matrix_plot.png')


