import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


dataset = pd.read_csv('your_dataset.csv')


print(dataset.head())  # Print the first few rows of the dataset
print(dataset.describe())  # Statistical summary of the dataset


X = dataset.drop('target_variable', axis=1)
y = dataset['target_variable']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LogisticRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
