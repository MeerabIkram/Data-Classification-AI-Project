from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

iris = load_iris()

print("Feature names:", iris.feature_names)
print("Target names:", iris.target_names)
print("Data shape:", iris.data.shape)

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

model = KNeighborsClassifier(n_neighbors=5)

model.fit(X_train, y_train)
predictions = model.predict(X_test)
print("Predictions:", predictions)

accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)

cm = confusion_matrix(y_test, predictions)
print("Confusion Matrix:")
print(cm)

f1 = f1_score(y_test, predictions, average="weighted")
print("F1 Score:", f1)

new_flower = [[5.1, 3.5, 1.4, 0.2]]

new_flower_scaled = scaler.transform(new_flower)
prediction = model.predict(new_flower_scaled)
print("New flower prediction:", prediction)