Data Classification Using AI

Overview

This project is part of my DecodeLabs internship. It demonstrates a simple supervised machine learning classification model using the Iris dataset.

The project uses the K-Nearest Neighbors (KNN) algorithm to classify flowers into different species.

Objective

The main objective of this project is to understand the basic workflow of a machine learning classification problem:

- Load a dataset
- Separate features and target
- Split data into training and testing sets
- Scale the features
- Train a KNN model
- Make predictions
- Evaluate the model
- Predict the class of a new flower

Dataset

The Iris dataset contains:

- 150 flower samples
- 4 features
- 3 flower classes

Features

- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

Classes

- Setosa
- Versicolor
- Virginica

Technologies Used

- Python
- Scikit learn
- VS Code

Machine Learning Workflow

The project follows these steps:

1. Load the Iris dataset.
2. Separate the features ("X") and target ("y").
3. Split the data into training and testing sets.
4. Scale the features using "StandardScaler".
5. Create a KNN classifier with "K = 5".
6. Train the model using the training data.
7. Make predictions on the test data.
8. Evaluate the model using accuracy, confusion matrix, and F1 score.
9. Test the model with a new flower sample.

Model

The classification model used in this project is K-Nearest Neighbors (KNN).

The model was created with:

KNeighborsClassifier(n_neighbors=5)

Results

For the selected train/test split:

- Training samples: 120
- Testing samples: 30
- Accuracy: 1.0
- F1 Score: 1.0

Confusion Matrix

[[10  0  0]
 [ 0  9  0]
 [ 0  0 11]]

This means all 30 test samples in this particular split were classified correctly.

New Flower Prediction

The model was also tested with a new flower sample:

[5.1, 3.5, 1.4, 0.2]

The model predicted this flower as Setosa.

How to Run

Make sure Python and the required libraries are installed.

Run the project using:

python classification.py

Learning Outcome

Through this project, I learned the basic workflow of supervised machine learning, including data preparation, feature scaling, model training, prediction, and model evaluation.