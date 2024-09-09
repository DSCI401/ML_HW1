import pandas as pd
import numpy as np
import os

# Set default paths 
default_train_path = "train.csv"  # Assume the file is in the same directory as the script
default_test_path = "test.csv"

train_path = input(f"Please enter the path to the train.csv file (default: {default_train_path}): ") or default_train_path
test_path = input(f"Please enter the path to the test.csv file (default: {default_test_path}): ") or default_test_path

# Check if the files exist
if not os.path.exists(train_path) or not os.path.exists(test_path):
    raise FileNotFoundError("The specified file path does not exist!")

# Load the datasets
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

print("Train dataset head:")
print(train_df.head())

print("Test dataset head:")
print(test_df.head())


# Preprocessing: Handle missing values and convert categorical variables

# Fill missing age values with the median age
train_df['Age'] = train_df['Age'].fillna(train_df['Age'].median())
test_df['Age'] = test_df['Age'].fillna(test_df['Age'].median())

# Fill missing embarked values with the most common value
train_df['Embarked'] = train_df['Embarked'].fillna(train_df['Embarked'].mode()[0])
test_df['Embarked'] = test_df['Embarked'].fillna(test_df['Embarked'].mode()[0])

# Fill missing fare values in test set with the median fare
test_df['Fare'] = test_df['Fare'].fillna(test_df['Fare'].median())

# Convert categorical variables into numeric ones
train_df['Sex'] = train_df['Sex'].map({'male': 0, 'female': 1})
test_df['Sex'] = test_df['Sex'].map({'male': 0, 'female': 1})

train_df['Embarked'] = train_df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
test_df['Embarked'] = test_df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# Drop irrelevant columns for simplicity
train_df = train_df.drop(columns=['Name', 'Ticket', 'Cabin'])
test_df = test_df.drop(columns=['Name', 'Ticket', 'Cabin'])

# Check the preprocessing results
print(train_df.head())
print(test_df.head())

# Simple rule-based model
def simple_model(X):
    predictions = []
    for row in X:
        if row[1] == 1:  # Female
            predictions.append(1)
        elif row[0] == 1 and row[2] < 18:  # Male in 1st class and young
            predictions.append(1)
        else:
            predictions.append(0)
    return np.array(predictions)

# Apply the model to the training data
X_train = train_df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']].values
y_train = train_df['Survived'].values

train_predictions = simple_model(X_train)

# Calculate accuracy
accuracy = (train_predictions == y_train).mean()
print(f"Simple Model Accuracy: {accuracy}")

# Define the weighted average model
def weighted_average_model(X, weights):
    pclass = X[:, 0]
    sex = X[:, 1]
    age = X[:, 2]
    sibsp = X[:, 3]
    parch = X[:, 4]
    fare = X[:, 5]
    embarked = X[:, 6]

    weighted_sum = (weights['Pclass'] * pclass +
                    weights['Sex'] * sex +
                    weights['Age'] * age +
                    weights['SibSp'] * sibsp +
                    weights['Parch'] * parch +
                    weights['Fare'] * fare +
                    weights['Embarked'] * embarked)

    predictions = (weighted_sum >= 0.5).astype(int)
    return predictions

# Initial weights
weights = {
    'Pclass': -0.5, 'Sex': 1.0, 'Age': -0.05, 'SibSp': -0.2,
    'Parch': -0.2, 'Fare': 0.003, 'Embarked': 0.1
}

# Apply the weighted average model to the training data
weighted_train_predictions = weighted_average_model(X_train, weights)

# Calculate accuracy
weighted_accuracy = (weighted_train_predictions == y_train).mean()
print(f"Weighted Model Accuracy: {weighted_accuracy}")

# First adjustment to weights
weights_adjusted_1 = {
    'Pclass': -0.8, 'Sex': 1.5, 'Age': -0.05, 'SibSp': -0.1,
    'Parch': -0.1, 'Fare': 0.003, 'Embarked': 0.1
}

weighted_train_predictions_adjusted_1 = weighted_average_model(X_train, weights_adjusted_1)
weighted_accuracy_adjusted_1 = (weighted_train_predictions_adjusted_1 == y_train).mean()
print(f"Adjusted Model Accuracy 1: {weighted_accuracy_adjusted_1}")

# Second adjustment to weights
weights_adjusted_2 = {
    'Pclass': -1.0, 'Sex': 1.5, 'Age': -0.1, 'SibSp': -0.1,
    'Parch': -0.1, 'Fare': 0.01, 'Embarked': 0.1
}

weighted_train_predictions_adjusted_2 = weighted_average_model(X_train, weights_adjusted_2)
weighted_accuracy_adjusted_2 = (weighted_train_predictions_adjusted_2 == y_train).mean()
print(f"Adjusted Model Accuracy 2: {weighted_accuracy_adjusted_2}")

# Third adjustment to weights
weights_adjusted_3 = {
    'Pclass': -1.0, 'Sex': 2.0, 'Age': -0.1, 'SibSp': -0.05,
    'Parch': -0.05, 'Fare': 0.02, 'Embarked': 0.1
}

weighted_train_predictions_adjusted_3 = weighted_average_model(X_train, weights_adjusted_3)
weighted_accuracy_adjusted_3 = (weighted_train_predictions_adjusted_3 == y_train).mean()
print(f"Adjusted Model Accuracy 3: {weighted_accuracy_adjusted_3}")

# Fourth adjustment to weights
weights_adjusted_4 = {
    'Pclass': -1.5, 'Sex': 2.5, 'Age': -0.1, 'SibSp': -0.05,
    'Parch': -0.05, 'Fare': 0.03, 'Embarked': 0.1
}

weighted_train_predictions_adjusted_4 = weighted_average_model(X_train, weights_adjusted_4)
weighted_accuracy_adjusted_4 = (weighted_train_predictions_adjusted_4 == y_train).mean()
print(f"Adjusted Model Accuracy 4: {weighted_accuracy_adjusted_4}")

# Eighth adjustment to weights
weights_adjusted_8 = {
    'Pclass': -1.8, 'Sex': 2.8, 'Age': -0.1, 'SibSp': -0.05,
    'Parch': -0.05, 'Fare': 0.065, 'Embarked': 0.1
}

weighted_train_predictions_adjusted_8 = weighted_average_model(X_train, weights_adjusted_8)
weighted_accuracy_adjusted_8 = (weighted_train_predictions_adjusted_8 == y_train).mean()
print(f"Adjusted Model Accuracy 8: {weighted_accuracy_adjusted_8}")

# Thirteenth adjustment to weights
weights_adjusted_13 = {
    'Pclass': -2.6, 'Sex': 3.8, 'Age': -0.1, 'SibSp': -0.05,
    'Parch': -0.05, 'Fare': 0.11, 'Embarked': 0.1
}

weighted_train_predictions_adjusted_13 = weighted_average_model(X_train, weights_adjusted_13)
weighted_accuracy_adjusted_13 = (weighted_train_predictions_adjusted_13 == y_train).mean()
print(f"Adjusted Model Accuracy 13: {weighted_accuracy_adjusted_13}")

# Apply the model to the test data using the best weights
X_test = test_df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']].values
test_predictions = weighted_average_model(X_test, weights_adjusted_13)

# Prepare the submission
submission = pd.DataFrame({
    'PassengerId': test_df['PassengerId'],
    'Survived': test_predictions
})

print("\nSubmission (First 5 rows):")
print(submission.head())

# Women and children first model
def women_and_children_first_model(X):
    predictions = []
    for row in X:
        sex = row[1]
        age = row[2]
        if sex == 1:
            predictions.append(1)
        elif sex == 0 and age < 14:
            predictions.append(1)
        else:
            predictions.append(0)
    return np.array(predictions)

# Apply the "Women and Children First" model to the training data
women_children_first_predictions = women_and_children_first_model(X_train)
women_children_first_accuracy = (women_children_first_predictions == y_train).mean()
print(f"Women and Children First Model Accuracy: {women_children_first_accuracy}")

# Apply the "Women and Children First" model to the test data
test_predictions_women_children_first = women_and_children_first_model(X_test)

# Prepare the submission DataFrame for "Women and Children First" model
submission_women_children_first = pd.DataFrame({
    'PassengerId': test_df['PassengerId'],
    'Survived': test_predictions_women_children_first
})

print("\nSubmission (Women and Children First Model - First 5 rows):")
print(submission_women_children_first.head())
