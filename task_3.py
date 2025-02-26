import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn import tree

# Load the dataset
dataset = pd.read_csv("dataset.csv", sep=";")  # Ensure your CSV file is in the working directory

# Display basic info
print("Dataset Head:")
print(dataset.head())

# Encode categorical columns using Label Encoding
categorical_cols = ["age", "balance", "day", "month", "duration", "campaign", "pdays", "previous", 
                    "job", "marital", "education", "default", "housing", "loan", "contact", "poutcome", "y"]

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    dataset[col] = le.fit_transform(dataset[col])
    label_encoders[col] = le  # Save encoders if needed later

# Define features and target
X = dataset.drop(columns=["y"])  # Features
y = dataset["y"]  # Target variable (0 = no, 1 = yes)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Decision Tree Classifier
clf = DecisionTreeClassifier(criterion="gini", max_depth=5, random_state=42)
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate the model
print("\nModel Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Visualize the Decision Tree
plt.figure(figsize=(20, 10))
tree.plot_tree(clf, feature_names=X.columns, class_names=["No", "Yes"], filled=True)
plt.tight_layout()
plt.show()

# Function to take manual input and prediction
def manual_prediction():
    print("\nEnter customer details manually for prediction (Enter values as 0, 1, 2, etc. wherever categorical data appears):")

    manual_data = []
    
    for col in X.columns:
        value = int(input(f"Enter {col}: "))  # Accept input as integer
        manual_data.append(value)
    
    # Convert to NumPy array and reshape
    manual_dataset = pd.DataFrame([manual_data], columns = X.columns)
    
    # Predict and display result
    prediction = clf.predict(manual_dataset)[0]
    print("\nPrediction Result:")

    if prediction == 1:
        print("The customer is likely to buy the product.")
              
    else :
        print("The customer is NOT likely to buy the product.")

# Call the function
manual_prediction()
