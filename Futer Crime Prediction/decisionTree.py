import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import numpy as np

# Load data from crimes.csv
file_path = './crimes.csv'
df = pd.read_csv(file_path)

# Encode the 'Primary Type' column to numerical values
label_encoder = LabelEncoder()
df['Primary Type'] = label_encoder.fit_transform(df['Primary Type'])

# Select features (PC1, PC2, PC3) and target (Encoded Primary Type)
features = df[['PC1', 'PC2', 'PC3']]
target = df['Primary Type']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the decision tree classifier
decision_tree_classifier = DecisionTreeClassifier(random_state=42)

# Fit the decision tree model on the training data
decision_tree_classifier.fit(X_train_scaled, y_train)

# Predict the classes on the test set using the decision tree
y_pred_decision_tree = decision_tree_classifier.predict(X_test_scaled)

# Add a new column with predicted classes using the decision tree to the DataFrame
df.loc[X_test.index, 'Predicted Primary Type Decision Tree'] = label_encoder.inverse_transform(y_pred_decision_tree)

# Calculate and print accuracy for the decision tree
accuracy_decision_tree = accuracy_score(y_test, y_pred_decision_tree)
print(f"Accuracy (Decision Tree): {accuracy_decision_tree:.2%}")

# Scatter plot for actual vs. predicted classes (Decision Tree) with decision boundary
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

scatter_actual_tree = ax.scatter(X_test['PC1'], X_test['PC2'], X_test['PC3'], c=y_test, cmap='viridis', marker='o', s=100, label='Actual')
scatter_pred_tree = ax.scatter(X_test['PC1'], X_test['PC2'], X_test['PC3'], c=y_pred_decision_tree, cmap='viridis', marker='x', s=100, label='Predicted')

ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
ax.legend()

plt.title('3D Scatter Plot with Decision Boundaries (Decision Tree)')
plt.show()
