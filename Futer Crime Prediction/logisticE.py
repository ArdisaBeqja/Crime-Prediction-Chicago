import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

def logistic_regression_classification(df):
    # Encode the 'Primary Type' column to numerical values
    label_encoder = LabelEncoder()
    df['Primary Type'] = label_encoder.fit_transform(df['Primary Type'])

    # Select features (Longitude and Latitude) and target (Encoded Primary Type)
    features = df[['PC1', 'PC2','PC3']]
    target = df['Primary Type']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=75)

    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize the logistic regression classifier
    logistic_reg_classifier = LogisticRegression(solver='newton-cg', max_iter=1000)

    # Fit the model on the training data
    logistic_reg_classifier.fit(X_train_scaled, y_train)

    # Predict the classes on the test set
    y_pred = logistic_reg_classifier.predict(X_test_scaled)

    # Add a new column with predicted classes to the DataFrame
    df.loc[X_test.index, 'Predicted Primary Type Linear Regression'] = label_encoder.inverse_transform(y_pred)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2%}")
       

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plotting points for each class in 3D
    for class_label in df['Predicted Primary Type Linear Regression'].unique():
        class_data = df[df['Predicted Primary Type Linear Regression'] == class_label]
        ax.scatter(class_data['PC1'], class_data['PC2'], class_data['PC3'], label=class_label)

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.set_title('3D Scatter Plot of Predicted Classes')
    ax.legend()
    plt.show()

    return df

