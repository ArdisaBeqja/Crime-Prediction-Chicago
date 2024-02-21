from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def random_forest_classification(df):
    label_encoder = LabelEncoder()
    df['Primary Type'] = label_encoder.fit_transform(df['Primary Type'])
    class_names = label_encoder.inverse_transform(df['Primary Type'].unique())
    
    features = df[['PC1', 'PC2', 'PC3']]
    target = df['Primary Type']

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=75)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
    random_forest.fit(X_train_scaled, y_train)

    y_pred = random_forest.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2%}")

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(121, projection='3d')

    # Plotting points for each class in 3D
    for class_label, class_name in zip(df['Primary Type'].unique(), class_names):
        class_data = df[df['Primary Type'] == class_label]
        ax.scatter(class_data['PC1'], class_data['PC2'], class_data['PC3'], label=class_name)

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.set_title('3D Scatter Plot of Predicted Classes-Random Forest)')
    ax.legend()

    ax.legend(loc='center left', bbox_to_anchor=(1.5, 0.5))
    plt.show()

    return accuracy


