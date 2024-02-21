# import the modules
import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


file_path = './crimes.csv'

df = pd.read_csv(file_path)

X = df[['PC3', 'PC1', 'PC2']]
y = df['Primary Type']

k_input = int(input("Enter k: "))
test_size=float(input("Enter test size:"))

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)



knn5 = KNeighborsClassifier(n_neighbors = k_input)#  Predictions for the KNN Classifiers
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)


knn5.fit(X_train, y_train)


pred_k5 = knn5.predict(X_test)

label_encoder = LabelEncoder()
pred_k5_encoded = label_encoder.fit_transform(pred_k5)

# print(pred_k5_encoded)
print("Accuracy with k="+str(k_input), accuracy_score(y_test, pred_k5)*100,"%")
sns.set_palette("Set2")  
sns.set_style("whitegrid")
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
scatter_k1 = sns.scatterplot(x=X_test.iloc[:, 0], y=X_test.iloc[:, 1], hue=pred_k5_encoded, palette="Set2")
plt.title("Predicted values with k ="+str(k_input), fontsize=20)
plt.legend(labels=df['Primary Type'].unique(), loc='upper right',bbox_to_anchor=(2, 1))

plt.show()