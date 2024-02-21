import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

file_path = 'crimes.csv'
df = pd.read_csv(file_path)

X = df[['PC3', 'PC1', 'PC2']]
y = df['Primary Type']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

gnb = GaussianNB()

gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)

accuracy = accuracy_score(y_test, y_pred) * 100
print("Accuracy:", accuracy, "%")

class_report = classification_report(y_test, y_pred, zero_division=1)

with open('classification_report.txt', 'w') as file:
    file.write(class_report)
