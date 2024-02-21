from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd


def read_file():
    file_path = 'crimes.csv'
    df = pd.read_csv(file_path)
    file = {
        'features': df.iloc[:, 1:],
        'target': df.iloc[:, 0]
    }
    return file


def splitter(size=0.78):
    df = read_file()
    X_train, X_test, y_train, y_test = train_test_split(df['features'], df['target'], train_size=size, random_state=42)
    df['X_train'] = X_train
    df['X_test'] = X_test
    df['y_train'] = y_train
    df['y_test'] = y_test
    return df


def color_config(df, y_pred, base_clr=150, clr_diff=30):
    colorset = {x: y for x, y in
                zip(set(df['target']), range(base_clr, base_clr + len(set(df['target'])) * clr_diff + 1, clr_diff))}
    color_train = [colorset[x] for x in df['y_train']]
    color_test = [colorset[x] for x in y_pred]
    return {
        'train': color_train,
        'test': color_test
    }


def plot_model(df, y_pred):
    colors = color_config(df, y_pred)
    plt.figure(figsize=(8, 6))
    plt.scatter(df['X_train'].iloc[:, 1], df['X_train'].iloc[:, 2], marker='o', label='Training Points',
                c=colors['train'])
    plt.scatter(df['X_test'].iloc[:, 1], df['X_test'].iloc[:, 2], marker='x', label='Testing Points', c=colors['test'])
    plt.title('SVM testing')
    plt.xlabel('PCA1')
    plt.ylabel('PCA2')
    plt.legend()
    plt.show()

def color_config(df, y_pred, base_clr=150, clr_diff=30):
    colorset = {x: y for x, y in
                zip(set(df['target']), range(base_clr, base_clr + len(set(df['target'])) * clr_diff + 1, clr_diff))}
    color_train = [colorset[x] for x in df['y_train']]
    color_test = [colorset[x] for x in y_pred]
    return {
        'train': color_train,
        'test': color_test
    }


def plot_model(df, y_pred):
    colors = color_config(df, y_pred)
    plt.figure(figsize=(8, 6))
    plt.scatter(df['X_train'].iloc[:, 1], df['X_train'].iloc[:, 2], marker='o', label='Training Points',
                c=colors['train'])
    plt.scatter(df['X_test'].iloc[:, 1], df['X_test'].iloc[:, 2], marker='x', label='Testing Points', c=colors['test'])
    plt.title('SVM testing')
    plt.xlabel('PCA1')
    plt.ylabel('PCA2')
    plt.legend()
    plt.show()

def svm_train_test(df, show=True):
    svm_model = SVC(kernel='linear')
    svm_model.fit(df['X_train'], df['y_train'])
    y_pred = svm_model.predict(df['X_test'])
    accuracy = accuracy_score(df['y_test'], y_pred)
    if show:
        print(f"Accuracy: {accuracy * 100:.2f}%")
    return {
        'accuracy': accuracy,
        'predictions': y_pred
    }


def play_svm(test_size=0.78, plot=True, show=True):
    df = splitter(size=test_size)
    results = svm_train_test(df, show=show)
    if plot:
        plot_model(df, results['predictions'])
    return {
        'dataset': df,
        'result': results
    }


if __name__ == '__main__':
    play_svm()
