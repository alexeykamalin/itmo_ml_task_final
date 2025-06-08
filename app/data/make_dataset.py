import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

def load_data():
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    df.to_csv('dataset/iris.csv', index=False)

def prepare_data():
    df = pd.read_csv('dataset/iris.csv')
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df.to_csv('dataset/iris_train.csv', index=False)
    test_df.to_csv('dataset/iris_test.csv', index=False)