import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

def train_model():
    train_df = pd.read_csv('dataset/iris_train.csv')
    X_train = train_df.drop('target', axis=1)
    y_train = train_df['target']
    
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    
    joblib.dump(model, 'model.pkl')