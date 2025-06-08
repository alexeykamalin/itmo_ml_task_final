import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report
import json

def test_model():
    test_df = pd.read_csv('dataset/iris_test.csv')
    X_test = test_df.drop('target', axis=1)
    y_test = test_df['target']
    
    model = joblib.load('model.pkl')
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    metrics = {'accuracy': accuracy, 'report': report}
    with open('model_metrics.json', 'w') as f:
        json.dump(metrics, f)