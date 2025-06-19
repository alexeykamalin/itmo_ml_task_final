from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
from clearml import Task
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import os

load_dotenv()

assert os.getenv("CLEARML_API_ACCESS_KEY"), "API ключ не установлен!"
assert os.getenv("CLEARML_API_SECRET_KEY"), "Секретный ключ API не установлен!"

config = {
    "random_state": 42, 
    "data": {
        "test_size": 0.2,
        "binary_classification": True, 
        "quality_threshold": 6,
    },
    "logistic_regression": {
        "max_iter": 1000, 
        "penalty": "l2",  
        "C": 1.0,  
        "class_weight": "balanced" 
    }
}

def all_in_one():
    df = pd.read_csv('/opt/airflow/dags/winequality-red.csv')
    
    X = df.drop('quality', axis=1).values
    y = df['quality'].values
    
    if config["data"]["binary_classification"]:
        y = np.where(y >= config["data"]["quality_threshold"], 1, 0)
        print("\nРежим: Бинарная классификация")
        print(f"Вино с качеством >= {config['data']['quality_threshold']} считается 'хорошим'")
    else:
        print("\nРежим: Многоклассовая классификация")
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["data"]["test_size"],
        random_state=config["random_state"],
        stratify=y 
    )
    
    model = LogisticRegression(
        max_iter=config["logistic_regression"]["max_iter"],
        penalty=config["logistic_regression"]["penalty"],
        C=config["logistic_regression"]["C"],
        random_state=config["random_state"],
        class_weight=config["logistic_regression"]["class_weight"]
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    
    if config["data"]["binary_classification"]:
        f1 = f1_score(y_test, y_pred, average='binary')
        roc_auc = roc_auc_score(y_test, y_proba[:, 1])
    else:
        f1 = f1_score(y_test, y_pred, average='macro')
        try:
            roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovo')
        except:
            roc_auc = 0.0
    
    cm = confusion_matrix(y_test, y_pred)
    
    # Логирование в ClearML
    task = Task.init(
        project_name="Прогнозирование качества вина",
        task_name=f"Логистическая регрессия ({'Бинарная' if config['data']['binary_classification'] else 'Многоклассовая'})",
        output_uri=True,
    )
    
    # Логируем метрики
    logger = task.get_logger()
    logger.report_scalar("Точность", "Тест", accuracy, 0)
    logger.report_scalar("F1-мера", "Тест", f1, 0)
    logger.report_scalar("ROC-AUC", "Тест", roc_auc, 0)
    logger.report_confusion_matrix(
        "Матрица ошибок",
        "Тест",
        matrix=cm,
        iteration=0
    )
    
    # Логируем параметры модели
    task.connect(config)
    
    # Логируем важность признаков
    feature_names = df.drop('quality', axis=1).columns.tolist()
    for feature, coef in zip(feature_names, model.coef_[0]):
        logger.report_scalar(
            "Важность признаков",
            feature,
            coef,
            0
        )
    
    # 5. Вывод результатов
    print("\n=== Результаты классификации ===")
    print(f"Точность модели: {accuracy:.4f}")
    print(f"F1-мера: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print("\nМатрица ошибок:")
    print(cm)
    
    # Закрываем задачу ClearML
    task.close()