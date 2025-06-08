from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_digits
from airflow.decorators import task
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score
from clearml import Task

import os

# Установка переменных окружения (альтернатива конфигурационному файлу)
os.environ['CLEARML_API_HOST'] = 'https://api.clear.ml'
os.environ['CLEARML_WEB_HOST'] = 'https://app.clear.ml'
os.environ['CLEARML_FILES_HOST'] = 'https://files.clear.ml'
os.environ['CLEARML_API_ACCESS_KEY'] = 'GGJ7LME47RFKTI2YZ9Z5XBQGRF5CXL'
os.environ['CLEARML_API_SECRET_KEY'] = 'gK1iU2Ez9wvMpSE8COwWRWvN7LqObNCwRQLp1htN8gtzaxnzcTiQNI_lmclruougT6Y'

config = {
    "random_state": 42,
    "data": {
        "test_size": 0.2,
    },
    "decision_tree": {
        "max_depth": 10,
        "criterion": "gini",
        "min_samples_split": 2
    }
}
def all_in_one_tree() -> None:
    dataset = load_digits(return_X_y=False)
    X = dataset.data
    y = dataset.target
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["data"]["test_size"],
        random_state=config["random_state"],
    )
    data = {
        "x_train": X_train,
        "x_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }

    model = DecisionTreeClassifier(
        random_state=config["random_state"],
        max_depth=config["decision_tree"]["max_depth"],
        criterion=config["decision_tree"]["criterion"],
        min_samples_split=config["decision_tree"]["min_samples_split"]
    )
    model.fit(data['x_train'], data['y_train'])

    y_pred = model.predict(data['x_test'])
    y_proba = model.predict_proba(data['x_test'])  # без [:, 1]!

    # Вычисляем метрики с учетом многоклассовости
    accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
    f1 = f1_score(y_true=y_test, y_pred=y_pred, average='macro')  # или 'micro', 'weighted'
    cm = confusion_matrix(y_true=y_test, y_pred=y_pred)
    roc_auc = roc_auc_score(y_true=y_test, y_score=y_proba, multi_class='ovo')  # или 'ovr'
    
    # Инициализируем задачу ClearML
    task = Task.init(
        project_name="Project", 
        task_name="Tree",
        output_uri=True,
    )
    
    # Логируем метрики в ClearML
    logger = Task.current_task().get_logger()
    logger.report_scalar(title="Accuracy", series="Test", value=accuracy, iteration=0)
    logger.report_scalar(title="F1 Score", series="Test", value=f1, iteration=0)
    logger.report_scalar(title="ROC-AUC", series="Test", value=roc_auc, iteration=0)
    logger.report_confusion_matrix(
        title="Confusion Matrix",
        series="Test",
        matrix=cm,
        iteration=0
    )
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print("Confusion Matrix:")
    print(cm)

    # Логируем параметры модели
    params = {
        "max_depth": config["decision_tree"]["max_depth"],
        "criterion": config["decision_tree"]["criterion"],
        "min_samples_split": config["decision_tree"]["min_samples_split"],
        "random_state": config["random_state"]
    }
    task.connect(params)

    logger = task.get_logger()
    logger.report_scalar(title="Tree Info", series="Depth", value=model.get_depth(), iteration=0)
    logger.report_scalar(title="Tree Info", series="Leaves", value=model.get_n_leaves(), iteration=0)

    task.close()
    