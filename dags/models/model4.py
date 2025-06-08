from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_digits
from airflow.decorators import task
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score
import wandb

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
    
    wandb.login(key="18522e7cc4d1f9edb53ed19c35dcfd0133b77451")
    # Инициализируем задачу wandb
    task = wandb.init(
        project="Project", 
        name="Regression",
    )
    
    # Логирование параметров
    task.config.update(config)

    # Логирование метрик
    task.log({
        "accuracy": accuracy,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": cm,
    })

    # Логирование структуры дерева
    task.log({
        "tree_depth": model.get_depth(),
        "n_leaves": model.get_n_leaves()
    })

    task.finish()
    