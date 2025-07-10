import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def evaluate_model(name, y_true, y_pred):
    print(f"Модель: {name}")
    print("R²:", r2_score(y_true, y_pred))
    print("MAE:", mean_absolute_error(y_true, y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_true, y_pred)))
    print("------")


df = pd.read_csv("dataset.csv")

print("Пропуски:\n", df.isnull().sum())

# Заполнение пропусков медианой
df.fillna(df.median(numeric_only=True), inplace=True)

# Удаление дубликатов (но их точно нет)))
df = df.drop_duplicates(subset="hash")

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
sns.histplot(df["ball"], bins=20, kde=True)
plt.title("Распределение ball")

plt.subplot(1, 2, 2)
sns.histplot(df["egkr"], bins=20, kde=True, color="orange")
plt.title("Распределение egkr")
plt.show()

plt.figure(figsize=(15, 10))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".1f")
plt.title("Корреляция признаков с ball и egkr")
plt.show()

X = df.drop(columns=["hash", "ball"])
y = df["ball"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Линейная регрессия
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
print("Linear Regression R²:", r2_score(y_test, y_pred_lr))

# Случайный лес
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("Random Forest R²:", r2_score(y_test, y_pred_rf))

evaluate_model("Linear Regression", y_test, y_pred_lr)
evaluate_model("Random Forest", y_test, y_pred_rf)