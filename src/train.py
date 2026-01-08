
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import joblib

from src.utils import load_data, basic_checks

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "study_hours.csv"
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)
MODEL_PATH = ARTIFACTS / "linear_model.joblib"


def run():
    # 1) Load
    df = load_data(str(DATA_PATH))

    # 2) Quick checks
    info = basic_checks(df)
    print("Dataset info:", info)

    # 3) EDA plots
    plt.figure(figsize=(6,4))
    sns.scatterplot(data=df, x="hours", y="score")
    plt.title("Study Hours vs Score")
    plt.xlabel("Hours")
    plt.ylabel("Score")
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(ARTIFACTS / "scatter_hours_score.png", dpi=150)
    plt.close()

    # 4) Train/Test split
    X = df[["hours"]].values
    y = df["score"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 5) Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 6) Predict & Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    print(f"MAE: {mae:.2f} | RMSE: {rmse:.2f} | R2: {r2:.3f}")

    # 7) Line fit plot (on all points for visual)
    x_sorted = np.sort(X.flatten())
    y_line = model.predict(x_sorted.reshape(-1,1))

    plt.figure(figsize=(6,4))
    sns.scatterplot(x=df["hours"], y=df["score"], label="Data")
    plt.plot(x_sorted, y_line, color="crimson", label="Linear fit")
    plt.title("Linear Regression Fit")
    plt.xlabel("Hours")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(ARTIFACTS / "linear_fit.png", dpi=150)
    plt.close()

    # 8) Save metrics & model
    with open(ARTIFACTS / "metrics.txt", "w", encoding = "utf-8") as f:
        f.write(
            f"MAE: {mae:.2f} \n"
            f"RMSE: {rmse:.2f} \n"
            f"R2: {r2:.3f} \n"
            f"Coefficient: {model.coef_[0]:.4f} \n"
            f"Intercept: {model.intercept_:.4f} \n"
        )

    joblib.dump(model, MODEL_PATH)

    print("Artifacts saved to:", ARTIFACTS.resolve())
    print("Model saved to:", MODEL_PATH.resolve())


if __name__ == "__main__":
    run()
