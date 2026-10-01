"""
Лабораторная работа 1: Линейная регрессия и факторный анализ (PCA)
Датасет: Boston Housing (schirmerchad/bostonhoustingmlnd)
"""

from pathlib import Path

import kagglehub
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ---------------------------------------------------------------------------
# Настройки
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 11

# ---------------------------------------------------------------------------
# 1. Загрузка данных
# ---------------------------------------------------------------------------
path = kagglehub.dataset_download("schirmerchad/bostonhoustingmlnd")
csv_path = Path(path) / "housing.csv"
df = pd.read_csv(csv_path)

print("=" * 60)
print("1. ЗАГРУЗКА ДАТАСЕТА")
print("=" * 60)
print(f"Путь: {csv_path}")
print(f"Размер: {df.shape[0]} строк, {df.shape[1]} столбцов")
print("\nПервые строки:")
print(df.head())
print("\nОписание признаков:")
print(
    """
RM      — среднее число комнат в жилище
LSTAT   — % населения с низким социальным статусом
PTRATIO — соотношение учеников и учителей в районе
MEDV    — медианная стоимость жилья (целевая переменная, $)
"""
)
print("\nСтатистика:")
print(df.describe())
print("\nПропуски:")
print(df.isnull().sum())
print("\nТипы данных:")
print(df.dtypes)

# ---------------------------------------------------------------------------
# 2. Визуализация распределений
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("2. ВИЗУАЛИЗАЦИЯ РАСПРЕДЕЛЕНИЙ")
print("=" * 60)

features = ["RM", "LSTAT", "PTRATIO"]
target = "MEDV"

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for ax, col in zip(axes.flat, features + [target]):
    sns.histplot(df[col], kde=True, ax=ax, color="steelblue")
    ax.set_title(f"Распределение {col}")
fig.suptitle("Распределения признаков и целевой переменной", fontsize=14)
plt.tight_layout()
plt.savefig(FIG_DIR / "01_distributions.png", dpi=150, bbox_inches="tight")
plt.close()

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, col in zip(axes, features):
    sns.scatterplot(data=df, x=col, y=target, ax=ax, alpha=0.6)
    ax.set_title(f"{col} vs {target}")
fig.suptitle("Зависимость целевой переменной от признаков", fontsize=14)
plt.tight_layout()
plt.savefig(FIG_DIR / "02_scatter_vs_target.png", dpi=150, bbox_inches="tight")
plt.close()
print("Сохранены: 01_distributions.png, 02_scatter_vs_target.png")

# ---------------------------------------------------------------------------
# 3. Предобработка
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("3. ПРЕДОБРАБОТКА ДАННЫХ")
print("=" * 60)

df_clean = df.dropna().copy()
print(f"После удаления пропусков: {df_clean.shape[0]} строк (было {df.shape[0]})")
print("Категориальных признаков нет — кодирование не требуется.")

X = df_clean[features]
y = df_clean[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)
print(f"Train: {X_train.shape[0]} ({100 * (1 - TEST_SIZE):.0f}%), Test: {X_test.shape[0]} ({100 * TEST_SIZE:.0f}%)")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Выполнена стандартизация признаков (StandardScaler) на train, применена к test.")

# ---------------------------------------------------------------------------
# 4. Корреляции и VIF
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("4. КОРРЕЛЯЦИИ И МУЛЬТИКОЛЛИНЕАРНОСТЬ (VIF)")
print("=" * 60)

corr = df_clean.corr()
print("\nМатрица корреляций:")
print(corr.round(3))

plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, fmt=".3f", square=True)
plt.title("Матрица корреляций")
plt.tight_layout()
plt.savefig(FIG_DIR / "03_correlation_matrix.png", dpi=150, bbox_inches="tight")
plt.close()

# VIF на стандартизованных признаках (весь набор для диагностики)
X_all_scaled = StandardScaler().fit_transform(X)
vif_data = pd.DataFrame(
    {
        "Признак": features,
        "VIF": [variance_inflation_factor(X_all_scaled, i) for i in range(X_all_scaled.shape[1])],
    }
)
print("\nVIF-коэффициенты (VIF > 5–10 — признак мультиколлинеарности):")
print(vif_data.to_string(index=False))
print(
    f"\nВывод: max VIF = {vif_data['VIF'].max():.3f}. "
    + (
        "Сильная мультиколлинеарность не обнаружена."
        if vif_data["VIF"].max() < 5
        else "Есть признаки мультиколлинеарности."
    )
)

# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------
def evaluate(y_true, y_pred, name=""):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    mape = float(mean_absolute_percentage_error(y_true, y_pred) * 100)
    return {"Модель": name, "RMSE": rmse, "R2": r2, "MAPE_%": mape}


def cv_scores(pipeline, X_data, y_data):
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    neg_mse = cross_val_score(pipeline, X_data, y_data, cv=kf, scoring="neg_mean_squared_error")
    r2 = cross_val_score(pipeline, X_data, y_data, cv=kf, scoring="r2")
    return {
        "CV_RMSE": float(np.sqrt(-neg_mse.mean())),
        "CV_RMSE_std": float(np.sqrt(-neg_mse).std()),
        "CV_R2": float(r2.mean()),
        "CV_R2_std": float(r2.std()),
    }


def fit_and_score(model, name, X_tr, X_te, y_tr, y_te, X_full, y_full):
    # Пайплайн: стандартизация + модель (для CV на исходных данных)
    pipe = Pipeline([("scaler", StandardScaler()), ("model", model)])
    pipe.fit(X_tr, y_tr)
    y_pred = pipe.predict(X_te)
    metrics = evaluate(y_te, y_pred, name)
    metrics.update(cv_scores(pipe, X_full, y_full))
    return metrics, y_pred, pipe


# ---------------------------------------------------------------------------
# 5. Модели на исходных признаках
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("5. РЕГРЕССИЯ НА ИСХОДНЫХ ПРИЗНАКАХ")
print("=" * 60)

models = {
    "Линейная": LinearRegression(),
    "Lasso": Lasso(alpha=1000.0, random_state=RANDOM_STATE, max_iter=10000),
    "Гребневая (Ridge)": Ridge(alpha=1.0, random_state=RANDOM_STATE),
}

results_raw = []
preds_raw = {}
for name, model in models.items():
    metrics, y_pred, pipe = fit_and_score(
        model, name, X_train, X_test, y_train, y_test, X, y
    )
    results_raw.append(metrics)
    preds_raw[name] = y_pred
    print(f"\n{name}:")
    print(f"  Test RMSE  = {metrics['RMSE']:.2f}")
    print(f"  Test R2    = {metrics['R2']:.4f}")
    print(f"  Test MAPE  = {metrics['MAPE_%']:.2f}%")
    print(f"  CV  RMSE   = {metrics['CV_RMSE']:.2f} ± {metrics['CV_RMSE_std']:.2f}")
    print(f"  CV  R2     = {metrics['CV_R2']:.4f} ± {metrics['CV_R2_std']:.4f}")

df_raw = pd.DataFrame(results_raw)
print("\nСводная таблица (исходные признаки):")
print(df_raw.round(4).to_string(index=False))

# График сравнения метрик
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, metric, title in zip(
    axes, ["RMSE", "R2", "MAPE_%"], ["RMSE (down лучше)", "R2 (up лучше)", "MAPE % (down лучше)"]
):
    sns.barplot(data=df_raw, x="Модель", y=metric, ax=ax, hue="Модель", legend=False, palette="Blues_d")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=15)
fig.suptitle("Метрики на исходных признаках (тестовая выборка)", fontsize=14)
plt.tight_layout()
plt.savefig(FIG_DIR / "04_metrics_raw.png", dpi=150, bbox_inches="tight")
plt.close()

# Фактические vs предсказанные
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (name, y_pred) in zip(axes, preds_raw.items()):
    ax.scatter(y_test, y_pred, alpha=0.6)
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", lw=2)
    ax.set_xlabel("Факт")
    ax.set_ylabel("Прогноз")
    ax.set_title(name)
fig.suptitle("Факт vs прогноз (исходные признаки)", fontsize=14)
plt.tight_layout()
plt.savefig(FIG_DIR / "05_pred_vs_actual_raw.png", dpi=150, bbox_inches="tight")
plt.close()
print("Сохранены: 04_metrics_raw.png, 05_pred_vs_actual_raw.png")

# ---------------------------------------------------------------------------
# 6. PCA
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("6. PCA: СТАНДАРТИЗАЦИЯ И СНИЖЕНИЕ РАЗМЕРНОСТИ")
print("=" * 60)

pca_full = PCA()
X_train_pca_full = pca_full.fit_transform(X_train_scaled)
explained = pca_full.explained_variance_ratio_
cumsum = np.cumsum(explained)

print("Доля объяснённой дисперсии по компонентам:")
for i, (ev, cs) in enumerate(zip(explained, cumsum), 1):
    print(f"  PC{i}: {ev:.4f} ({ev * 100:.2f}%), накопленно: {cs * 100:.2f}%")

# Каменистая осыпь (scree plot)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].bar(range(1, len(explained) + 1), explained * 100, color="steelblue")
axes[0].plot(range(1, len(explained) + 1), explained * 100, "o-", color="darkred")
axes[0].set_xlabel("Главная компонента")
axes[0].set_ylabel("Объяснённая дисперсия, %")
axes[0].set_title("График каменистой осыпи (Scree plot)")
axes[0].set_xticks(range(1, len(explained) + 1))

axes[1].plot(range(1, len(cumsum) + 1), cumsum * 100, "o-", color="darkgreen")
axes[1].axhline(90, color="red", linestyle="--", label="90%")
axes[1].axhline(95, color="orange", linestyle="--", label="95%")
axes[1].set_xlabel("Число компонент")
axes[1].set_ylabel("Накопленная дисперсия, %")
axes[1].set_title("Накопленная объяснённая дисперсия")
axes[1].legend()
axes[1].set_xticks(range(1, len(cumsum) + 1))
plt.tight_layout()
plt.savefig(FIG_DIR / "06_pca_scree.png", dpi=150, bbox_inches="tight")
plt.close()

# По scree plot: после PC2 прирост дисперсии заметно падает.
# Берём 2 компоненты (~87% дисперсии) — реальное снижение размерности 3 -> 2.
# Для порога 95% потребовались бы все 3 компоненты (тогда PCA не снижает размерность).
n_components = 2
print(f"\nВыбрано компонент: {n_components} (по графику каменистой осыпи)")
print(f"Объясняют: {cumsum[n_components - 1] * 100:.2f}% дисперсии")
print(f"Для порога 95% нужно было бы {int(np.argmax(cumsum >= 0.95) + 1)} компоненты.")

pca = PCA(n_components=n_components)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# VIF после PCA (на главных компонентах — по определению ортогональны)
vif_pca = pd.DataFrame(
    {
        "Компонента": [f"PC{i}" for i in range(1, n_components + 1)],
        "VIF": [variance_inflation_factor(X_train_pca, i) for i in range(n_components)],
    }
)
print("\nVIF после PCA (ожидаем ≈1 — компоненты ортогональны):")
print(vif_pca.to_string(index=False))

loadings = pd.DataFrame(
    pca.components_.T,
    columns=[f"PC{i}" for i in range(1, n_components + 1)],
    index=features,
)
print("\nНагрузки главных компонент:")
print(loadings.round(3))
print("Сохранён: 06_pca_scree.png")

# ---------------------------------------------------------------------------
# 7. Модели на главных компонентах
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("7. РЕГРЕССИЯ НА ГЛАВНЫХ КОМПОНЕНТАХ")
print("=" * 60)


def fit_and_score_pca(model, name, X_tr, X_te, y_tr, y_te, X_full_scaled, y_full):
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    metrics = evaluate(y_te, y_pred, name)

    # CV: каждый фолд — свой scaler + PCA + модель
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=n_components)),
            ("model", type(model)(**model.get_params())),
        ]
    )
    metrics.update(cv_scores(pipe, X, y_full))
    return metrics, y_pred


results_pca = []
preds_pca = {}
models_pca = {
    "Линейная": LinearRegression(),
    "Lasso": Lasso(alpha=1000.0, random_state=RANDOM_STATE, max_iter=10000),
    "Гребневая (Ridge)": Ridge(alpha=1.0, random_state=RANDOM_STATE),
}

for name, model in models_pca.items():
    metrics, y_pred = fit_and_score_pca(
        model, name, X_train_pca, X_test_pca, y_train, y_test, X_train_scaled, y
    )
    results_pca.append(metrics)
    preds_pca[name] = y_pred
    print(f"\n{name} (PCA):")
    print(f"  Test RMSE  = {metrics['RMSE']:.2f}")
    print(f"  Test R2    = {metrics['R2']:.4f}")
    print(f"  Test MAPE  = {metrics['MAPE_%']:.2f}%")
    print(f"  CV  RMSE   = {metrics['CV_RMSE']:.2f} ± {metrics['CV_RMSE_std']:.2f}")
    print(f"  CV  R2     = {metrics['CV_R2']:.4f} ± {metrics['CV_R2_std']:.4f}")

df_pca = pd.DataFrame(results_pca)
print("\nСводная таблица (главные компоненты):")
print(df_pca.round(4).to_string(index=False))

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, metric, title in zip(
    axes, ["RMSE", "R2", "MAPE_%"], ["RMSE (down лучше)", "R2 (up лучше)", "MAPE % (down лучше)"]
):
    sns.barplot(data=df_pca, x="Модель", y=metric, ax=ax, hue="Модель", legend=False, palette="Greens_d")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=15)
fig.suptitle("Метрики на главных компонентах (тестовая выборка)", fontsize=14)
plt.tight_layout()
plt.savefig(FIG_DIR / "07_metrics_pca.png", dpi=150, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# 8. Сравнение
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("8. СРАВНЕНИЕ: ИСХОДНЫЕ ПРИЗНАКИ vs PCA")
print("=" * 60)

df_raw["Вариант"] = "Исходные признаки"
df_pca["Вариант"] = "Главные компоненты"
df_cmp = pd.concat([df_raw, df_pca], ignore_index=True)

print(df_cmp[["Вариант", "Модель", "RMSE", "R2", "MAPE_%", "CV_RMSE", "CV_R2"]].round(4).to_string(index=False))
df_cmp.to_csv("metrics_comparison.csv", index=False, encoding="utf-8-sig")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, metric, title in zip(
    axes, ["RMSE", "R2", "MAPE_%"], ["RMSE (down лучше)", "R2 (up лучше)", "MAPE % (down лучше)"]
):
    sns.barplot(data=df_cmp, x="Модель", y=metric, hue="Вариант", ax=ax)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=15)
    ax.legend(fontsize=8)
fig.suptitle("Сравнение качества: исходные признаки vs PCA", fontsize=14)
plt.tight_layout()
plt.savefig(FIG_DIR / "08_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nСохранены: 07_metrics_pca.png, 08_comparison.png, metrics_comparison.csv")

print("\n" + "=" * 60)
print("ГОТОВО")
print("=" * 60)
print(f"Графики: {FIG_DIR.resolve()}")


