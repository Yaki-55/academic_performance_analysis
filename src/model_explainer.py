import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm


def calculate_and_plot_vif(df: pd.DataFrame, features: list, threshold: float = 5.0):
    """
    Calcula el VIF para un conjunto de variables numéricas, agregando la constante
    matemática requerida. Genera un gráfico de barras explicativo.
    """
    # 1. Filtrar solo variables numéricas y eliminar nulos
    X = df[features].select_dtypes(include=[np.number]).dropna()

    # 2. Agregar constante (Requisito estadístico indispensable para VIF)
    X_const = sm.add_constant(X)

    # 3. Calcular el VIF
    vif_data = pd.DataFrame()
    vif_data["Variable"] = X_const.columns
    vif_data["VIF"] = [
        variance_inflation_factor(X_const.values, i) for i in range(X_const.shape[1])
    ]

    # 4. Eliminar la constante de los resultados para la visualización y ordenar
    vif_data = (
        vif_data[vif_data["Variable"] != "const"]
        .sort_values(by="VIF", ascending=True)
        .reset_index(drop=True)
    )

    # 5. Determinar estado y colores
    vif_data["Estado"] = np.where(vif_data["VIF"] > threshold, "Eliminada", "Conservada")
    vif_data["Color"] = np.where(vif_data["VIF"] > threshold, "#aa0000", "#4A90E2")

    # 6. Generar gráfico
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(
        vif_data["Variable"], vif_data["VIF"], color=vif_data["Color"], edgecolor="black"
    )

    ax.axvline(
        x=threshold, color="black", linestyle="--", linewidth=2, label=f"Umbral VIF ({threshold})"
    )
    ax.set_xlabel("Factor de Inflación de la Varianza (VIF)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Variables Institucionales", fontsize=12, fontweight="bold")
    ax.set_title(
        "Análisis de Multicolinealidad (VIF) previo al Modelado", fontsize=14, fontweight="bold"
    )
    ax.legend(loc="lower right")

    for bar in bars:
        ax.text(
            bar.get_width() + 0.1,
            bar.get_y() + bar.get_height() / 2,
            f"{bar.get_width():.2f}",
            va="center",
            fontsize=10,
        )

    plt.tight_layout()
    plt.show()

    return vif_data[["Variable", "VIF", "Estado"]]


def calculate_vif(df: pd.DataFrame, features: list):
    """
    Calcula el VIF para un conjunto de variables numéricas agregando la constante
    matemática requerida, y retorna un DataFrame con los resultados limpios.
    Ideal para usarse como filtro automatizado en pipelines de Machine Learning.
    """
    # 1. Filtrar solo variables numéricas y eliminar nulos
    X = df[features].select_dtypes(include=[np.number]).dropna()

    # 2. Agregar constante (Requisito estadístico indispensable para VIF)
    X_const = sm.add_constant(X)

    # 3. Calcular el VIF para cada variable
    vif_data = pd.DataFrame()
    vif_data["Variable"] = X_const.columns
    vif_data["VIF"] = [
        variance_inflation_factor(X_const.values, i) for i in range(X_const.shape[1])
    ]

    # 4. Eliminar la constante de los resultados y ordenar
    vif_data = (
        vif_data[vif_data["Variable"] != "const"]
        .sort_values(by="VIF", ascending=True)
        .reset_index(drop=True)
    )

    return vif_data
