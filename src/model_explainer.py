import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import chi2_contingency
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


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """
    Calcula el coeficiente V de Cramér, con la corrección de sesgo de Bergsma-Wicher,
    para medir la fuerza de asociación entre dos variables categóricas.
    Retorna un valor en [0, 1]: 0 = independencia total, 1 = asociación perfecta.
    Es el equivalente categórico de un coeficiente de correlación.
    """
    contingency = pd.crosstab(x, y)
    n = contingency.sum().sum()
    r, k = contingency.shape

    if r < 2 or k < 2 or n == 0:
        return 0.0

    chi2 = chi2_contingency(contingency, correction=False)[0]
    phi2 = chi2 / n

    phi2_corr = max(0.0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    r_corr = r - ((r - 1) ** 2) / (n - 1)
    k_corr = k - ((k - 1) ** 2) / (n - 1)
    denom = min(k_corr - 1, r_corr - 1)

    if denom <= 0:
        return 0.0

    return float(np.sqrt(phi2_corr / denom))


def calculate_cramers_v_matrix(df: pd.DataFrame, features: list) -> pd.DataFrame:
    """
    Calcula la matriz de V de Cramér entre todas las variables categóricas dadas.
    Es el equivalente categórico de una matriz de correlación: valores altos entre
    dos variables candidatas señalan redundancia (multicolinealidad categórica).
    """
    matrix = pd.DataFrame(index=features, columns=features, dtype=float)
    for var_a in features:
        for var_b in features:
            if var_a == var_b:
                matrix.loc[var_a, var_b] = 1.0
            else:
                matrix.loc[var_a, var_b] = cramers_v(df[var_a], df[var_b])
    return matrix


def plot_cramers_v_heatmap(df: pd.DataFrame, features: list) -> pd.DataFrame:
    """
    Genera un mapa de calor de la matriz de V de Cramér entre variables categóricas
    (para detectar redundancia entre ellas) y retorna la matriz calculada.
    """
    matrix = calculate_cramers_v_matrix(df, features)
    values = matrix.values.astype(float)

    fig, ax = plt.subplots(figsize=(1.2 * len(features) + 3, 1.2 * len(features) + 2))
    im = ax.imshow(values, cmap="Purples", vmin=0, vmax=1)

    ax.set_xticks(range(len(features)))
    ax.set_yticks(range(len(features)))
    ax.set_xticklabels(features, rotation=45, ha="right")
    ax.set_yticklabels(features)

    for i in range(len(features)):
        for j in range(len(features)):
            ax.text(
                j,
                i,
                f"{values[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if values[i, j] > 0.5 else "black",
                fontsize=9,
            )

    ax.set_title(
        "Matriz de Asociación entre Variables Categóricas (V de Cramér)",
        fontsize=13,
        fontweight="bold",
    )
    fig.colorbar(im, ax=ax, label="V de Cramér")
    plt.tight_layout()
    plt.show()

    return matrix


def select_categorical_features(
    df: pd.DataFrame, features: list, target: str, min_effect_size: float = 0.05
) -> pd.DataFrame:
    """
    Evalúa la relevancia de cada variable categórica frente al target mediante una
    prueba Chi-cuadrada de independencia, usando el tamaño de efecto (V de Cramér)
    como criterio principal de conservación.

    Con datasets grandes casi cualquier variable resulta "significativa" (p<0.05)
    aunque su efecto real sea trivial; por eso una variable solo se conserva si,
    además de ser significativa, su V de Cramér alcanza `min_effect_size`.
    Genera un gráfico de barras análogo al de VIF, para consistencia visual en tesis.
    """
    resultados = []

    for feature in features:
        contingency = pd.crosstab(df[feature], df[target])
        chi2, p_value, dof, _ = chi2_contingency(contingency)
        v = cramers_v(df[feature], df[target])

        resultados.append(
            {
                "Variable": feature,
                "Chi2": chi2,
                "Grados de Libertad": dof,
                "P>|chi2|": p_value,
                "V de Cramér": v,
            }
        )

    tabla = pd.DataFrame(resultados)
    tabla["Estado"] = np.where(
        (tabla["V de Cramér"] >= min_effect_size) & (tabla["P>|chi2|"] < 0.05),
        "Conservada",
        "Eliminada",
    )
    tabla = tabla.sort_values(by="V de Cramér", ascending=True).reset_index(drop=True)

    colors = np.where(tabla["Estado"] == "Conservada", "#4A90E2", "#aa0000")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(tabla["Variable"], tabla["V de Cramér"], color=colors, edgecolor="black")
    ax.axvline(
        x=min_effect_size,
        color="black",
        linestyle="--",
        linewidth=2,
        label=f"Umbral V de Cramér ({min_effect_size})",
    )
    ax.set_xlabel("V de Cramér (Tamaño de Efecto)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Variables Institucionales", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Relevancia de Variables Categóricas sobre '{target}'",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.show()

    return (
        tabla[["Variable", "Chi2", "Grados de Libertad", "P>|chi2|", "V de Cramér", "Estado"]]
        .sort_values(by="V de Cramér", ascending=False)
        .reset_index(drop=True)
    )


def build_combined_design_matrix(
    df: pd.DataFrame,
    numeric_features: list,
    categorical_features: list,
    ordinal_mappings: dict = None,
) -> pd.DataFrame:
    """
    Combina las variables numéricas (ya depuradas por VIF) y categóricas (ya
    depuradas por V de Cramér) en una sola matriz de diseño.

    Las categóricas nominales se codifican como dummies (categoría de referencia
    excluida con drop_first). Las categóricas con un orden natural (ej.
    `categoria_asistencia`: Riesgo < Regular < Bueno < Muy Bueno < Excelente) deben
    pasarse en `ordinal_mappings` como {"columna": {"etiqueta": rango_numérico}} en
    vez de dejarlas como nominales: codificarlas con dummies permite que VIF o el
    stepwise eliminen niveles intermedios de forma independiente, lo que rompe la
    categoría de referencia implícita de los niveles restantes y vuelve el Odds
    Ratio final ilegible. Como columna ordinal única, el modelo produce un solo
    Odds Ratio interpretable ("por cada nivel adicional...").
    """
    ordinal_mappings = ordinal_mappings or {}
    nominal_features = [f for f in categorical_features if f not in ordinal_mappings]

    X_numeric = df[numeric_features].copy()
    for column, mapping in ordinal_mappings.items():
        # .map() sobre una Serie de dtype "category" preserva ese dtype (no se
        # vuelve numérica sola); se fuerza a entero para que VIF/Logit la traten
        # como numérica real.
        X_numeric[column] = df[column].map(mapping).astype(int)

    if not nominal_features:
        return X_numeric

    # pd.get_dummies solo codifica columnas object/category/string: una nominal
    # guardada como entero (id_carrera) o booleano (cambio_carrera) pasaría sin
    # tocarse, colándose como si tuviera una escala numérica real. Se fuerza el
    # dtype "category" para garantizar que toda nominal se dummifique.
    X_nominal_source = df[nominal_features].astype("category")
    X_nominal = pd.get_dummies(X_nominal_source, drop_first=True, dtype=int)
    return pd.concat([X_numeric, X_nominal], axis=1)


def stepwise_logit_selection(X: pd.DataFrame, y: pd.Series, significance_level: float = 0.05):
    """
    Selección hacia atrás (backward elimination) por p-valor sobre un modelo Logit:
    ajusta el modelo con todas las variables y elimina iterativamente la menos
    significativa hasta que todas las restantes sean significativas
    (p < significance_level). Se prefiere sobre un Lasso para el modelo
    explicativo porque conserva la interpretación directa de Odds Ratios.

    Retorna el modelo final ajustado y la bitácora de variables eliminadas, para
    documentar el proceso de depuración en la tesis.
    """
    features = list(X.columns)
    eliminadas = []

    while True:
        X_const = sm.add_constant(X[features])
        model = sm.Logit(y, X_const).fit(disp=False)
        pvalues = model.pvalues.drop("const")

        peor_pvalue = pvalues.max()
        if peor_pvalue < significance_level or len(features) == 1:
            break

        peor_variable = pvalues.idxmax()
        eliminadas.append({"Variable": peor_variable, "P>|z|": peor_pvalue})
        features.remove(peor_variable)

    print(f"Variables eliminadas por selección stepwise ({len(eliminadas)}):")
    for paso in eliminadas:
        print(f"  - {paso['Variable']} (P>|z| = {paso['P>|z|']:.4f})")

    return model, pd.DataFrame(eliminadas)


def extract_odds_ratios(model) -> pd.DataFrame:
    """
    Extrae la tabla de Odds Ratios (razón de momios) de un modelo Logit ajustado
    de statsmodels, con intervalos de confianza al 95% y significancia.
    """
    odds_ratios = pd.DataFrame(
        {
            "Odds Ratio": np.exp(model.params),
            "P>|z|": model.pvalues,
            "Límite Inf (2.5%)": np.exp(model.conf_int()[0]),
            "Límite Sup (97.5%)": np.exp(model.conf_int()[1]),
        }
    )
    odds_ratios["Significativo (P<0.05)"] = np.where(odds_ratios["P>|z|"] < 0.05, "Sí", "No")
    odds_ratios = odds_ratios.round(4)
    return odds_ratios.sort_values(by="Odds Ratio", ascending=False)
