import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, recall_score,
    precision_score, confusion_matrix, roc_curve, brier_score_loss,
)
from sklearn.model_selection import StratifiedGroupKFold, RandomizedSearchCV, cross_val_predict
from sklearn.calibration import calibration_curve

# Variables que NUNCA deben usarse como predictoras en un modelo dinámico por
# landmark, aunque hayan sido válidas como variables explicativas en Fase 2:
#
# - aprobo_semestre / materias_reprobadas_este_semestre: se calculan con el
#   aprobo_materia REAL de fin de semestre (feature_engineer.build_landmark_
#   snapshots), por lo que permanecen constantes en los 3 landmarks -- son la
#   etiqueta de corto plazo, no informacion disponible en el landmark 1 o 2.
# - resultado_final: es el target.
# - es_censurado: bandera de filtrado, no predictora.
# - cohorte_ingreso: se usa para el split train/test (simular generaciones
#   futuras); dejarla como feature rompe ese objetivo, porque una cohorte
#   futura real nunca tendra una categoria "vista" por el modelo.
_COLUMNAS_NO_PREDICTORAS = {
    "aprobo_semestre", "materias_reprobadas_este_semestre",
    "resultado_final", "es_censurado", "cohorte_ingreso",
    "alumno_carrera_hash",
}

# Excluida por fuga de informacion documentada en Fase 2 (separacion
# cuasi-perfecta), no por multicolinealidad -- se mantiene fuera aunque los
# modelos de arbol/ensemble toleren mejor la colinealidad que el Logit.
_PREFIJO_FUGA = "cobertura_"

_CATEGORICAS_NOMINALES = ["id_carrera"]
_CATEGORICAS_ORDINALES = {
    "categoria_asistencia": {
        "1. Riesgo (<80%)": 1, "2. Regular (80-84%)": 2, "3. Bueno (85-89%)": 3,
        "4. Muy Bueno (90-94%)": 4, "5. Excelente (>=95%)": 5,
    }
}
_BOOLEANAS = ["cambio_carrera", "carrera_abandonada"]


def get_predictor_features(df: pd.DataFrame) -> list:
    """
    Lista de columnas utilizables como predictoras en el modelo dinámico de
    Fase 3 (Opción B: un solo modelo con `landmark_parcial` como feature).
    """
    excluidas = set(_COLUMNAS_NO_PREDICTORAS)
    excluidas |= {c for c in df.columns if c.startswith(_PREFIJO_FUGA)}
    return [c for c in df.columns if c not in excluidas]


def split_by_cohort(df: pd.DataFrame, test_cohorts: list):
    """
    Divide las instantáneas por landmark en train/test según `cohorte_ingreso`,
    simulando la predicción sobre generaciones futuras: el modelo nunca ve
    durante el entrenamiento ninguna cohorte incluida en `test_cohorts`.
    Al ser la cohorte una propiedad fija del alumno, este split garantiza
    además que ningún alumno aparezca simultáneamente en train y test.
    """
    es_test = df["cohorte_ingreso"].isin(test_cohorts)
    train_df = df[~es_test].reset_index(drop=True)
    test_df = df[es_test].reset_index(drop=True)
    return train_df, test_df


def build_dynamic_design_matrix(df: pd.DataFrame, features: list) -> pd.DataFrame:
    """
    Construye la matriz de diseño para el modelo dinámico de Fase 3. A
    diferencia de `model_explainer.build_combined_design_matrix` (pensado
    para preservar Odds Ratios interpretables en el Logit de Fase 2), aquí
    no hay restricción de interpretabilidad: `id_carrera` se codifica como
    dummy completo (sin excluir categoría de referencia), pues no se ajustará
    ningún modelo lineal sensible a colinealidad.
    """
    numeric_and_bool = [
        f for f in features
        if f not in _CATEGORICAS_NOMINALES and f not in _CATEGORICAS_ORDINALES
    ]
    X = df[numeric_and_bool].copy()
    for col in _BOOLEANAS:
        if col in X.columns:
            X[col] = X[col].astype(int)

    for columna, mapping in _CATEGORICAS_ORDINALES.items():
        if columna in features:
            X[columna] = df[columna].map(mapping).astype(int)

    nominales_presentes = [f for f in _CATEGORICAS_NOMINALES if f in features]
    if nominales_presentes:
        X_nominal = pd.get_dummies(
            df[nominales_presentes].astype("category"), drop_first=False, dtype=int
        )
        X = pd.concat([X, X_nominal], axis=1)

    return X


def build_catboost_design_matrix(df: pd.DataFrame, features: list):
    """
    Variante de la matriz de diseño para CatBoost: en vez de dummificar
    `id_carrera` (perdiendo la ventaja nativa de CatBoost sobre categóricas),
    se conserva como columna de texto y se declara en `cat_features` al
    ajustar el modelo. Retorna la matriz y la lista de nombres de columnas
    categóricas para pasar a `CatBoostClassifier(cat_features=...)`.
    """
    numeric_and_bool = [
        f for f in features
        if f not in _CATEGORICAS_NOMINALES and f not in _CATEGORICAS_ORDINALES
    ]
    X = df[numeric_and_bool].copy()
    for col in _BOOLEANAS:
        if col in X.columns:
            X[col] = X[col].astype(int)

    for columna, mapping in _CATEGORICAS_ORDINALES.items():
        if columna in features:
            X[columna] = df[columna].map(mapping).astype(int)

    columnas_categoricas = [f for f in _CATEGORICAS_NOMINALES if f in features]
    for columna in columnas_categoricas:
        X[columna] = df[columna].astype(str)

    return X, columnas_categoricas


def get_baseline_candidate_models(y_train: pd.Series, random_state: int = 42) -> dict:
    """
    Construye el diccionario de modelos candidatos para la comparación base
    de Fase 3 (antes de tuning): dos ensembles de árboles ya evaluados
    (Random Forest, LightGBM), y dos candidatos adicionales:

    - CatBoost: gradient boosting con soporte nativo de categóricas -- se
      evalúa aparte con `build_catboost_design_matrix` para aprovechar esa
      ventaja sobre `id_carrera` en vez de forzarlo a dummies.
    - Regresión Logística: baseline lineal simple, para cuantificar cuánto
      aporta realmente la complejidad de los ensembles de árboles sobre un
      modelo lineal directo con el mismo conjunto de variables.

    XGBoost se maneja aparte porque es el único que pasa por tuning de
    hiperparámetros (`tune_xgboost_hyperparameters`).
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from lightgbm import LGBMClassifier
    from catboost import CatBoostClassifier

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=500, class_weight="balanced", random_state=random_state, n_jobs=-1,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=500, learning_rate=0.05, class_weight="balanced",
            random_state=random_state, n_jobs=-1, verbosity=-1,
        ),
        "CatBoost": CatBoostClassifier(
            iterations=500, learning_rate=0.05, depth=6,
            scale_pos_weight=scale_pos_weight, random_state=random_state,
            verbose=False,
        ),
        "LogisticRegression": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                class_weight="balanced", max_iter=2000, random_state=random_state,
            ),
        ),
    }


def evaluar_modelo(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Calcula el conjunto de métricas relevante para un problema desbalanceado
    (79% Graduación / 21% Deserción): ROC-AUC y PR-AUC globales, y precisión/
    recall/F1 enfocados en la clase de interés real para un sistema de alerta
    temprana -- Deserción (0) -- donde un falso negativo (predecir que se
    graduará a quien va a desertar) es el error más costoso de ignorar.
    """
    proba = model.predict_proba(X_test)[:, 1]
    pred = model.predict(X_test)

    return {
        "ROC-AUC": roc_auc_score(y_test, proba),
        "PR-AUC": average_precision_score((y_test == 0).astype(int), 1 - proba),
        "Precision (Deserción)": precision_score(y_test, pred, pos_label=0, zero_division=0),
        "Recall (Deserción)": recall_score(y_test, pred, pos_label=0, zero_division=0),
        "F1 (Deserción)": f1_score(y_test, pred, pos_label=0, zero_division=0),
        "Matriz de confusión": confusion_matrix(y_test, pred),
    }


def evaluar_por_landmark(model, X_test: pd.DataFrame, y_test: pd.Series, landmark_test: pd.Series) -> pd.DataFrame:
    """
    Desglosa el desempeño del modelo dinámico por cada uno de los 3 momentos
    de revisión de Tutorías (landmark_parcial 1, 2, 3), para documentar en la
    tesis que la certeza de la predicción aumenta conforme se dispone de más
    información dentro del semestre en curso.
    """
    filas = []
    for landmark in sorted(landmark_test.unique()):
        mascara = landmark_test == landmark
        metrics = evaluar_modelo(model, X_test[mascara], y_test[mascara])
        filas.append({
            "Landmark": landmark,
            "N": int(mascara.sum()),
            "ROC-AUC": metrics["ROC-AUC"],
            "PR-AUC": metrics["PR-AUC"],
            "Precision (Deserción)": metrics["Precision (Deserción)"],
            "Recall (Deserción)": metrics["Recall (Deserción)"],
            "F1 (Deserción)": metrics["F1 (Deserción)"],
        })
    return pd.DataFrame(filas)


def train_option_a_models(
    train_df: pd.DataFrame, test_df: pd.DataFrame, predictor_features: list,
    n_iter: int = 20, cv_splits: int = 5, random_state: int = 42,
) -> dict:
    """
    Opción A: un modelo de Regresión Logística (L1) **independiente por cada
    landmark** (1, 2, 3), en vez del modelo único de la Opción B. Se usa la
    misma familia de modelo que ganó en la Opción B (no XGBoost) para aislar
    la pregunta real de esta comparación -- ¿ayuda especializar por landmark,
    o pesa más perder dos tercios de los datos de entrenamiento en cada uno?
    -- de la pregunta, ya resuelta, de qué algoritmo es mejor.

    `landmark_parcial` se excluye de las features: dentro de cada landmark
    por separado es una columna constante, sin información que aportar.

    Retorna un dict {landmark: {"model", "params", "metrics", "n_train", "n_test"}}.
    """
    features_sin_landmark = [f for f in predictor_features if f != "landmark_parcial"]
    resultados = {}

    for landmark in sorted(train_df["landmark_parcial"].unique()):
        print(f"\n=== Opción A: entrenando modelo del landmark {landmark} ===")
        train_lm = train_df[train_df["landmark_parcial"] == landmark]
        test_lm = test_df[test_df["landmark_parcial"] == landmark]

        X_train_lm = build_dynamic_design_matrix(train_lm, features_sin_landmark)
        X_test_lm = build_dynamic_design_matrix(test_lm, features_sin_landmark)
        X_test_lm = X_test_lm.reindex(columns=X_train_lm.columns, fill_value=0)

        y_train_lm = train_lm["resultado_final"].astype(int)
        y_test_lm = test_lm["resultado_final"].astype(int)
        groups_lm = train_lm["alumno_carrera_hash"]

        model, params, cv_score = tune_logistic_regression_hyperparameters(
            X_train_lm, y_train_lm, groups_lm, n_iter=n_iter, cv_splits=cv_splits,
            random_state=random_state,
        )
        metrics = evaluar_modelo(model, X_test_lm, y_test_lm)
        print({k: v for k, v in metrics.items() if k != "Matriz de confusión"})

        resultados[landmark] = {
            "model": model, "params": params, "metrics": metrics,
            "n_train": len(train_lm), "n_test": len(test_lm),
        }

    return resultados


def _pr_auc_desercion_scorer(estimator, X, y):
    """
    Scorer para RandomizedSearchCV enfocado en la clase de interés real
    (Deserción=0), no en la clase mayoritaria (Graduación=1) que usan las
    métricas por defecto de sklearn.
    """
    proba_graduacion = estimator.predict_proba(X)[:, 1]
    return average_precision_score((y == 0).astype(int), 1 - proba_graduacion)


def tune_xgboost_hyperparameters(
    X_train: pd.DataFrame, y_train: pd.Series, groups: pd.Series,
    n_iter: int = 40, cv_splits: int = 5, random_state: int = 42,
):
    """
    Búsqueda aleatoria de hiperparámetros para XGBoost con validación cruzada
    agrupada por alumno (`StratifiedGroupKFold`): un mismo estudiante aporta
    varias filas (una por semestre x landmark) y no debe quedar repartido
    entre el pliegue de entrenamiento y el de validación dentro del CV, o el
    ajuste de hiperparámetros heredaría la misma fuga que ya se evitó en el
    split train/test por cohorte.

    Se optimiza sobre PR-AUC de la clase Deserción (no ROC-AUC ni accuracy),
    porque es la clase minoritaria y la que le importa a un sistema de alerta
    temprana.
    """
    from xgboost import XGBClassifier

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    cv = StratifiedGroupKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    param_distributions = {
        "n_estimators": [200, 300, 500, 800],
        "max_depth": [3, 4, 5, 6, 8],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "min_child_weight": [1, 3, 5, 10],
    }

    base_model = XGBClassifier(
        scale_pos_weight=scale_pos_weight, random_state=random_state,
        eval_metric="logloss", n_jobs=-1,
    )

    search = RandomizedSearchCV(
        base_model, param_distributions, n_iter=n_iter,
        scoring=_pr_auc_desercion_scorer, cv=cv, random_state=random_state,
        n_jobs=-1, verbose=1,
    )
    search.fit(X_train, y_train, groups=groups)

    print(f"Mejor PR-AUC (Deserción) en CV: {search.best_score_:.4f}")
    print("Mejores hiperparámetros:", search.best_params_)

    return search.best_estimator_, search.best_params_, search.best_score_


def tune_logistic_regression_hyperparameters(
    X_train: pd.DataFrame, y_train: pd.Series, groups: pd.Series,
    n_iter: int = 20, cv_splits: int = 5, random_state: int = 42,
):
    """
    Búsqueda aleatoria de hiperparámetros para la Regresión Logística
    (fuerza de regularización `C` y tipo de penalización L1/L2), con la misma
    validación cruzada agrupada por alumno y el mismo scorer (PR-AUC de
    Deserción) que `tune_xgboost_hyperparameters`, para que la comparación
    entre ambos candidatos tuneados sea justa.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    cv = StratifiedGroupKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    param_distributions = {
        "logisticregression__C": np.logspace(-3, 2, 30),
        "logisticregression__penalty": ["l1", "l2"],
    }

    base_pipeline = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced", max_iter=2000, solver="liblinear", random_state=random_state,
        ),
    )

    search = RandomizedSearchCV(
        base_pipeline, param_distributions, n_iter=n_iter,
        scoring=_pr_auc_desercion_scorer, cv=cv, random_state=random_state,
        n_jobs=-1, verbose=1,
    )
    search.fit(X_train, y_train, groups=groups)

    print(f"Mejor PR-AUC (Deserción) en CV: {search.best_score_:.4f}")
    print("Mejores hiperparámetros:", search.best_params_)

    return search.best_estimator_, search.best_params_, search.best_score_


def get_feature_importance(model, feature_names: list) -> pd.Series:
    """
    Extrae una serie de importancia de variables sin importar si `model` es
    un ensemble de árboles (`feature_importances_`) o una Regresión Logística
    dentro de un Pipeline (`coef_`, se reporta el valor absoluto porque el
    signo ya no es el interés aquí -- eso ya se documentó como Odds Ratio en
    Fase 2, aquí solo importa la magnitud de la contribución predictiva).
    """
    if hasattr(model, "feature_importances_"):
        valores = model.feature_importances_
    elif hasattr(model, "named_steps"):
        ultimo_paso = list(model.named_steps.values())[-1]
        valores = np.abs(ultimo_paso.coef_[0])
    else:
        valores = np.abs(model.coef_[0])
    return pd.Series(valores, index=feature_names)


def optimize_threshold_by_cv(
    model, X_train: pd.DataFrame, y_train: pd.Series, groups: pd.Series,
    cv_splits: int = 5, random_state: int = 42,
) -> tuple:
    """
    Determina el umbral de decisión que maximiza el F1 de la clase Deserción,
    usando probabilidades out-of-fold obtenidas por validación cruzada
    agrupada por alumno sobre el propio conjunto de entrenamiento -- nunca
    sobre el test, para no contaminar la evaluación final con una decisión
    ajustada a esos datos.

    El umbral por defecto de 0.5 no tiene ninguna razón estadística especial
    de ser el óptimo, sobre todo con clases desbalanceadas y `scale_pos_
    weight`, que desplaza las probabilidades predichas.
    """
    cv = StratifiedGroupKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    proba_oof = cross_val_predict(
        model, X_train, y_train, groups=groups, cv=cv, method="predict_proba", n_jobs=-1,
    )[:, 1]

    candidatos = np.arange(0.05, 0.96, 0.01)
    resultados = []
    for umbral in candidatos:
        pred = (proba_oof >= umbral).astype(int)
        f1 = f1_score(y_train, pred, pos_label=0, zero_division=0)
        resultados.append({"Umbral": umbral, "F1 (Deserción)": f1})

    tabla = pd.DataFrame(resultados)
    mejor_umbral = tabla.loc[tabla["F1 (Deserción)"].idxmax(), "Umbral"]
    print(f"Umbral óptimo (por F1 de Deserción, out-of-fold en train): {mejor_umbral:.2f}")
    return float(mejor_umbral), tabla


def plot_calibration_curve(y_test: pd.Series, proba: np.ndarray, model_name: str = "Modelo", n_bins: int = 10):
    """
    Diagrama de confiabilidad: compara la probabilidad predicha de graduación
    contra la fracción observada de graduados en cada decil de probabilidad.
    Relevante si Tutorías va a usar el puntaje de riesgo directamente (no solo
    la clase predicha) -- un modelo con buen ROC-AUC puede seguir estando mal
    calibrado, sobre todo tras aplicar `scale_pos_weight` para balancear
    clases, que reescala la función de pérdida y desplaza las probabilidades
    crudas respecto a su frecuencia real.
    """
    frac_observada, prob_media_predicha = calibration_curve(
        y_test, proba, n_bins=n_bins, strategy="quantile"
    )
    brier = brier_score_loss(y_test, proba)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(prob_media_predicha, frac_observada, marker="o", label=f"{model_name} (Brier={brier:.4f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Calibración perfecta")
    ax.set_xlabel("Probabilidad de Graduación predicha (promedio por bin)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Fracción observada de graduados", fontsize=12, fontweight="bold")
    ax.set_title(f"Curva de Calibración: {model_name}", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.show()

    return brier


def plot_roc_curves(models_dict: dict, X_test: pd.DataFrame, y_test: pd.Series):
    """
    Grafica la curva ROC de cada modelo candidato sobre el mismo test set,
    para comparar visualmente su poder predictivo global.
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    for nombre, model in models_dict.items():
        proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, proba)
        auc = roc_auc_score(y_test, proba)
        ax.plot(fpr, tpr, label=f"{nombre} (AUC={auc:.3f})")

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Azar (AUC=0.5)")
    ax.set_xlabel("Tasa de Falsos Positivos", fontsize=12, fontweight="bold")
    ax.set_ylabel("Tasa de Verdaderos Positivos", fontsize=12, fontweight="bold")
    ax.set_title("Curvas ROC: Modelos Predictivos Dinámicos (Fase 3)", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.show()
