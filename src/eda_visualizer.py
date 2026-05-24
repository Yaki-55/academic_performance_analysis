import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd
import numpy as np
import seaborn as sns

sns.set_theme(style="whitegrid")


def plot_grade_distribution(df: pd.DataFrame):
    """Generates subplots showing grade distributions split by academic major."""
    print("Rendering grade distributions...")
    g = sns.FacetGrid(df, col="id_carrera", col_wrap=4, height=3.5, sharey=False)
    g.map(sns.histplot, "pf", kde=True, bins=20, color="darkblue")
    g.fig.suptitle("Final Grade (pf) Distribution Across Majors", y=1.03, fontsize=16)
    g.set_axis_labels("Final Grade", "Frequency")
    g.set_titles("Major ID: {col_name}")
    plt.tight_layout()
    plt.show()


def plot_attendance_vs_grades(df: pd.DataFrame):
    """
    Plots an aggregated student-level scatter plot. Reduces crowding
    by introducing alpha transparencies and trend regressions.
    """
    print("Aggregating student levels and rendering scatter distributions...")

    # DataFrame temporary columns updated to Spanish keys for consistency
    aggregated_student_profiles = (
        df.groupby(["id_carrera", "alumno_carrera_hash"])
        .agg(
            promedio_calificacion=("pf", "mean"),
            promedio_asistencia=("pa", "mean"),
        )
        .reset_index()
    )

    g = sns.FacetGrid(aggregated_student_profiles, col="id_carrera", col_wrap=4, height=3.5)
    g.map(
        sns.scatterplot,
        "promedio_asistencia",
        "promedio_calificacion",
        alpha=0.4,
        s=15,
        color="steelblue",
    )
    g.map(
        sns.regplot,
        "promedio_asistencia",
        "promedio_calificacion",
        scatter=False,
        color="crimson",
    )

    g.fig.suptitle("Aggregated Attendance vs. Grades by Major", y=1.03, fontsize=16)
    g.set_axis_labels("Mean Attendance (%)", "Mean Grade")
    g.set_titles("Major ID: {col_name}")
    plt.tight_layout()
    plt.show()


def plot_correlation_matrix(df: pd.DataFrame):
    """
    Generates a heatmap of the Pearson correlation matrix for numerical features.
    """
    print("Calculando matriz de correlación multivariable...")
    numeric_df = df.select_dtypes(include=[np.number])

    cols_to_drop = [col for col in ["id_carrera", "matricula"] if col in numeric_df.columns]
    numeric_df = numeric_df.drop(columns=cols_to_drop)

    corr_matrix = numeric_df.corr(method="pearson")

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        cbar_kws={"label": "Coeficiente de Correlación de Pearson"},
    )
    plt.title("Matriz de Correlación de Variables Institucionales", fontsize=14, pad=15)
    plt.tight_layout()
    plt.show()


def plot_target_correlation(df: pd.DataFrame, target_col: str = "resultado_final"):
    """
    Plots a bar chart showing the correlation of all features directly against the Target.
    Since target is binary (0, 1), this acts as a Point-Biserial correlation.
    """
    print(f"Evaluando el peso predictivo lineal sobre la variable objetivo: {target_col}")
    numeric_df = df.select_dtypes(include=[np.number])

    if target_col not in numeric_df.columns:
        print(f"Error: La columna {target_col} no está en el dataset numérico.")
        return

    target_corr = numeric_df.corr()[target_col].drop(target_col).sort_values()

    plt.figure(figsize=(8, 6))
    sns.barplot(x=target_corr.values, y=target_corr.index, palette="vlag")
    plt.title("Impacto Lineal de las Variables sobre la Retención Escolar", fontsize=14)
    plt.xlabel("Coeficiente de Correlación (Punto-Biserial)")
    plt.ylabel("Variables Institucionales")
    plt.axvline(x=0, color="black", linestyle="--", linewidth=1)
    plt.tight_layout()
    plt.show()


def perform_anova_study(df: pd.DataFrame, continuous_var: str, categorical_var: str):
    """
    Performs a One-Way ANOVA test to evaluate non-linear/grouped relationships.
    Example: Does the mean GPA differ significantly between Dropouts and Graduates?
    """
    print(f"\n--- ESTUDIO ANOVA: {continuous_var} agrupado por {categorical_var} ---")

    groups = [group[continuous_var].dropna().values for name, group in df.groupby(categorical_var)]

    f_stat, p_value = stats.f_oneway(*groups)

    print(f"Estadístico F: {f_stat:.4f}")
    print(f"P-Valor (p-value): {p_value:.4e}")

    if p_value < 0.05:
        print("Conclusión: Existe una diferencia ESTADÍSTICAMENTE SIGNIFICATIVA en las medias.")
        print(f"La variable '{categorical_var}' tiene un efecto real sobre '{continuous_var}'.")
    else:
        print("Conclusión: NO hay diferencia significativa. La varianza observada es azar.")

    plt.figure(figsize=(7, 5))
    sns.boxplot(data=df, x=categorical_var, y=continuous_var, palette="Set2")
    plt.title(f"Distribución de {continuous_var} por {categorical_var} (ANOVA)", fontsize=12)
    plt.tight_layout()
    plt.show()


def perform_chi_square_study(df: pd.DataFrame, cat_var: str, target_var: str = "resultado_final"):
    """
    Performs a Chi-Square Test of Independence for two categorical variables
    and plots a 100% stacked bar chart showing the retention rates per category.
    """
    print(f"\n--- ESTUDIO CHI-CUADRADA: {cat_var} vs {target_var} ---")

    contingency_table = pd.crosstab(df[cat_var], df[target_var])

    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)

    print(f"Estadístico Chi-Cuadrada: {chi2:.4f}")
    print(f"P-Valor (p-value): {p_value:.4e}")
    print(f"Grados de libertad: {dof}")

    if p_value < 0.05:
        print("Conclusión: Existe una DEPENDENCIA ESTADÍSTICAMENTE SIGNIFICATIVA.")
        print(f"La categoría de '{cat_var}' altera dramáticamente el '{target_var}'.")
    else:
        print("Conclusión: NO hay evidencia de relación estadística (Independientes).")

    crosstab_norm = contingency_table.div(contingency_table.sum(axis=1), axis=0) * 100

    crosstab_norm.plot(
        kind="bar",
        stacked=True,
        figsize=(9, 6),
        color=["#e74c3c", "#2ecc71"],
    )

    plt.title(f"Tasa de Retención Escolar por {cat_var}", fontsize=14, pad=15)
    plt.ylabel("Porcentaje de Estudiantes (%)", fontsize=12)
    plt.xlabel("Categorías", fontsize=12)

    plt.legend(
        ["0 - Deserción", "1 - Retención/Graduación"], loc="upper left", bbox_to_anchor=(1.02, 1)
    )
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def plot_predictive_power_by_category(df: pd.DataFrame):
    """
    Plots the Point-Biserial correlation specifically for the subject category averages.
    """
    print("\n--- Analizando Fuerza Predictiva por Área de Conocimiento ---")
    # Filtramos solo las columnas de promedios por categoría y el target
    cols_promedios = [col for col in df.columns if col.startswith("promedio_pf_")]
    cols_impacto = cols_promedios + ["resultado_final"]

    df_impacto = df[cols_impacto].copy()

    # Limpiamos los nombres para que la gráfica sea legible en la tesis
    df_impacto.columns = [
        (
            col.replace("promedio_pf_", "").replace("_", " ").title()
            if col != "resultado_final"
            else col
        )
        for col in df_impacto.columns
    ]

    target_corr = df_impacto.corr()["resultado_final"].drop("resultado_final").sort_values()

    plt.figure(figsize=(10, 6))
    sns.barplot(x=target_corr.values, y=target_corr.index, palette="coolwarm")
    plt.title("Fuerza Predictiva de Calificaciones por Área de Conocimiento", fontsize=14, pad=15)
    plt.xlabel("Correlación con la Graduación (Punto-Biserial)", fontsize=12)
    plt.ylabel("Área de Conocimiento", fontsize=12)
    plt.axvline(x=0, color="black", linestyle="-", linewidth=1.5)
    plt.tight_layout()
    plt.show()


def plot_kde_survival_curve(df: pd.DataFrame, col_critica: str, title_subject: str):
    """
    Plots KDE density curves to find the 'survival threshold' for a specific subject.
    Example: col_critica='promedio_pf_matemáticas_estadística', title_subject='Matemáticas'
    """
    print(f"\n--- Generando Curvas de Densidad (KDE) para {title_subject} ---")

    if col_critica in df.columns:
        plt.figure(figsize=(9, 5))
        # Filtramos los ceros (casos donde el alumno aún no cursa la materia)
        df_validos = df[df[col_critica] > 0]

        sns.kdeplot(
            data=df_validos,
            x=col_critica,
            hue="resultado_final",
            fill=True,
            common_norm=False,
            palette=["#e74c3c", "#2ecc71"],
            alpha=0.5,
        )

        plt.title(
            f"Distribución de Calificaciones en {title_subject}: Desertores vs Graduados",
            fontsize=14,
            pad=15,
        )
        plt.xlabel(f"Promedio Acumulado en {title_subject}", fontsize=12)
        plt.ylabel("Densidad de Estudiantes", fontsize=12)
        plt.legend(["1 - Graduado", "0 - Desertor"], title="Estado Final")
        plt.tight_layout()
        plt.show()
    else:
        print(f"Error: La columna {col_critica} no existe en el DataFrame.")


def plot_backlog_penalty(df: pd.DataFrame, target_categories: list):
    """
    Plots a boxplot evaluating the impact of failed subjects (backlog) per area.
    Expects a list of column names like ['materias_reprobadas_totales', ...]
    """
    print("\n--- Analizando Tolerancia al Rezago por Categoría ---")

    # Asegurarse de que las columnas existan
    valid_cols = [col for col in target_categories if col in df.columns]

    if not valid_cols:
        print("Error: Ninguna de las columnas proporcionadas existe en el DataFrame.")
        return

    df_melted = df.melt(
        id_vars=["resultado_final"],
        value_vars=valid_cols,
        var_name="Categoría de Rezago",
        value_name="Cantidad de Reprobadas",
    )

    # Limpiar nombres para el eje X
    df_melted["Categoría de Rezago"] = (
        df_melted["Categoría de Rezago"]
        .str.replace("materias_reprobadas_", "")
        .str.replace("_", " ")
        .str.title()
    )

    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=df_melted,
        x="Categoría de Rezago",
        y="Cantidad de Reprobadas",
        hue="resultado_final",
        palette=["#e74c3c", "#2ecc71"],
        showfliers=False,
    )

    plt.title(
        "Tolerancia al Rezago: Materias Reprobadas por Área de Conocimiento", fontsize=14, pad=15
    )
    plt.ylabel("Número de Materias Reprobadas", fontsize=12)
    plt.xticks(rotation=15)
    plt.legend(title="Estado Final", labels=["0 - Desertor", "1 - Graduado"])
    plt.tight_layout()
    plt.show()
