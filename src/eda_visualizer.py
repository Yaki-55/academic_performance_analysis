import matplotlib.pyplot as plt
import pandas as pd
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
