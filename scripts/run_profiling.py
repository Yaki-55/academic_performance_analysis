import os
import sys

# 1. Path Fix: This MUST happen BEFORE importing from src or config packages
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from data_profiling import ProfileReport
from src.data_cleaner import sanitize_grade_history
from src.db_connector import fetch_raw_grade_history
from src.feature_engineer import apply_graduation_labels, enrich_academic_dimensions


def run_academic_profiling(sample_size: int = 20000, use_sample: bool = True):
    """
    Executes the automated Exploratory Data Analysis pipeline and outputs
    a comprehensive HTML profile report.
    """
    print("=" * 60)
    print("STARTING DATA PROFILING PIPELINE")
    print("=" * 60)

    # Step 1: Data Extraction
    raw_history = fetch_raw_grade_history()

    # Step 2: Data Cleansing
    cleaned_history = sanitize_grade_history(raw_history)

    # Step 3: Feature Enrichment & Target Labeling
    enriched_history = enrich_academic_dimensions(cleaned_history)
    final_academic_dataset = apply_graduation_labels(enriched_history)

    # Step 4: Handle Data Sampling for Performance Tuning
    if use_sample:
        print(f"\n[Performance Optimization] Sampling {sample_size} rows...")
        profile_data = final_academic_dataset.sample(n=sample_size, random_state=42)
    else:
        print(
            "\n[Warning] Processing full dataset (133k+ rows). "
            "This will require significant RAM and time."
        )
        profile_data = final_academic_dataset

    print(f"Data ready for profiling. Total rows to analyze: {len(profile_data)}")

    # Step 5: Configure and Generate YData Profiling Report
    print("\nInitializing YData-Profiling engine...")
    report = ProfileReport(
        profile_data, title="Academic Performance & Retention Report - Thesis EDA", explorative=True
    )

    output_filename = "academic_profiling_report.html"
    output_path = os.path.join(ROOT_DIR, output_filename)

    print("Generating HTML report structures...")
    report.to_file(output_path)

    print("=" * 60)
    print("SUCCESS: Profiling report generated successfully!")
    print(f"Saved location: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    # You can change use_sample=False if you want to test the full 133k dataset
    run_academic_profiling(sample_size=25000, use_sample=False)
