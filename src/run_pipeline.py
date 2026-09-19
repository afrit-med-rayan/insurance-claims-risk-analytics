"""
End-to-End Automation Pipeline.

Executes all project steps in order:
1. Data loading and merging
2. Data cleaning
3. Feature engineering
4. Exploratory Data Analysis (EDA)
5. Frequency Modeling
6. Severity Modeling
7. SHAP Explainability
"""

import time


def print_step(num: int, total: int, name: str) -> None:
    print(f"\n[{num:02d}/{total:02d}] {name}...")
    print("-" * 50)


def run_pipeline() -> None:
    start_time = time.time()
    total_steps = 7
    
    print("Starting Insurance Risk Analytics Pipeline")
    print("=" * 50)
    
    # Step 1
    print_step(1, total_steps, "Loading data")
    from src import load_data
    merged = load_data.load_and_merge()
    
    # Step 2
    print_step(2, total_steps, "Cleaning data")
    from src import clean_data
    cleaned = clean_data.clean(merged)
    
    # Step 3
    print_step(3, total_steps, "Engineering features")
    from src import feature_engineering
    # We call run() directly since it loads the cleaned data, adds features, and saves them
    feature_engineering.run()
    
    # Step 4
    print_step(4, total_steps, "Running EDA")
    from src import eda
    eda.run()
    
    # Step 5
    print_step(5, total_steps, "Training frequency model")
    from src import model_frequency
    model_frequency.run()
    
    # Step 6
    print_step(6, total_steps, "Training severity model")
    from src import model_severity
    model_severity.run()
    
    # Step 7
    print_step(7, total_steps, "Computing SHAP explainability")
    from src import explainability
    explainability.run()
    
    elapsed = time.time() - start_time
    print("\n" + "=" * 50)
    print(f"Pipeline completed successfully in {elapsed:.1f} seconds.")
    print("All outputs written to data/processed/ and reports/")


if __name__ == "__main__":
    run_pipeline()
