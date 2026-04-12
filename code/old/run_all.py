""" run_all.py """
""" Master script to run all gravity model scripts in the correct order """

import subprocess
import sys
import time

# Define the scripts in execution order
scripts = [
    # Phase 1: Data Acquisition
    ("data_fetch.py", "Fetching GDP, CPI, and movie data via APIs"),

    # Phase 2: Data Cleaning
    ("gdp_data_cleaning.py", "Cleaning GDP data"),
    ("distance_data_cleaning.py", "Cleaning CEPII distance/gravity data"),
    ("efw_data_cleaning.py", "Cleaning Economic Freedom of the World data"),
    ("production_company_mappings.py", "Mapping production companies to home countries"),
    ("correcting_budgets.py", "Correcting erroneous budget values"),
    ("film_data_cleaning.py", "Cleaning film dataset"),

    # Phase 3: Data Processing
    ("build_trade_flows.py", "Building bilateral trade flows from film data"),
    ("adjust_inflation.py", "Adjusting trade values for inflation"),
    ("calculate_remoteness.py", "Calculating remoteness variables"),

    # Phase 4: Merging
    ("merge_gravity_data.py", "Merging all data into gravity dataset"),
    ("merge_film_incentives.py", "Create film incentive dummy and merge into gravity dataset"),

    # Phase 5: Analysis
    ("summary_statistics.py", "Generates summary statistics"),
    ("estimate_gravity.py", "Estimates 7 models with different specifications"),
    ("investigate_positive_distance_coeffs.py", "Investigating the drivers of the positive distance coefficient"),
    ("robustness_checks_data.py"),
    ("robustness_checks.py"),
    ("estimate_gravity_non_anglo.py"),
    ("estimate_gravity_exclude_anglo_pairs.py")
]

def run_script(script_name, description):
    """Run a single script and report result."""
    print(f"\n{'='*70}")
    print(f"RUNNING: {script_name}")
    print(f"Purpose: {description}")
    print("="*70)

    start_time = time.time()

    try:
        result = subprocess.run(
            [sys.executable, f"C:/Users/kurtl/PycharmProjects/gravity_model/code/{script_name}"],
            capture_output=True,
            text=True
        )

        elapsed = time.time() - start_time

        if result.returncode == 0:
            print(f"✓ Completed successfully in {elapsed:.1f} seconds")
            if result.stdout:
                # Print last 20 lines of output to avoid flooding
                lines = result.stdout.strip().split('\n')
                if len(lines) > 20:
                    print(f"\n... (showing last 20 lines of {len(lines)} total)")
                    print('\n'.join(lines[-20:]))
                else:
                    print(result.stdout)
            return True
        else:
            print(f"✗ FAILED with return code {result.returncode}")
            if result.stderr:
                print(f"Error: {result.stderr}")
            return False

    except Exception as e:
        print(f"✗ FAILED with exception: {e}")
        return False

def main():
    print("="*70)
    print("GRAVITY MODEL PROJECT - FULL PIPELINE")
    print("="*70)
    print(f"\nWill run {len(scripts)} scripts in order.")

    input("\nPress Enter to start (or Ctrl+C to cancel)...")

    start_total = time.time()
    results = []

    for script_name, description in scripts:
        success = run_script(script_name, description)
        results.append((script_name, success))

        if not success:
            print(f"\n⚠ Pipeline stopped due to failure in {script_name}")
            response = input("Continue anyway? (y/n): ")
            if response.lower() != 'y':
                break

    # Summary
    elapsed_total = time.time() - start_total
    print("\n" + "="*70)
    print("PIPELINE SUMMARY")
    print("="*70)

    successful = sum(1 for _, s in results if s)
    failed = sum(1 for _, s in results if not s)

    print(f"\nTotal time: {elapsed_total/60:.1f} minutes")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")

    if failed > 0:
        print("\nFailed scripts:")
        for name, success in results:
            if not success:
                print(f"  - {name}")

if __name__ == "__main__":
    main()