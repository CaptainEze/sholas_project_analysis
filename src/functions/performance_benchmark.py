import pandas as pd
import matplotlib.pyplot as plt
import os

def performance_benchmark(file_path, output_dir="../results/benchmarking",
                           mtbf_col="MTBF", mttr_col="MTTR", cost_col="Estimated costs",
                           target_col="Target", benchmark_values=None):
    """
    Performance benchmarking script.
    Calculates MTBF, MTTR, and cost per failure, compares to benchmarks, and saves results.

    Parameters:
    - file_path: path to preprocessed data file (Excel or CSV)
    - output_dir: folder to save results
    - mtbf_col, mttr_col, cost_col: column names for metrics
    - target_col: column indicating failure (1 = fail, 0 = no fail)
    - benchmark_values: dict with optional industry/historical benchmarks
    """

    os.makedirs(output_dir, exist_ok=True)

    # Load data
    if file_path.endswith(".xlsx"):
        df = pd.read_excel(file_path)
    else:
        df = pd.read_csv(file_path)

    # Clean column names
    df.columns = df.columns.str.strip()

    # Calculate KPIs
    total_failures = df[df[target_col] == 1].shape[0]
    avg_mtbf = df[mtbf_col].mean()
    avg_mttr = df[mttr_col].mean()
    avg_cost_per_failure = df[cost_col].sum() / total_failures if total_failures > 0 else 0

    results = pd.DataFrame({
        "Metric": ["Total Failures", "Average MTBF", "Average MTTR", "Avg Cost per Failure"],
        "Value": [total_failures, avg_mtbf, avg_mttr, avg_cost_per_failure]
    })

    # Add benchmark comparison
    if benchmark_values:
        results["Benchmark"] = results["Metric"].map(benchmark_values)
        results["Gap"] = results["Value"] - results["Benchmark"]

    # Save results table
    results.to_csv(os.path.join(output_dir, "performance_benchmark.csv"), index=False)

    # Plot KPIs
    plt.figure(figsize=(6,4))
    plt.bar(results["Metric"], results["Value"], color="skyblue")
    plt.title("Performance KPIs")
    plt.ylabel("Value")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "performance_kpis.png"))
    plt.close()

    print(f"Performance benchmarking completed. Results saved to {output_dir}")

    return results

if __name__ == "__main__":
    # Example usage
    benchmark_values = {
        "Average MTBF": 150,  # example benchmark
        "Average MTTR": 5,
        "Avg Cost per Failure": 2000
    }
    performance_benchmark("../data/preprocessed_data.xlsx", benchmark_values=benchmark_values)
