# performance_benchmark.py
import pandas as pd
import os
import matplotlib.pyplot as plt

DATA_PATH = "../data/preprocessed_data.xlsx"
OUT_DIR = "../results/benchmarking"
os.makedirs(OUT_DIR, exist_ok=True)

def run(mtbf_col="MTBF", mttr_col="MTTR", cost_col="Estimated costs", target_col="Target"):
    df = pd.read_excel(DATA_PATH)
    df.columns = df.columns.str.strip()
    if target_col not in df.columns:
        if "Failure rate" in df.columns:
            df[target_col] = (df["Failure rate"] > 0).astype(int)
    total_failures = df[df[target_col] == 1].shape[0]
    avg_mtbf = df[mtbf_col].mean()
    avg_mttr = df[mttr_col].mean()
    avg_cost = df[cost_col].sum() / total_failures if total_failures>0 else 0
    summary = pd.DataFrame({
        "Metric":["Total failures","Avg_MTBF","Avg_MTTR","Avg_Cost_per_Failure"],
        "Value":[total_failures, avg_mtbf, avg_mttr, avg_cost]
    })
    summary.to_csv(os.path.join(OUT_DIR,"benchmark_summary.csv"), index=False)
    # quick bar
    plt.figure(figsize=(6,4))
    plt.bar(summary["Metric"], summary["Value"])
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR,"benchmark_plot.png"))
    plt.close()
    print("Benchmarking complete.")

if __name__ == "__main__":
    run()
