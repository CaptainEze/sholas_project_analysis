# eda.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

DATA_PATH = "../data/preprocessed_data.xlsx"
OUT_DIR = "../results/plots/eda"
TABLE_DIR = "../results/tables"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)

def load_data(path=DATA_PATH):
    df = pd.read_excel(path)
    df.columns = df.columns.str.strip()
    print(f"Loaded {len(df)} rows.")
    return df

def pareto(df, top_n=10):
    counts = df['FunctLocDescrip.'].value_counts()
    top = counts.head(top_n)
    cumperc = (top.cumsum() / counts.sum() * 100).round(2)
    pareto_df = pd.DataFrame({"Equipment": top.index, "Count": top.values, "Cumulative%": cumperc.values})
    pareto_df.to_csv(os.path.join(TABLE_DIR, "pareto_top10.csv"), index=False)
    plt.figure(figsize=(10,6))
    ax = sns.barplot(x=top.index, y=top.values, palette="Blues_d")
    ax2 = ax.twinx()
    ax2.plot(range(len(top)), cumperc.values, color='red', marker='o')
    ax2.axhline(80, color='gray', linestyle='--')
    ax.set_xticklabels(top.index, rotation=45, ha='right')
    ax.set_ylabel("Failure Count")
    ax2.set_ylabel("Cumulative %")
    plt.title("Pareto - Top equipment by failures")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "pareto_failures.png"))
    plt.close()
    print("Saved pareto_failures.png")

def correlation_heatmap(df):
    numeric = df.select_dtypes(include=[np.number])
    corr = numeric.corr()
    corr.to_csv(os.path.join(TABLE_DIR, "correlation_matrix.csv"))
    plt.figure(figsize=(12,10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "correlation_heatmap.png"))
    plt.close()
    print("Saved correlation_heatmap.png")

def monthly_trend(df):
    if 'Basic start date' not in df.columns:
        print("No Basic start date column; skipping monthly trend.")
        return
    df = df.copy()
    df['Basic start date'] = pd.to_datetime(df['Basic start date'], errors='coerce')
    monthly = df.groupby(df['Basic start date'].dt.to_period('M'))['FunctLocDescrip.'].count()
    monthly.index = monthly.index.to_timestamp()
    monthly.to_csv(os.path.join(TABLE_DIR, "monthly_failure_counts.csv"))
    plt.figure(figsize=(12,6))
    monthly.plot(marker='o')
    plt.title("Monthly Failure Trend")
    plt.ylabel("Count")
    plt.xlabel("Month")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "monthly_trend.png"))
    plt.close()
    print("Saved monthly_trend.png")

def main():
    df = load_data()
    pareto(df)
    correlation_heatmap(df)
    monthly_trend(df)
    print("EDA complete. Tables saved in results/tables and plots in results/plots/eda.")

if __name__ == "__main__":
    main()
