import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

def load_preprocessed_data(file_path='../data/preprocessed_data.xlsx'):
    try:
        df = pd.read_excel(file_path)
        print(f"Loaded preprocessed data with {len(df)} rows.")
        return df
    except Exception as e:
        print(f"Error loading preprocessed data: {e}")
        return None

def pareto_analysis(df, output_dir):
    """Pareto chart showing cumulative % of failures by equipment type."""
    failure_counts = df['FunctLocDescrip.'].value_counts().reset_index()
    failure_counts.columns = ['FunctLocDescrip.', 'Count']
    failure_counts['CumPerc'] = failure_counts['Count'].cumsum() / failure_counts['Count'].sum() * 100

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x='FunctLocDescrip.', y='Count', data=failure_counts, color='skyblue')
    ax2 = ax.twinx()
    ax2.plot(failure_counts['FunctLocDescrip.'], failure_counts['CumPerc'], color='red', marker='o')
    ax2.axhline(80, color='grey', linestyle='--')
    plt.title('Pareto Analysis - Equipment Failures')
    ax.set_xlabel('Equipment Type')
    ax.set_ylabel('Failure Count')
    ax2.set_ylabel('Cumulative %')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pareto_failures.png'))
    plt.close()
    print("Pareto chart saved.")

def correlation_heatmap(df, output_dir):
    """Correlation heatmap for numeric features."""
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'))
    plt.close()
    print("Correlation heatmap saved.")

def seasonal_trends(df, output_dir):
    """Monthly/seasonal trends in failures."""
    if 'Basic start date' in df.columns:
        df['Basic start date'] = pd.to_datetime(df['Basic start date'], errors='coerce')
        monthly_trend = df.groupby(df['Basic start date'].dt.to_period('M'))['FunctLocDescrip.'].count()
        monthly_trend.index = monthly_trend.index.to_timestamp()

        plt.figure(figsize=(12, 6))
        monthly_trend.plot(kind='line', marker='o', color='teal')
        plt.title('Monthly Failure Trend')
        plt.xlabel('Month')
        plt.ylabel('Failure Count')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'monthly_trend.png'))
        plt.close()
        print("Monthly trend chart saved.")

def perform_eda(df, output_dir='../results/plots/eda'):
    if df is None:
        print("No data to analyze.")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Existing summary
    print("\nSummary Statistics:")
    print(df.describe(include='all'))

    # New analyses
    pareto_analysis(df, output_dir)
    correlation_heatmap(df, output_dir)
    seasonal_trends(df, output_dir)

    print(f"EDA outputs saved in {output_dir}")

if __name__ == "__main__":
    df = load_preprocessed_data()
    if df is not None:
        perform_eda(df)
