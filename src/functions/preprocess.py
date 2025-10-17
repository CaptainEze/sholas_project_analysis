import pandas as pd
import numpy as np
import os

def load_data(file_path='../data/sheets.xlsx'):
    sheets = ['Combined_Logs', 'Reliability_Metrics']
    data = {}
    for sheet in sheets:
        try:
            df = pd.read_excel(file_path, sheet_name=sheet)
            df.columns = df.columns.str.strip()
            data[sheet] = df
            print(f"Loaded {sheet} with {len(df)} rows.")
        except Exception as e:
            print(f"Error loading sheet {sheet}: {e}")
    return data

def calculate_reliability_metrics(combined):
    combined = combined.sort_values(['Functional Location', 'Actual Finish Date'])
    combined['Actual Finish Date'] = pd.to_datetime(combined['Actual Finish Date'])

    # MTBF (mean time between failures in days)
    combined['Time_Diff'] = combined.groupby('Functional Location')['Actual Finish Date'].diff().dt.days
    mtbf_by_loc = combined.groupby('Functional Location')['Time_Diff'].mean().fillna(0)

    # MTTR (mean time to repair, proxy using Estimated costs normalized)
    mttr_by_loc = combined.groupby('Functional Location')['Estimated costs'].mean().fillna(0) / 100  # Normalize to days

    # Failure rate (failures per year)
    time_span = (combined['Actual Finish Date'].max() - combined['Actual Finish Date'].min()).days / 365
    failure_rate_by_loc = combined['Functional Location'].value_counts() / time_span

    # Merge calculated metrics
    combined = combined.merge(mtbf_by_loc.rename('MTBF'), on='Functional Location', how='left')
    combined = combined.merge(mttr_by_loc.rename('MTTR'), on='Functional Location', how='left')
    combined = combined.merge(failure_rate_by_loc.rename('Failure rate'), on='Functional Location', how='left')

    return combined

def preprocess_data(data, output_file='../data/preprocessed_data.xlsx'):
    combined = data['Combined_Logs'].copy()
    combined.columns = combined.columns.str.strip()

    expected_rows = 6640
    print(f"Rows in Combined_Logs: {len(combined)} (Expected: ~{expected_rows})")
    if len(combined) > expected_rows * 1.1 or len(combined) < expected_rows * 0.9:
        print("Warning: Row count significantly deviates from expected.")

    if 'Order' in combined.columns and 'Functional Location' in combined.columns:
        combined.loc[:, 'Functional Location'] = combined['Functional Location'].astype(str).str.strip().str.lower()
        dup_count = combined.duplicated(subset=['Order', 'Functional Location']).sum()
        print(f"Duplicates in Combined_Logs (Order, Functional Location): {dup_count} (Expected: 0)")
        if dup_count > 0:
            combined = combined.drop_duplicates(subset=['Order', 'Functional Location'], keep='first')
            print(f"Removed {dup_count} duplicates. Rows remaining: {len(combined)}")

    date_cols = ['Basic start date', 'Basic finish date', 'Actual Finish Date']
    for col in date_cols:
        if col in combined.columns:
            combined.loc[:, col] = pd.to_datetime(combined[col], errors='coerce')
            invalid_dates = combined[col].isna().sum()
            print(f"Invalid dates in {col}: {invalid_dates}")
            if col == 'Actual Finish Date' and invalid_dates > 0:
                original_len = len(combined)
                combined = combined.dropna(subset=[col])
                print(f"Dropped {original_len - len(combined)} rows with invalid {col}.")

    for col in ['Estimated costs', 'Actual Finish Time']:
        if col in combined.columns:
            missing = combined[col].isna().sum()
            print(f"Missing values in {col}: {missing} (Expected: 0)")
            if missing > 0:
                if combined[col].dtype in [np.float64, np.int64]:
                    combined.loc[:, col] = combined[col].fillna(combined[col].median())
                else:
                    combined.loc[:, col] = combined[col].fillna(combined[col].mode().iloc[0])

    # Calculate reliability metrics
    combined = calculate_reliability_metrics(combined)

    reliability_metrics = data['Reliability_Metrics'].copy()
    reliability_metrics.columns = reliability_metrics.columns.str.strip()
    reliability_metrics = reliability_metrics.drop_duplicates(subset=['Functional Location'], keep='first')
    reliability_metrics.loc[:, 'Functional Location'] = reliability_metrics['Functional Location'].astype(str).str.strip().str.lower()

    print(f"Duplicates in Reliability_Metrics Functional Location: {len(data['Reliability_Metrics']) - len(reliability_metrics)}")
    print(f"Rows in Reliability_Metrics: {len(reliability_metrics)}")
    print(f"Unique Functional Locations in Reliability_Metrics: {len(reliability_metrics['Functional Location'].unique())}")

    merge_cols = ['Failure rate', 'MTBF', 'MTTR']
    merged_data = combined.merge(reliability_metrics[['Functional Location'] + merge_cols], on='Functional Location', how='left', suffixes=('_calc', '_rm'))
    for col in merge_cols:
        merged_data[col] = merged_data[f'{col}_rm'].fillna(merged_data[f'{col}_calc'])
        merged_data.drop(columns=[f'{col}_rm', f'{col}_calc'], inplace=True)

    print(f"Rows after merging with Reliability_Metrics: {len(merged_data)} (Expected: ~{len(combined)})")
    unmatched_rows = merged_data[merge_cols[0]].isna().sum()
    if unmatched_rows > 0:
        print(f"Warning: {unmatched_rows} rows missing Reliability_Metrics after merge.")
        unmatched_flocs = merged_data[merged_data[merge_cols[0]].isna()]['Functional Location'].unique()
        print("Sample unmatched Functional Locations:", unmatched_flocs[:5])

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    merged_data.to_excel(output_file, index=False)
    print(f"Preprocessing complete. Cleaned data saved as '{output_file}'.")
    data['Combined_Logs'] = merged_data
    return data

if __name__ == "__main__":
    data = load_data()
    preprocess_data(data)
