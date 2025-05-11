import zipfile
import os
import pandas as pd
import numpy as np
from scipy.signal import resample
from datetime import datetime

def extract_zip(zip_path, extract_to=None):
    """
    Extracts all files from a zip archive and returns a list of extracted file paths.

    Parameters:
        zip_path (str): Path to the zip file.
        extract_to (str): Directory to extract files to. If None, extracts to same directory as zip file.

    Returns:
        List[str]: List of paths to the extracted files.
    """
    if extract_to is None:
        extract_to = os.path.splitext(zip_path)[0]  # Folder with same name as zip

    os.makedirs(extract_to, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
        extracted_files = [os.path.join(extract_to, f) for f in zip_ref.namelist()]

    return extracted_files

def process_bcg_file(input_path, output_path, resampled_output_path=None):
    """
    Processes the BCG CSV file by extracting amplitude values, creating a time vector,
    and optionally resampling to 50 Hz with a DateTime column.

    Parameters:
        input_path (str): Path to the raw BCG CSV file.
        output_path (str): Path to save the processed CSV.
        resampled_output_path (str): Path to save the resampled CSV (optional).

    """
    with open(input_path, 'r') as f:
        lines = f.readlines()

    # Extract header and fs
    header = lines[0].strip().split(',')
    first_line = lines[1].strip().split(',')
    fs = int(first_line[2]) if len(first_line) > 2 else 140  # Default fs = 140 if not provided
    start_timestamp = int(first_line[1]) / 1000  # Convert from ms to seconds

    # Extract BCG values
    bcg_values = []
    for line in lines[1:]:
        parts = line.strip().split(',')
        if parts[0].lstrip('-').isdigit():
            bcg_values.append(int(parts[0]))

    # Generate time vector based on fs (rounded to the nearest second)
    time_vector = [start_timestamp + i / fs for i in range(len(bcg_values))]  # Keep as float for precision

    # Create DataFrame
    df = pd.DataFrame({
        'Time (s)': time_vector,
        'BCG Amplitude': bcg_values
    })

    # Save to processed CSV
    df.to_csv(output_path, index=False)
    print(f"Processed CSV saved to: {output_path}")

    # Resample to 50 Hz if resampled_output_path is provided
    if resampled_output_path:
        original_fs = fs
        target_fs = 50

        # Extract data
        time = df["Time (s)"].values
        amplitude = df["BCG Amplitude"].values

        # Duration and number of target samples
        duration = time[-1] - time[0]
        n_target = int(duration * target_fs)

        # Resample amplitude
        amplitude_resampled = resample(amplitude, n_target)

        # Create new time vector
        time_resampled = np.linspace(time[0], time[-1], n_target)

        # Convert time_resampled (seconds since epoch) to datetime
        # Assuming time_resampled is seconds since epoch (adjust base_time if needed)
        base_time = datetime(1970, 1, 1)  # Unix epoch start
        datetime_resampled = [base_time + pd.Timedelta(seconds=t) for t in time_resampled]
        datetime_str = [dt.strftime('%Y/%m/%d %H:%M:%S') for dt in datetime_resampled]

        # Create new DataFrame with DateTime column
        df_resampled = pd.DataFrame({
            "DateTime": datetime_str,
            "Time (s)": time_resampled,
            "BCG Amplitude": amplitude_resampled
        })
        df_resampled.to_csv(resampled_output_path, index=False)
        print(f"Resampled data saved to: {resampled_output_path}")

def synchronize_bcg_ecg(bcg_csv_in, ecg_csv_in, bcg_csv_out, ecg_csv_out):
    """
    Synchronizes BCG and ECG CSV files based on their overlapping time window.

    Parameters:
        bcg_csv_in (str): Path to the resampled BCG CSV file.
        ecg_csv_in (str): Path to the ECG (RR) CSV file.
        bcg_csv_out (str): Path to save the synchronized BCG CSV.
        ecg_csv_out (str): Path to save the synchronized ECG CSV.
    """
    # Load both CSVs
    df_ecg = pd.read_csv(ecg_csv_in)
    df_bcg = pd.read_csv(bcg_csv_in)

    # Parse their time columns to pandas datetime
    df_ecg['dt'] = pd.to_datetime(df_ecg['Timestamp'], format='%Y/%m/%d %H:%M:%S')
    df_bcg['dt'] = pd.to_datetime(df_bcg['DateTime'], format='%Y/%m/%d %H:%M:%S')

    # Compute overlapping window
    start_ecg = df_ecg['dt'].min()
    end_ecg = df_ecg['dt'].max()
    start_bcg = df_bcg['dt'].min()
    end_bcg = df_bcg['dt'].max()

    # The sync window is from the later of the two starts to the earlier of the two ends
    sync_start = max(start_ecg, start_bcg)
    sync_end = min(end_ecg, end_bcg)

    print(f"Synchronizing both signals to window:\n  start = {sync_start}\n    end = {sync_end}")

    # Trim each DataFrame to that window
    df_ecg_sync = df_ecg[(df_ecg['dt'] >= sync_start) & (df_ecg['dt'] <= sync_end)].copy()
    df_bcg_sync = df_bcg[(df_bcg['dt'] >= sync_start) & (df_bcg['dt'] <= sync_end)].copy()  # Corrected to use df_bcg['dt']

    # Drop the helper 'dt' columns
    df_ecg_sync = df_ecg_sync.drop(columns=['dt'])
    df_bcg_sync = df_bcg_sync.drop(columns=['dt'])

    # Save out the synchronized CSVs
    df_ecg_sync.to_csv(ecg_csv_out, index=False)
    df_bcg_sync.to_csv(bcg_csv_out, index=False)

    print("Saved synchronized ECG to:", ecg_csv_out)
    print("Saved synchronized BCG to:", bcg_csv_out)

# Example usage
zip_file_path = r'C:\Users\maram\Desktop\New folder (5)\code\sample.zip'
extracted_files = extract_zip(zip_file_path)

# Find the BCG and ECG CSV file paths
bcg_file_path = None
ecg_file_path = None
for file_path in extracted_files:
    if file_path.endswith('BCG.csv'):  # Adjust if BCG file has a different name
        bcg_file_path = file_path
    elif file_path.endswith('01_20231105_RR.csv'):  # ECG file name
        ecg_file_path = file_path

if bcg_file_path and ecg_file_path:
    output_file_path = bcg_file_path.replace('.csv', '_processed.csv')
    resampled_output_file_path = bcg_file_path.replace('.csv', '_processed_50Hz.csv')
    sync_bcg_output_path = bcg_file_path.replace('.csv', '_processed_50Hz_sync.csv')
    sync_ecg_output_path = ecg_file_path.replace('.csv', '_sync.csv')

    # Process and resample BCG
    process_bcg_file(bcg_file_path, output_file_path, resampled_output_file_path)

    # Synchronize BCG and ECG
    synchronize_bcg_ecg(resampled_output_file_path, ecg_file_path, sync_bcg_output_path, sync_ecg_output_path)
else:
    if not bcg_file_path:
        print("BCG file not found in the extracted files.")
    if not ecg_file_path:
        print("ECG file (01_20231105_RR.csv) not found in the extracted files.")