import zipfile
import os
import pandas as pd
import numpy as np
from scipy.signal import resample, savgol_filter, butter, filtfilt
from datetime import datetime
import glob
import math
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Placeholder for external functions (ensure these are available in your environment)
from band_pass_filtering import band_pass_filtering
from compute_vitals import vitals
from detect_apnea_events import apnea_events
from modwt_matlab_fft import modwt
from modwt_mra_matlab_fft import modwtmra
from remove_nonLinear_trend import remove_nonLinear_trend
from data_subplot import data_subplot
from beat_to_beat import compute_rate
from detect_body_movement_1 import detect_patterns1  # Add this import

def extract_zip(zip_path, extract_to=None):
    """
    Extracts all files from a zip archive and returns a list of extracted file paths.
    """
    if extract_to is None:
        extract_to = os.path.splitext(zip_path)[0]

    os.makedirs(extract_to, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
        extracted_files = [os.path.join(extract_to, f) for f in zip_ref.namelist()]

    return extracted_files

def process_bcg_file(input_path, output_path, resampled_output_path=None):
    """
    Processes the BCG CSV file by extracting amplitude values, creating a time vector,
    and optionally resampling to 50 Hz with a DateTime column.
    """
    with open(input_path, 'r') as f:
        lines = f.readlines()

    # Extract header and fs
    header = lines[0].strip().split(',')
    first_line = lines[1].strip().split(',')
    fs = int(first_line[2]) if len(first_line) > 2 else 140
    start_timestamp = int(first_line[1]) / 1000

    # Extract BCG values
    bcg_values = []
    for line in lines[1:]:
        parts = line.strip().split(',')
        if parts[0].lstrip('-').isdigit():
            bcg_values.append(int(parts[0]))

    # Generate time vector
    time_vector = [start_timestamp + i / fs for i in range(len(bcg_values))]

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

        # Convert time_resampled to datetime
        base_time = datetime(1970, 1, 1)
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
    """
    try:
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

        # Check if there's an overlap
        if sync_start >= sync_end:
            print(f"No synchronization possible for {os.path.basename(bcg_csv_in)} and {os.path.basename(ecg_csv_in)}: no overlapping time window.")
            return False

        print(f"Synchronizing both signals to window:\n  start = {sync_start}\n    end = {sync_end}")

        # Trim each DataFrame to that window
        df_ecg_sync = df_ecg[(df_ecg['dt'] >= sync_start) & (df_ecg['dt'] <= sync_end)].copy()
        df_bcg_sync = df_bcg[(df_bcg['dt'] >= sync_start) & (df_bcg['dt'] <= sync_end)].copy()

        # Drop the helper 'dt' columns
        df_ecg_sync = df_ecg_sync.drop(columns=['dt'])
        df_bcg_sync = df_bcg_sync.drop(columns=['dt'])

        # Save out the synchronized CSVs
        df_ecg_sync.to_csv(ecg_csv_out, index=False)
        df_bcg_sync.to_csv(bcg_csv_out, index=False)

        print("Saved synchronized ECG to:", ecg_csv_out)
        print("Saved synchronized BCG to:", bcg_csv_out)
        return True

    except Exception as e:
        print(f"Error during synchronization of {os.path.basename(bcg_csv_in)} and {os.path.basename(ecg_csv_in)}: {str(e)}")
        return False

def main_analysis(bcg_file, ecg_file_path, results_dir, pair_id):
    """
    Performs heart rate and respiratory rate analysis on synchronized BCG and ECG files.
    """
    print('\nstart processing ...')

    # Define sampling frequency
    target_fs = 50  # Hz

    # Load BCG data
    bcg_data = pd.read_csv(bcg_file, sep=",", header=None, skiprows=1).values
    bcg_data_stream = bcg_data[:, 2].astype(float)  # BCG Amplitude
    bcg_timestamps = bcg_data[:, 1].astype(np.int64) * 1000  # Time (s) to ms

    print("Sample BCG timestamps (Unix ms):", bcg_timestamps[:5])
    print("Sample BCG signal data:", bcg_data_stream[:5])

    # Load ECG data
    ecg_data = pd.read_csv(ecg_file_path, sep=",", header=None, skiprows=1).values
    ecg_timestamps_str = ecg_data[:, 0]  # Timestamp
    ecg_hr = ecg_data[:, 1].astype(float)  # Heart Rate
    ecg_rr = ecg_data[:, 2].astype(float)  # RR

    try:
        ecg_timestamps = pd.to_datetime(ecg_timestamps_str, format='%Y/%m/%d %H:%M:%S').astype('int64') // 10**6
    except ValueError as e:
        print("Error parsing ECG timestamps:", e)
        raise

    print("Sample ECG timestamps (converted to Unix ms):", ecg_timestamps[:5])
    print(f"Original ECG data points: {len(ecg_hr)}")
    print(f"Original BCG data points: {len(bcg_data_stream)}")

    # Improved movement detection
    start_point, end_point, window_shift = 0, 500, 500
    filtered_bcg, filtered_bcg_time = detect_patterns1(
        start_point, end_point, window_shift, 
        bcg_data_stream, bcg_timestamps, plot=1
    )

    # Time alignment and filtering
    filtered_bcg_seconds = np.unique(filtered_bcg_time // 1000)
    ecg_seconds = ecg_timestamps // 1000
    ecg_mask = np.isin(ecg_seconds, filtered_bcg_seconds)

    filtered_ecg_hr = ecg_hr[ecg_mask]
    filtered_ecg_timestamps = ecg_timestamps[ecg_mask]
    filtered_ecg_rr = ecg_rr[ecg_mask]

    print(f"\nFiltered ECG data points: {len(filtered_ecg_hr)}")
    print(f"Filtered BCG data points: {len(filtered_bcg)}")
    print(f"Removed {len(ecg_hr) - len(filtered_ecg_hr)} ECG points ({(1 - len(filtered_ecg_hr)/len(ecg_hr))*100:.1f}%)")

    # Signal processing
    movement = band_pass_filtering(filtered_bcg, target_fs, "bcg")
    breathing = band_pass_filtering(filtered_bcg, target_fs, "breath")
    breathing = remove_nonLinear_trend(breathing, 3)
    breathing = savgol_filter(breathing, 15, 3)

    # Wavelet analysis
    w = modwt(movement, 'bior3.9', 4)
    dc = modwtmra(w, 'bior3.9')
    wavelet_cycle = dc[4]

    # Heart rate detection
    hr_mpd = int(0.35 * target_fs)
    print(f"\nUsing optimized MPD: {hr_mpd} samples")
    beats = vitals(0, 500, 500, int(math.floor(breathing.size/500)), 
                  wavelet_cycle, filtered_bcg_time, mpd=hr_mpd, plot=0)
    print(f"\nLength of detected BCG beats: {len(beats)}")
    print(f"Length of ECG HR window: {len(filtered_ecg_hr)}")

    # Time alignment compensation
    shift_samples = int(0.3 * target_fs)
    beats_shifted = beats[shift_samples:]

    # Smoothing and cleaning
    window_size = 10
    beats_smoothed = np.convolve(beats_shifted, np.ones(window_size)/window_size, mode='valid')
    min_length = min(len(beats_smoothed), len(filtered_ecg_hr))
    bcg_hr = beats_smoothed[:min_length]
    ecg_hr_ref = filtered_ecg_hr[:min_length]

    valid_mask = (bcg_hr > 30) & (bcg_hr < 200) & (ecg_hr_ref > 30) & (ecg_hr_ref < 200)
    bcg_hr_clean = bcg_hr[valid_mask]
    ecg_hr_clean = ecg_hr_ref[valid_mask]

    print(f"\nValid data after cleaning: {len(bcg_hr_clean)}/{len(bcg_hr)}")
    print("BCG HR stats:")
    print(f"Min: {np.min(bcg_hr_clean):.1f}, Max: {np.max(bcg_hr_clean):.1f}, Mean: {np.mean(bcg_hr_clean):.1f}")

    # Respiratory analysis
    breaths = vitals(0, 500, 500, int(math.floor(breathing.size/500)), 
                    breathing, filtered_bcg_time, mpd=75, plot=0)
    print("\nRespiratory Rate:")
    print(f"Min: {np.min(breaths):.1f}, Max: {np.max(breaths):.1f}, Mean: {np.mean(breaths):.1f}")

    # Error metrics
    mae = np.mean(np.abs(bcg_hr_clean - ecg_hr_clean))
    rmse = np.sqrt(np.mean((bcg_hr_clean - ecg_hr_clean) ** 2))
    corr, _ = pearsonr(ecg_hr_clean, bcg_hr_clean)

    print("\nValidation Metrics:")
    print(f"MAE: {mae:.2f} BPM | RMSE: {rmse:.2f} BPM")
    print(f"Pearson Correlation: {corr:.2f}")

    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)

    # Enhanced plots
    plt.figure(figsize=(10, 4))
    plt.plot(ecg_hr_clean, label='ECG HR')
    plt.plot(bcg_hr_clean, label='BCG HR')
    plt.xlabel('Time (samples)')
    plt.ylabel('Heart Rate (BPM)')
    plt.title(f'Heart Rate Comparison (Pair {pair_id})')
    plt.legend()
    plt.savefig(os.path.join(results_dir, f'1_hr_comparison_pair_{pair_id}.png'))
    plt.close()

    # Bland-Altman Plot
    mean_hr = (bcg_hr_clean + ecg_hr_clean) / 2
    diff_hr = bcg_hr_clean - ecg_hr_clean
    mean_diff = np.mean(diff_hr)
    std_diff = np.std(diff_hr)

    plt.figure(figsize=(8, 6))
    plt.scatter(mean_hr, diff_hr, c='blue', alpha=0.5)
    plt.axhline(mean_diff, color='red', linestyle='--', label='Mean Bias')
    plt.axhline(mean_diff + 1.96*std_diff, color='grey', linestyle='--')
    plt.axhline(mean_diff - 1.96*std_diff, color='grey', linestyle='--')
    plt.xlabel('Average Heart Rate (BPM)')
    plt.ylabel('Difference (BCG - ECG)')
    plt.title(f'Bland-Altman Plot (Pair {pair_id})\nMean Bias: {mean_diff:.2f} ± {std_diff:.2f} BPM')
    plt.legend()
    plt.savefig(os.path.join(results_dir, f'2_bland_altman_pair_{pair_id}.png'))
    plt.close()

    # Correlation Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(ecg_hr_clean, bcg_hr_clean, alpha=0.5)
    plt.plot([40, 180], [40, 180], 'r--')
    plt.xlabel('ECG Heart Rate (BPM)')
    plt.ylabel('BCG Heart Rate (BPM)')
    plt.title(f'Heart Rate Correlation (Pair {pair_id}, r = {corr:.2f})')
    plt.savefig(os.path.join(results_dir, f'3_correlation_patterns1_pair_{pair_id}.png'))
    plt.close()

    print(f'\nProcessing complete for pair {pair_id}. Results saved to {results_dir}')

# Directory containing multiple zip files
zip_dir = r"C:\Users\omara\Downloads\01"
results_base_dir = r"C:\Users\omara\Downloads\results"

# Get all zip files in the directory
zip_files = glob.glob(os.path.join(zip_dir, "*.zip"))

if not zip_files:
    print("No zip files found in the directory:", zip_dir)
    exit()

# Process each zip file
for zip_file_path in zip_files:
    zip_name = os.path.splitext(os.path.basename(zip_file_path))[0]
    print(f"\nProcessing zip file: {zip_name}")
    
    # Create a results directory for this zip file
    results_dir = os.path.join(results_base_dir, zip_name)
    os.makedirs(results_dir, exist_ok=True)

    # Extract files from the zip
    extracted_files = extract_zip(zip_file_path)

    # Organize files by type
    bcg_files = []
    rr_files = []

    for file_path in extracted_files:
        if 'BCG/' in file_path and file_path.endswith('.csv'):
            bcg_files.append(file_path)
        elif 'Reference/RR/' in file_path and file_path.endswith('.csv'):
            rr_files.append(file_path)

    # Debug: Print detected files
    print("BCG files:", [os.path.basename(f) for f in bcg_files])
    print("RR files:", [os.path.basename(f) for f in rr_files])

    # Extract dates from filenames and match BCG with RR files
    paired_files = []
    for bcg_file in bcg_files:
        # Extract date from BCG filename (e.g., 01_20231105_BCG.csv -> 20231105)
        bcg_filename = os.path.basename(bcg_file)
        try:
            bcg_date = bcg_filename.split('_')[1]  # Assumes date is second part after splitting by '_'
        except IndexError:
            print(f"Could not extract date from BCG file: {bcg_filename}. Skipping.")
            continue

        # Look for matching RR file
        matching_rr = None
        for rr_file in rr_files:
            rr_filename = os.path.basename(rr_file)
            try:
                rr_date = rr_filename.split('_')[1]  # Assumes date is second part after splitting by '_'
            except IndexError:
                print(f"Could not extract date from RR file: {rr_filename}. Skipping.")
                continue
            if rr_date == bcg_date:
                matching_rr = rr_file
                break

        if matching_rr:
            paired_files.append((bcg_file, matching_rr))
        else:
            print(f"No matching RR file found for BCG file: {bcg_filename}. Skipping.")

    # Process each pair
    for idx, (bcg_file_path, ecg_file_path) in enumerate(paired_files, 1):
        print(f"\nProcessing pair {idx}: {os.path.basename(bcg_file_path)} and {os.path.basename(ecg_file_path)}")
        
        # Define output paths
        output_file_path = bcg_file_path.replace('.csv', '_processed.csv')
        resampled_output_file_path = bcg_file_path.replace('.csv', '_processed_50Hz.csv')
        sync_bcg_output_path = bcg_file_path.replace('.csv', '_processed_50Hz_sync.csv')
        sync_ecg_output_path = ecg_file_path.replace('.csv', '_sync.csv')

        # Process and resample BCG
        process_bcg_file(bcg_file_path, output_file_path, resampled_output_file_path)

        # Synchronize BCG and ECG
        if synchronize_bcg_ecg(resampled_output_file_path, ecg_file_path, sync_bcg_output_path, sync_ecg_output_path):
            # Run main analysis on synchronized files
            main_analysis(sync_bcg_output_path, sync_ecg_output_path, results_dir, f"{zip_name}_{idx}")

    if not paired_files:
        print(f"No matching BCG and RR file pairs found for zip file: {zip_name}. No processing performed.")