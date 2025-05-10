"""
Created on %(25/09/2017)
Function to perform a Chebyshev type I bandpass filter for heart rate and breathing with adjusted high-frequency noise removal.
"""

from scipy.signal import cheby1, filtfilt
import numpy as np


def band_pass_filtering(data, fs, filter_type):
    # Check signal amplitude to avoid over-filtering
    if np.std(data) < 1e-6:
        print(f"Warning: Input signal for {filter_type} has very low amplitude ({np.std(data)}).")
        return data

    if filter_type == "bcg":
        # High-pass filter to remove low-frequency noise
        [b_cheby_high, a_cheby_high] = cheby1(2, 0.5, [2.5 / (fs / 2)], btype='high', analog=False)
        bcg_ = filtfilt(b_cheby_high, a_cheby_high, data)
        # Adjusted low-pass filter to remove high-frequency noise (>15 Hz)
        [b_cheby_high_noise, a_cheby_high_noise] = cheby1(2, 0.5, [15.0 / (fs / 2)], btype='low', analog=False)
        filtered_data = filtfilt(b_cheby_high_noise, a_cheby_high_noise, bcg_)
    elif filter_type == "breath":
        # High-pass filter to remove very low-frequency noise
        [b_cheby_high, a_cheby_high] = cheby1(2, 0.5, [0.01 / (fs / 2)], btype='high', analog=False)
        bcg_ = filtfilt(b_cheby_high, a_cheby_high, data)
        # Adjusted low-pass filter to remove high-frequency noise (>2 Hz)
        [b_cheby_high_noise, a_cheby_high_noise] = cheby1(2, 0.5, [2.0 / (fs / 2)], btype='low', analog=False)
        filtered_data = filtfilt(b_cheby_high_noise, a_cheby_high_noise, bcg_)
    else:
        filtered_data = data

    # Check output signal amplitude
    if np.std(filtered_data) < 1e-6:
        print(f"Warning: Filtered signal for {filter_type} has very low amplitude ({np.std(filtered_data)}).")
    
    return filtered_data