"""
Created on %(27/10/2016)
Function to detect bed patterns
"""
# The segmentation is performed based on the standard deviation of each time window
# In general if the std is less than 15, it means tha the there is no any pressure applied to the mat.
# if the std if above 2 * MAD all time-windows SD it means, we are facing body movements.
# On the other hand, if the std is between 15 and 2 * MAD of all time-windows SD,
# there will be a uniform pressure to the mat. Then, we can analyze the sleep patterns
import math
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, savefig, figure
from matplotlib.patches import Rectangle

def detect_patterns1(pt1, pt2, win_size, data, time, plot=0):
    # Store start and end point
    pt1_, pt2_ = pt1, pt2

    limit = int(math.floor(data.size / win_size))
    flag = np.zeros([data.size, 1])
    event_flags = np.zeros([limit, 1])
    segments_sd = []

    # Compute standard deviation for each window
    for i in range(limit):
        sub_data = data[pt1:pt2]
        segments_sd.append(np.std(sub_data, ddof=1))
        pt1 = pt2
        pt2 += win_size

    # Compute MAD
    mad = np.sum(np.abs(segments_sd - np.mean(segments_sd))) / len(segments_sd)
    thresh1, thresh2 = 15, 2 * mad

    # Reset pt1 and pt2
    pt1, pt2 = pt1_, pt2_

    # Classify each window
    for j in range(limit):
        std_fos = np.around(segments_sd[j])
        if std_fos < thresh1:
            flag[pt1:pt2] = 3  # No movement
            event_flags[j] = 3
        elif std_fos > thresh2:
            flag[pt1:pt2] = 2  # Movement
            event_flags[j] = 2
        else:
            flag[pt1:pt2] = 1  # Sleeping
            event_flags[j] = 1
        pt1 = pt2
        pt2 += win_size

    # Plotting
    if plot == 1:
        data_for_plot = data
        width = np.min(data_for_plot)
        height = np.max(data_for_plot) + abs(width) if width < 0 else np.max(data_for_plot)

        current_axis = plt.gca()
        plt.plot(np.arange(data.size), data_for_plot, '-k', linewidth=1)
        plt.xlabel('Time [Samples]')
        plt.ylabel('Amplitude [mV]')
        plt.gcf().autofmt_xdate()

        for j in range(limit):
            start_idx = j * win_size
            end_idx = start_idx + win_size
            sub_data = data_for_plot[start_idx:end_idx]
            sub_time = np.arange(start_idx, end_idx) / 50.0

            if event_flags[j] == 3:  # No-movement
                plt.plot(sub_time, sub_data, '-k', linewidth=1)
                current_axis.add_patch(Rectangle((start_idx, width), win_size, height, facecolor="#FAF0BE", alpha=.2))
            elif event_flags[j] == 2:  # Movement
                plt.plot(sub_time, sub_data, '-k', linewidth=1)
                current_axis.add_patch(Rectangle((start_idx, width), win_size, height, facecolor="#FF004F", alpha=1.0))
            else:  # Sleeping
                plt.plot(sub_time, sub_data, '-k', linewidth=1)
                current_axis.add_patch(Rectangle((start_idx, width), win_size, height, facecolor="#00FFFF", alpha=.2))

        plt.savefig('./results/rawData.png')

    # Identify indices to remove (movement and no-movement)
    bad_indices = []
    for j in range(limit):
        start_idx = j * win_size
        end_idx = start_idx + win_size
        if event_flags[j] == 2 or event_flags[j] == 3:
            bad_indices.extend(range(start_idx, end_idx))

    bad_indices = np.array(bad_indices)
    valid_indices = bad_indices[(bad_indices >= 0) & (bad_indices < len(time))]  # Ensure safe indexing


    # Mask out the bad indices
    mask = np.ones(len(data), dtype=bool)
    mask[valid_indices] = False
    filtered_data = data[mask]
    filtered_time = time[mask]

    return filtered_data, filtered_time