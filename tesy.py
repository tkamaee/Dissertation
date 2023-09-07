import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pywt
import re
import pandas as pd


def set_parameters():
    show_sign = False
    save = False
    first_frame = 1800
    last_frame = 5000

    widths = np.arange(1, 100)
    scale_point = 60
    peak_threshold = 0.25
    dip_threshold = 0.0005


    exclude_distance = 50


    return (show_sign, save, first_frame, last_frame, widths, peak_threshold, dip_threshold, scale_point, exclude_distance)


def old_timestamps(timestamps_file):

    timestamp_df = pd.read_csv(timestamps_file)
    original_frames = [int(time * 30) for time in timestamp_df['relative_time'].tolist() if time != 0]
    return original_frames


def find_peaks(widths, peak_threshold, dip_threshold, scale_point, first_frame, last_frame, file_name, save, matrix, exclude_distance, original_frames):

    mean_intensity = matrix[first_frame:last_frame]  # np.mean(matrix, axis=(1, 2))

    wavelet ='mexh'

    if mean_intensity.size == 0:
        raise ValueError("mean_intensity array is empty.")
    else:
        cwt_matrix, _ = pywt.cwt(mean_intensity, widths, wavelet)

    significant_peaks = np.where(cwt_matrix[scale_point] > peak_threshold)[0]
    # print(significant_peaks)

    significant_dips = np.where(cwt_matrix[scale_point] < -dip_threshold)[0]
    #print(significant_dips)

    zero_crossings = np.where(np.diff(np.sign(cwt_matrix[scale_point])))[0]

    end_frame = len(mean_intensity[:last_frame]) - 1
    end_threshold = (10 / 100) * end_frame
    start_frame = len(mean_intensity) - end_frame
    start_threshold = (5 / 100) * start_frame

    significant_start_frames = []

    for peak_frame in significant_peaks:
        left_dips = significant_dips[significant_dips < peak_frame]
        if len(left_dips) > 0:
            closest_dip = np.max(left_dips)
            if peak_frame - closest_dip >= exclude_distance:
                significant_start_frames.append(closest_dip)

    #print("Significant Start Frames:", significant_start_frames)



    significant_start_frames = list(set(significant_start_frames))
    significant_start_frames.sort()

    shifted_start_frames = [frame + 2 for frame in significant_start_frames]


    print("Shifted Significant Start Frames:", shifted_start_frames)


    real_start_frames = [x + first_frame for x in shifted_start_frames]

    print("Real Start Frames:", real_start_frames)
    print("Original frames:", original_frames)

    original_frames = np.array(original_frames)
    real_start_frames = np.array(real_start_frames)
    min_length = min(len(original_frames), len(real_start_frames))
    original_frames = original_frames[:min_length]
    real_start_frames = real_start_frames[:min_length]
    frames_difference = original_frames - real_start_frames
    print("Frames difference:", frames_difference)



    if file_name is not None:
        if save is True:
            parameters = {
                "original_frames": original_frames,
                "real_start_frames": real_start_frames,
                "frames_difference:": frames_difference,
                "first_frame": first_frame,
                "last_frame": last_frame,
                "widths": widths,
                "peak_threshold": peak_threshold,
                "dip_threshold": dip_threshold,
                "scale_point": scale_point,
                "exclude_distance": exclude_distance,
            }

            parameters_file = os.path.join(output_folder, f'{file_name}_parameters.npy')
            np.save(parameters_file,parameters )
            print(f'Saved generated timestamps to {parameters_file}')
            timestamps_file = os.path.join(output_folder, f'{file_name}_generated_timestamps.npy')
            np.save(timestamps_file, real_start_frames)
            print(f'Saved generated timestamps to {timestamps_file}')

    else:
        print('Saving Timestamps Failed.')
    return (real_start_frames, cwt_matrix, mean_intensity, significant_peaks, significant_start_frames)

def plot_main(mean_intensity, widths, cwt_matrix, file_name, save, significant_peaks, significant_start_frames):
    fig, ax = plt.subplots(figsize=(10.8, 7.2))


    im = ax.imshow(np.abs(cwt_matrix), extent=[0, len(mean_intensity), min(widths), max(widths)], aspect='auto',
                   cmap='cividis')
    if save is True:
        ax.set_xlabel('Time (seconds)')
    else:
        ax.set_xlabel('Frames')
    ax.set_ylabel('Width of Wavelet')
    ax.set_title(f'dF/F Change projection Overlayed on CWT Scalogram for {file_name} ')

    cbar = fig.colorbar(im, ax=ax, pad=0.1)
    cbar.set_label('Magnitude')
    cax = cbar.ax

    if dip_threshold is not peak_threshold:
        cax = cbar.ax
        cax.hlines(peak_threshold, 0, 1, colors='black', linewidth=2, linestyles='dashed', label='Peak Threshold')
    cax.hlines(dip_threshold, 0, 1, colors='white', linewidth=2, linestyles='dashed', label='Magnitude Threshold')
    cax.legend(loc='upper left', fontsize='x-small')



    num_frames = len(mean_intensity)
    frame_indices = np.arange(0, num_frames, step=600)
    if save is True:
        time_values = (frame_indices + first_frame) / 30
    else:
        time_values = frame_indices + first_frame
    time_labels = [f'{time:.1f}' for time in time_values]

    ax2 = ax.twinx()
    ax2.set_xticks(frame_indices, time_labels)

    if save is True:
        ax2.set_xlabel('Time (seconds)')
    else:
        ax2.set_xlabel('Frames')
    ax2.plot(mean_intensity, color='black', label='Mean Intensity', alpha=0.8, linewidth=1, zorder=1)
    ax2.set_ylabel('\u0394F/F')
    ax2.legend(loc='upper right')

    plt.tight_layout()

    ax.scatter(significant_peaks, np.ones(len(significant_peaks)) * scale_point, color='white', marker='o',
               label='Significant Peaks', zorder=10)

    for frame in significant_start_frames:
        ax.axvline(x=frame, color='white', linestyle='--', linewidth=1, zorder=15)

    scalogram = os.path.join(output_folder, f'df_f_scalogram_session{file_name}.svg')
    if save is True:
        fig.savefig(scalogram, dpi=600)
    plt.show()
    return None

def plot_sign(cwt_matrix, save):

    cwt_sign = np.sign(cwt_matrix[scale_point])
    cmap = matplotlib.colors.ListedColormap(['black', 'white'])
    norm = matplotlib.colors.Normalize(vmin=-1, vmax=1)


    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cwt_sign.reshape(1, -1), cmap=cmap, norm=norm, aspect='auto')
    if save is True:
        ax.set_xlabel('Time (seconds)')
    else:
        ax.set_xlabel('frames')
    ax.set_ylabel('Sign of CWT Response')
    ax.set_title('Sign of CWT Response at Scale {}'.format(scale_point))
    ax.set_yticks([])

    num_frames = len(cwt_sign)
    frame_indices = np.arange(0, num_frames, step=600)
    if save is True:
        time_values = (frame_indices + first_frame) / 30
    else:
        time_values = frame_indices + first_frame
    time_labels = [f'{time:.1f}' for time in time_values]


    ax.set_xticks(frame_indices)
    ax.set_xticklabels(time_labels)
    plt.xticks(rotation=45)

    cbar = fig.colorbar(im, ax=ax, orientation='horizontal', ticks=[-1, 1], pad=0.2)
    cbar.set_label('Sign')
    plt.tight_layout()

    sign_CWT = os.path.join(output_folder, f'CWT_sign_session{file_name}.svg')
    if save is True:
        fig.savefig(sign_CWT, dpi=600)
    # fig.savefig(f'CWT_sign_session{session_number}.svg', dpi=600)

    plt.show()
    return None

if __name__ == "__main__":

    input_file = "Z:/capsid_testing/imaging/cscope/final_pipeline/gcamp1/final_df_f/gcamp1_session0_fluorescence_matrix.npy"
    timestamps_folder = "Z:/capsid_testing/imaging/cscope/gcamp1/processed_timestamps/timestamp0.csv"
    output_folder = 'Z:/capsid_testing/imaging/cscope/final_pipeline/gcamp1/contra_generated_timestamps'
    os.makedirs(output_folder, exist_ok=True)

    file_name_match = re.search(r"((Cap\d+)|(gcamp1))[^/]*(session\d+)", input_file)
    if file_name_match:
        file_name = file_name_match.group()
    else:
        file_name = None


    matrix = np.load(input_file)

    original_frames = old_timestamps(timestamps_folder)
    show_sign, save, first_frame, last_frame, widths, peak_threshold, dip_threshold, scale_point, exclude_distance = set_parameters()
    real_start_frames, cwt_matrix, mean_intensity, significant_peaks, significant_start_frames = find_peaks(widths, peak_threshold, dip_threshold, scale_point, first_frame, last_frame, file_name, save, matrix, exclude_distance, original_frames)
    plot_main(mean_intensity, widths, cwt_matrix, file_name, save, significant_peaks, significant_start_frames)
    if show_sign is True:
        plot_sign(cwt_matrix, save, scale_point)

















