import os
import numpy as np
import matplotlib.pyplot as plt
import re
from matplotlib.ticker import PercentFormatter
import pandas as pd



def load_matrix(matrix_filepath):
    matrix = np.load(matrix_filepath)
    print("Loaded matrix shape:", matrix.shape)
    if matrix.size == 0:
        print(f"Empty matrix loaded from {matrix_filepath}. Skipping...")
        return None
    else:

        mean_intensity = np.mean(matrix, axis=(1, 2))
        print("mean intensity:", mean_intensity)
        print("mean intensity shape:", mean_intensity.shape)
    return mean_intensity

def calculate_baseline(mean_intensity):
    end_frame = len(mean_intensity)
    relevant_frames = 1800
    start_frame = 0

    accumulated_frames_from_1 = np.sum(mean_intensity[start_frame:relevant_frames-1], axis=0)
    print("accumulated_frames_from_1:", accumulated_frames_from_1.shape)


    baseline_from_1 = accumulated_frames_from_1 / (relevant_frames-start_frame)
    print("Baseline from 1 shape:", baseline_from_1.shape)

    baseline = baseline_from_1
    first_frame_excluded = True

    return baseline, first_frame_excluded, start_frame


def calculate_dF_F(mean_intensity, baseline, first_frame_excluded, start_frame):
    dF_F_stack = np.zeros_like(mean_intensity, dtype=np.float32)
    print(dF_F_stack.shape)
    end_frame = len(dF_F_stack) - 1
    print(end_frame)

    if first_frame_excluded:
        subtracted_stack = mean_intensity[:end_frame] - baseline


    dF_F_stack = subtracted_stack / baseline

    return dF_F_stack



def save_fluorescence_mean_intensity(dF_F_stack, output_folder, output_file_name, start_frame):
    data_to_save = dF_F_stack[start_frame:]
    fluorescence_file = os.path.join(output_folder, f"{output_file_name}_fluorescence_matrix.npy")
    np.save(fluorescence_file, data_to_save)
    print(f"Fluorescence matrix saved to: {fluorescence_file}")
    return None


def plot_z_profile_projection(dF_F_stack, output_folder, output_file_name, frames, start_frame):
    z_profile_projection = dF_F_stack[:]
    print("z_profile_projection_shape:", z_profile_projection.shape)
    print(start_frame)



    plt.figure(figsize=(10.8, 7.2), dpi=600)
    plt.plot(z_profile_projection[start_frame:], color='cyan', label='Average Intensity')
    plt.grid(False)

    plot_frames = True

    if plot_frames is True:
        for frame in frames:
            plt.axvline(x=frame-start_frame, color='black', linestyle='dashed', linewidth=0.75)

    num_frames = len(z_profile_projection)
    start_frame = start_frame
    frame_indices = np.arange(0, num_frames - start_frame, step=450)
    time_values = (frame_indices + start_frame) / 30
    time_labels = [f'{time:.1f}' for time in time_values]

    plt.xticks(np.arange(0, num_frames - start_frame, step=450), time_labels, fontsize=16)
    plt.xlabel('Time (seconds)', fontsize=16)
    plt.ylabel('\u0394F/F (%)', fontsize=16)
    plt.title(f' \u0394F/F Over Time', fontsize=18)

    plt.gca().yaxis.set_major_formatter((PercentFormatter(xmax=1, decimals=1)))
    plt.gca().yaxis.set_tick_params(labelsize=16)

    # plt.grid(which='both', axis='x')

    animal_type = re.search(r"(Cap\d+|gcamp1)", output_file_name).group()
    if animal_type == "Cap40":
        plt.ylim(-0.15, 0.15)
    elif animal_type == "Cap43":
        plt.ylim(-0.2, 0.2)
    elif animal_type == "Cap44":
        plt.ylim(-0.3, 0.2)
    elif animal_type == "Cap48":
        plt.ylim(-0.4, 0.2)
    elif animal_type == "Cap52":
        plt.ylim(-0.5, 0.5)
    elif animal_type == "gcamp1":
        plt.ylim(-0.15, 0.15)



    plt.savefig(os.path.join(output_folder, f'{output_file_name}_z_profile_projection.svg'), dpi=600)
    plt.close()
    print(f"Plots saved to: {os.path.join(output_folder, 'z_profile_projections.svg')}")




def process_folders(input_folder, output_folder, timestamp_folder):
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith("_matrix.npy"):
                matrix_filepath = os.path.join(root, file)
                video_number = get_video_number(matrix_filepath)
                frames = load_timestamps(timestamp_folder, video_number)

                match = re.search(r"(Cap\d+|gcamp1).*(session\d+)", file)#, video_number)
                if match:
                    matched_text = match.group()
                    output_file_name = matched_text

                    mean_intensity = load_matrix(matrix_filepath)
                    baseline, first_frame_excluded, start_frame = calculate_baseline(mean_intensity)
                    dF_F_stack = calculate_dF_F(mean_intensity, baseline, first_frame_excluded, start_frame)

                    save_fluorescence_mean_intensity(dF_F_stack, output_folder, output_file_name, start_frame)
                    plot_z_profile_projection(dF_F_stack, output_folder, output_file_name, frames, start_frame)

    print("Processing completed.")


def get_video_number(matrix_filepath):
    filename = os.path.basename(matrix_filepath)
    parts = filename.split("_")
    for part in parts:
        print(f"Checking part: {part}")
        if part.startswith("session") and part[7:].isdigit():
            video_number = int(part[7:])
            print(f"Found video number: {video_number}")
            return video_number
    print("No valid video number found.")
    return None


def load_timestamps(timestamps_folder, video_number):
    timestamps_file = None
    for root, _, files in os.walk(timestamps_folder):
        for file in files:
            if file.startswith("timestamp") and file[9:-4] == str(video_number):
                timestamps_file = os.path.join(root, file)
                break

    if timestamps_file is None:
        print(f"Warning: No matching timestamp file found for video '{video_number}'. Skipping...")
        return None

    timestamp_df = pd.read_csv(timestamps_file)
    frames = [int(time * 30) for time in timestamp_df['relative_time'].tolist() if time != 0]
    print("frames:", frames)
    return frames

if __name__ == "__main__":

    input_folders = [
        #"//ceph/akrami/capsid_testing/imaging/cscope/final_pipeline/Cap40/final_matrix",
        #"//ceph/akrami/capsid_testing/imaging/cscope/final_pipeline/Cap43/final_matrix",
        #"//ceph/akrami/capsid_testing/imaging/cscope/final_pipeline/Cap44/final_matrix",
        #"//ceph/akrami/capsid_testing/imaging/cscope/final_pipeline/Cap48/final_matrix",
        #"//ceph/akrami/capsid_testing/imaging/cscope/final_pipeline/Cap52/final_matrix",
        #"//ceph/akrami/capsid_testing/imaging/cscope/final_pipeline/gcamp1/final_matrix",
        #"//ceph/akrami/capsid_testing/imaging/cscope/final_pipeline/Cap43/no_bv_crop/roi_a/final_matrix",
        #"//ceph/akrami/capsid_testing/imaging/cscope/final_pipeline/Cap44/no_bv_crop/roi_a/final_matrix"
        "Z:/capsid_testing/imaging/cscope/final_pipeline/gcamp1/final_matrix"
    ]

    output_folders = [
        #"//ceph/akrami/capsid_testing/imaging/cscope/final_pipeline/Cap40/final_df_f",
        #"//ceph/akrami/capsid_testing/imaging/cscope/final_pipeline/Cap43/final_df_f",
        #"//ceph/akrami/capsid_testing/imaging/cscope/final_pipeline/Cap44/final_df_f",
        #"//ceph/akrami/capsid_testing/imaging/cscope/final_pipeline/Cap48/final_df_f",
        #"//ceph/akrami/capsid_testing/imaging/cscope/final_pipeline/Cap52/final_df_f",
        #"//ceph/akrami/capsid_testing/imaging/cscope/final_pipeline/gcamp1/final_df_f",
        #"//ceph/akrami/capsid_testing/imaging/cscope/final_pipeline/Cap43/no_bv_crop/roi_a/final_df_f",
        #"//ceph/akrami/capsid_testing/imaging/cscope/final_pipeline/Cap44/no_bv_crop/roi_a/final_df_f"
        "Z:/capsid_testing/imaging/cscope/final_pipeline/thesis/figure2"
    ]
    timestamp_folder = "Z:/capsid_testing/imaging/cscope/gcamp1/processed_timestamps"

    for output_folder in output_folders:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

    for input_folder, output_folder in zip(input_folders, output_folders):
        process_folders(input_folder, output_folder, timestamp_folder)

