import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

def crop_matrix_around_stimulus(mean_intensity, stimulus_frame, num_frames):
    window_start = max(0, stimulus_frame - num_frames)
    window_end = min(len(mean_intensity), stimulus_frame + num_frames + 1)
    response = mean_intensity[window_start:window_end]
    return response

def align_timestamps(mean_intensity, stimulus_timestamps, num_frames):
    aligned_responses = []
    individual_responses = []
    for timestamp in stimulus_timestamps:
        stimulus_frame = timestamp
        response = crop_matrix_around_stimulus(mean_intensity, stimulus_frame, num_frames)
        aligned_responses.append(response)
        individual_responses.append(response)

    if not aligned_responses:
        return None, None

    aligned_responses = np.mean(aligned_responses, axis=0)

    return aligned_responses, individual_responses

def process_session(matrix_file, timestamps_file, output_folder, session_number, num_frames):

    matrix = np.load(matrix_file)
    mean_intensity = matrix

    stimulus_timestamps= np.load(timestamps_file)

    aligned_responses, individual_responses = align_timestamps(mean_intensity, stimulus_timestamps, num_frames)

    if aligned_responses is None:
        print(f"Warning: No valid timestamps found in {timestamps_file}. Skipping...")
        return

    aligned_responses_array = np.array(aligned_responses, dtype=np.float32)

    print("aligned_responses_array shape:", aligned_responses_array.shape)
    print("aligned_responses_array data:", aligned_responses_array[:10])


    # filename = os.path.splitext(os.path.basename(matrix_file))[0]
    #  session_number = int(filename.split('session')[1].split('_')[0])


    output_file = os.path.join(output_folder, f"session{session_number}_averaged_response.npy")
    np.save(output_file, aligned_responses_array)

    frame_duration = 1 / 30
    num_frames = 200
    time_values = np.arange(-num_frames, num_frames + 1) * frame_duration


    plt.figure(figsize=(10.8, 7.2), dpi=600)

    for ind_response in individual_responses:
     plt.plot(time_values, ind_response, alpha=0.5, color='gray')

    plt.plot(time_values, aligned_responses, label='Mean Response', color="dodgerblue")
    plt.axvline(0, color='black', linestyle='dashed', label='Stimulus Frame')
    plt.xlabel('Time from Stimulus(seconds)', fontsize=16)
    plt.xticks(fontsize=16)
    plt.ylabel('\u0394F/F (%)', fontsize=16)
    plt.ylim(-0.10, 0.10)
    plt.title(f'Mean \u0394F/F Aligned About Stimuli Application ', fontsize=18)
    plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=1))
    plt.gca().yaxis.set_tick_params(labelsize=16)
    plt.legend(fontsize=16)
    output_plot_file = os.path.join(output_folder, f"session{session_number}_average_response_plot.svg")
    plt.savefig(output_plot_file)
    plt.close()

if __name__ == "__main__":
    matrix_folder = "Z:/capsid_testing/imaging/cscope/final_pipeline/Cap44/final_df_f"
    timestamps_folder = "Z:/capsid_testing/imaging/cscope/final_pipeline/Cap44/generated_timestamps"
    output_folder = "Z:/capsid_testing/imaging/cscope/final_pipeline/Cap44/ff"

    num_frames = 200

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for root, _, files in os.walk(matrix_folder):
        for file in files:
            if file.endswith("_fluorescence_matrix.npy"):
                matrix_file = os.path.join(root, file)
                session_match = re.search(r"Cap\d+_session(\d+)_", file)
                if session_match:
                    session_number = int(session_match.group(1))
                    print(f"Matrix File: {matrix_file}, Session Number: {session_number}")

                    timestamps_file = os.path.join(timestamps_folder,
                                                   f"Cap44_session{session_number}_generated_timestamps.npy")
                    if not os.path.exists(timestamps_file):
                        print(f"Warning: Timestamps file for session {session_number} not found. Skipping...")
                        continue

                    process_session(matrix_file, timestamps_file, output_folder, session_number, num_frames)
