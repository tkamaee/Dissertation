

import os
import numpy as np
import cv2
import re

def extract_cap_number(filepath):
    match = re.search(r"gcamp\d+", filepath)
    if match:
        return match.group()
    return None

def avi_to_matrix(video_path, grayscale=False):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yuv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        y_channel = yuv_frame[:, :, 0]
        if grayscale:
            frames.append(y_channel)
        else:
            frames.append(frame)

    cap.release()
    print('Matrix created')
    return np.array(frames)

def process_avi_folder(input_folder, output_folder, cap_number):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith("_crop.avi"):
                video_path = os.path.join(root, file)
                print("file found")
                matrix = avi_to_matrix(video_path, grayscale=True) 
                print(matrix.shape)

                video_name = os.path.splitext(file)[0]
                matrix_filename = f"{cap_number}_{video_name}_matrix.npy"
                matrix_filepath = os.path.join(output_folder, matrix_filename)

                np.save(matrix_filepath, matrix)
                print(f"Saved matrix of {video_name} to: {matrix_filepath}")


if __name__ == "__main__":
    input_folders = [
        #"//ceph/akrami/capsid_testing/imaging/cscope/Cap40/processed_video",
        #"//ceph/akrami/capsid_testing/imaging/cscope/Cap43/processed_video",
        #"//ceph/akrami/capsid_testing/imaging/cscope/Cap44/processed_video",
        #"//ceph/akrami/capsid_testing/imaging/cscope/Cap48/processed_video",
        #"//ceph/akrami/capsid_testing/imaging/cscope/Cap52/FMP4/processed_video",
        #"//ceph/akrami/capsid_testing/imaging/cscope/gcamp1/processed_video",
        #"//ceph/akrami/capsid_testing/imaging/cscope/Cap43/conditions/blood/no_bv",
        #"//ceph/akrami/capsid_testing/imaging/cscope/Cap44/raw_data/no_bv_crop"

    ]

    output_folders = [
        #"//ceph/akrami/capsid_testing/imaging/cscope/final_pipeline/Cap40/final_matrix",
        #"//ceph/akrami/capsid_testing/imaging/cscope/final_pipeline/Cap43/final_matrix",
        #"//ceph/akrami/capsid_testing/imaging/cscope/final_pipeline/Cap44/final_matrix",
        #"//ceph/akrami/capsid_testing/imaging/cscope/final_pipeline/Cap48/final_matrix",
        #"//ceph/akrami/capsid_testing/imaging/cscope/final_pipeline/Cap52/final_matrix",
        #"//ceph/akrami/capsid_testing/imaging/cscope/final_pipeline/gcamp1/final_matrix",
        #"//ceph/akrami/capsid_testing/imaging/cscope/final_pipeline/Cap43/no_bv_crop/roi_a/final_matrix",
        #"//ceph/akrami/capsid_testing/imaging/cscope/final_pipeline/Cap44/no_bv_crop/roi_a/final_matrix"
    ]

    for input_folder, output_folder in zip(input_folders, output_folders):
        cap_number = extract_cap_number(input_folder)
        if cap_number is None:
            print("Unable to extract Cap number from input folder.")
            continue
        process_avi_folder(input_folder, output_folder, cap_number)


