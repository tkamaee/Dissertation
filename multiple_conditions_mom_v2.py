import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from matplotlib.ticker import PercentFormatter


def process_arrays(npy_files, input_folder):
    combined_mean_responses = []
    individual_mean_responses = []

    for npy_file in npy_files:
        mean_response = np.load(os.path.join(input_folder, npy_file))
        print("npy file loaded:", mean_response.shape)
        combined_mean_responses.append(mean_response)
        print("CMR:", combined_mean_responses)
        individual_mean_responses.append(mean_response)
        print("IMR:", individual_mean_responses)

    combined_mean_responses = np.mean(combined_mean_responses, axis=0)
    print("CMR:", combined_mean_responses)
    return combined_mean_responses, individual_mean_responses

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a),a.std()  # a.std()
    h = se * scipy.stats.t.ppf((1 - confidence) / 2., n-1)
    return m, m-h, m+h

def plotvar(data, plot, timeseries=None, col_str=None, n_samples=500):
    rand_samples = [np.random.choice(data.shape[0], data.shape[0], replace=True)
                    for i in range(n_samples)]
    rand_npdample = [data[rand_indices, :].mean(axis=0)
                     for rand_indices in rand_samples]
    rand_npdample = np.array(rand_npdample)
    ci = np.apply_along_axis(mean_confidence_interval, axis=0, arr=rand_npdample)

    if col_str:
        plot[1].fill_between(time_values, ci[1, :], ci[2, :], alpha=0.1, facecolor=col_str)
    else:
        plot[1].fill_between(time_values, ci[1, :], ci[2, :], alpha=0.1)


if __name__ == "__main__":
    animal_name = "cross_animals"
    variable_name = ""
    variable_file = ""
    hpc = False
    Three = False
    stop = False
    if hpc is True:
        path_start = "//ceph/akrami"
    else:
        path_start = "Z:"

    input_folder_a = f"{path_start}/capsid_testing/imaging/cscope/final_pipeline/{animal_name}/green/Contralateral"
    input_folder_b = f"{path_start}/capsid_testing/imaging/cscope/final_pipeline/{animal_name}/green/Ipsilateral"
    if Three is True:
        input_folder_c = f"{path_start}/capsid_testing/imaging/cscope/final_pipeline/{animal_name}/vg/Ipsilateral/Cap48_52_BI"

    output_folder = f"{path_start}/capsid_testing/imaging/cscope/final_pipeline/thesis/figure8/mom"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if Three is not True and stop is not True:

        if "blue" or "B" in input_folder_a and "blue" or "B" in input_folder_b:
            light_colour = "Blue"
            if "BC" or "Contralateral" in input_folder_a:
                condition_a = "Contralateral"
                color_a = "cyan"
                if "BI" or "Ipsilateral"in input_folder_b:
                    condition_b = "Ipsilateral"
                    color_b = "dodgerblue"
            if "BI" in input_folder_a:
                color_a = "dodgerblue"
                condition_a = "Ipsilateral"
                if "BC" in input_folder_b:
                    condition_b = "Contralateral"
                    color_b = "cyan"

        if "green" or "G" in input_folder_a and "green" or "G" in input_folder_b:
            light_colour = "Green"
            if "Contralateral" in input_folder_a:
                 condition_a = "Contralateral"
                 color_a = "lime"
                 if "Ipsilateral" in input_folder_b:
                    condition_b = "Ipsilateral"
                    color_b = "green"

        if "B" in input_folder_a and "_G" in input_folder_b:
            condition_a = "Blue"
            condition_b = "Green"
            if "BC" in input_folder_a:
                light_colour = "Contralateral"
                color_a = "cyan"
                color_b = "lime"
            if "BI" in input_folder_a:
                light_colour = "Ipsilateral"
                color_a = "dodgerblue"
                color_b = "green"

    if Three is not True and stop is True:
        if "B" in input_folder_a and "B" in input_folder_b:
            light_colour = "Blue"
            if "W" in input_folder_a:
                condition_a = "Whisker"
                color_a = "dodgerblue"
                if "FP" in input_folder_b:
                    condition_b = "Hind Foot Poke"
                    color_b = "purple"
            if ".7" in input_folder_a:
                condition_a = "Ipsilateral"
                if "BC" in input_folder_b:
                    condition_b = "Contralateral"

    if Three is True:
        if "B" in input_folder_a and "B" in input_folder_b and "B" in input_folder_c:
            light_colour = "Blue"
            if "Cap44" in input_folder_a:
                condition_a = "6.71x10¹² vg "
                color_a = "red"
                if "Cap43" in input_folder_b:
                    condition_b = "3.43x10¹² vg"
                    color_b = "orange"
                if "Cap48" in input_folder_c:
                    condition_c = "1.76x10¹⁰ vg"
                    color_c = "darkkhaki"
            if ".7" in input_folder_a:
                condition_a = "Ipsilateral"
                if "BC" in input_folder_b:
                    condition_b = "Contralateral"

        if "G" in input_folder_a and "G" in input_folder_b and "G" in input_folder_c:
            light_colour = "Green"
            if "3" in input_folder_a:
                condition_a = "3% Isoflurane"
                if ".75" in input_folder_b:
                    condition_b = "0.75% Isoflurane"
                if ".7" in input_folder_c:
                    condition_c = "0.7% Isoflurane"
            if ".7" in input_folder_a:
                condition_a = "Ipsilateral"
                if "BC" in input_folder_b:
                    condition_b = "Contralateral"


    npy_files_a = [file for file in os.listdir(input_folder_a) if file.endswith(".npy")]
    npy_files_b = [file for file in os.listdir(input_folder_b) if file.endswith(".npy")]
    if Three is True:
        npy_files_c = [file for file in os.listdir(input_folder_c) if file.endswith(".npy")]

    frame_duration = 1 / 30
    num_frames = 200
    time_values = np.arange(-num_frames, num_frames + 1) * frame_duration

    combined_mean_responses_a, individual_mean_responses_a = process_arrays(npy_files_a, input_folder_a)
    combined_mean_responses_b, individual_mean_responses_b = process_arrays(npy_files_b, input_folder_b)
    print("IMR b:", individual_mean_responses_b)
    if Three is True:
        combined_mean_responses_c, individual_mean_responses_c = process_arrays(npy_files_c, input_folder_c)



    plt.figure(figsize=(10.8, 7.2), dpi=600)
    plot = plt.subplots(figsize=(10.8, 7.2))

    plot[1].plot(time_values, combined_mean_responses_a, label=f'Mean of Means {condition_a}', color=f'{color_a}')
    plot[1].plot(time_values, combined_mean_responses_b, label=f'Mean of Means {condition_b}', color=f'{color_b}')
    if Three is True:
        plot[1].plot(time_values, combined_mean_responses_c, label=f'Mean of Means  {condition_c}', color=f'{color_c}')
    # plt.plot(np.arange(-num_frames, num_frames + 1), combined_mean_responses_c, label='Mean of Means 200μL', color='forestgreen')
    plot[1].axvline(0, color='black', linestyle='dashed', label='Stimulus Frame')
    plt.xticks(fontsize=16)
    plot[1].set_xlabel('Time From Stimulus (seconds)', fontsize=16)
    plot[1].set_ylabel('\u0394F/F (%)', fontsize=16)
    plot[1].set_title(f'{light_colour}{variable_name} Stimulation Response Curves \n {condition_a} vs {condition_b}', fontsize=18)
    plot[1].yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=1))
    plt.gca().yaxis.set_tick_params(labelsize=16)
    plt.ylim(-0.15, 0.15)
    plot[1].legend(fontsize=16)


    plotvar(data=np.array(individual_mean_responses_a), plot=(plot[0], plot[1]), col_str=f'{color_a}')
    plotvar(data=np.array(individual_mean_responses_b), plot=(plot[0], plot[1]), col_str=f'{color_b}')
    if Three is True:
        plotvar(data=np.array(individual_mean_responses_c), plot=(plot[0], plot[1]), col_str=f'{color_c}')
        output_plot_file = os.path.join(output_folder, f"{variable_file}{light_colour}_{condition_a}_{condition_b}__mean_response_plot.svg")
    else:
        output_plot_file = os.path.join(output_folder, f"{variable_file}{light_colour}_{condition_a}_{condition_b}_mean_response_plot.svg")

    plt.savefig(output_plot_file)
    plt.close()

    combined_mean_responses_a = np.array(combined_mean_responses_a)
    combined_mean_responses_b = np.array(combined_mean_responses_b)
    if Three is True:
        combined_mean_responses_c = np.array(combined_mean_responses_c)
        output_file = os.path.join(output_folder, f"{variable_file}{light_colour}_{condition_a}_{condition_b}_mean_response.npy")
        np.savez(output_file, combined_mean_responses_a, combined_mean_responses_b, combined_mean_responses_c)

    else:
        output_file = os.path.join(output_folder, f"{variable_file}{light_colour}_{condition_a}_{condition_b}_mean_response.npy")
        np.savez(output_file, combined_mean_responses_a, combined_mean_responses_b)

