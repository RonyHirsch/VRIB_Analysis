import pandas as pd
import numpy as np
import math
import os
import pickle
import parse_data_files
import empatica_parser
import analysis_manager
import plotter
import cluster_based_permutation_analysis
import gaze_analysis

SAMPLE_NUM = "sampleNum"
DELTA_TEMP = "deltaTemp"
AVERSIVE = "Aversive"
NEUTRAL = "Neutral"

TEMPERATURE = "temperature"
TEMPERATURE_NUM_SAMPLES = 252  # a trial's length is always 63 seconds, and empatica E4's sampling rate is 4Hz

EDA = "eda"

HR = "hr"

columns = [analysis_manager.SUB, f"PAS1 {NEUTRAL}", f"PAS1 {AVERSIVE}", f"PAS2-4 {NEUTRAL}", f"PAS2-4 {AVERSIVE}"]


def manual_sample_assignment(sub, temp_trials):
    """
    As temperature is measured by Empatica E4 with 4Hz sampling rate and each trial is 1:03 minutes, we expect each
    trial to have TEMPERATURE_NUM_SAMPLES samples. In reality, some trials might have TEMPERATURE_NUM_SAMPLES +-1
    and so, in order to add a "sample number" column to temp_trials, we need to manually count how many samples
    were actually in each trial.
    :param sub: subject code
    :param temp_trials: df, to be added a column named SAMPLE_NUM where each trial it marks each row with a sample number
    :return: temp_trials with [SAMPLE_NUM] column
    """
    trials_not_matching = [i for i in range(parse_data_files.NUM_TRIALS) if temp_trials[temp_trials[parse_data_files.TRIAL_NUMBER] == i].shape[0] != TEMPERATURE_NUM_SAMPLES]
    if len(trials_not_matching) > 0:
        print(f"FYI: Subject {sub} has {len(trials_not_matching)} trials with more/less than {TEMPERATURE_NUM_SAMPLES} samples")
    for i in range(parse_data_files.NUM_TRIALS):  # for each trial
        num_samps = temp_trials[temp_trials[parse_data_files.TRIAL_NUMBER] == i].shape[0]  # count how many samples are in this trial
        temp_trials.loc[temp_trials[parse_data_files.TRIAL_NUMBER] == i, SAMPLE_NUM] = [samp for samp in range(num_samps)]  # and number them from 0 to the last one
    return temp_trials


def calculate_avg_dT_in_trial(sub_data_dict, temp_output_path):
    """
    Calculate the average dT in trials grouped by conditions.
    :param sub_data_dict:
    :param temp_output_path:
    :return:
    """
    subjects = list(sub_data_dict.keys())
    result_list = list()
    result_list_long = list()
    for sub in subjects:
        temp = sub_data_dict[sub][empatica_parser.EMPATICA_OUTPUT_FOLDER][empatica_parser.TEMP]
        temp_trials = temp[temp[parse_data_files.TRIAL_NUMBER] != -1]  # just real trials

        temp_trials_long = temp_trials.groupby([parse_data_files.TRIAL_NUMBER, parse_data_files.SUBJ_ANS, parse_data_files.TRIAL_STIM_VAL]).mean().reset_index()
        temp_trials_long.loc[:, analysis_manager.SUB] = sub
        result_list_long.append(temp_trials_long)

        mean_temps = temp_trials.groupby([parse_data_files.SUBJ_ANS_IS1, parse_data_files.TRIAL_STIM_VAL])[
            empatica_parser.TEMP_COL_DELTA].mean().reset_index()
        mean_temps.loc[:, parse_data_files.TRIAL_STIM_VAL] = mean_temps[parse_data_files.TRIAL_STIM_VAL].map({0: NEUTRAL, 1: AVERSIVE})
        row = [sub]
        for col in columns[1:]:
            # for each column name, add the averaged temperature for this subject in this condition (column)
            col_info = col.split(" ")
            ispas1 = 1 if col_info[0] == "PAS1" else 0
            try:
                temp = mean_temps.loc[(mean_temps[parse_data_files.SUBJ_ANS_IS1] == ispas1) & (mean_temps[parse_data_files.TRIAL_STIM_VAL] == col_info[1])][empatica_parser.TEMP_COL_DELTA].tolist()[0]
            except IndexError:  # this means that there are NO TRIALS WITH THIS CONDITION FOR THIS SUBJECT (IN UAT/AT)
                temp = np.nan
            row.append(temp)
        result_list.append(row)
    result_df = pd.DataFrame(result_list, columns=columns)
    result_df.to_csv(os.path.join(temp_output_path, "dT_averaged_per_sub.csv"))

    result_df_long = pd.concat(result_list_long)
    result_df_long.to_csv(os.path.join(temp_output_path, "dT_averaged_per_sub_long.csv"), index=False)
    return result_df


def calculate_continuous_dT_in_trial(sub_data_dict, temp_output_path, samp_output_path):
    """
    Create a dictionary where keys=conditions (PAS type x Valence type; 4 keys) and value = df where
    column = subject, and row = sample number. The value in each cell is the averaged dT value for this sample number
    for this subject (across their trials in this condition).
    (1 per subject)
    :param sub_data_dict: subject dictionary with all the data
    :param temp_output_path: temperature output path
    :param samp_output_path: per-sample analysis output path
    :return: result_dict: key=cond, value = df with row per sample, col per sub, values are averaged dT in sample
    """
    subjects = list(sub_data_dict.keys())
    temp_per_samp_file = os.path.join(temp_output_path, "dT_averaged_per_sample.pickle")
    if os.path.exists(temp_per_samp_file):
        fl = open(temp_per_samp_file, 'rb')
        result_dict = pickle.load(fl)
        fl.close()
    else:
        condition_dfs = {col: list() for col in columns[1:]}
        result_dict = dict()

        for sub in subjects:
            temp = sub_data_dict[sub][empatica_parser.EMPATICA_OUTPUT_FOLDER][empatica_parser.TEMP]
            temp_trials = temp[temp[parse_data_files.TRIAL_NUMBER] != -1]  # just real trials
            # mark samples in each trial to create a sample column (SAMPLE_NUM)
            temp_trials = manual_sample_assignment(sub, temp_trials)
            mean_temps_samples = \
            temp_trials.groupby([SAMPLE_NUM, parse_data_files.SUBJ_ANS_IS1, parse_data_files.TRIAL_STIM_VAL])[
                empatica_parser.TEMP_COL_DELTA].mean().reset_index()
            mean_temps_samples[parse_data_files.TRIAL_STIM_VAL].replace({0: NEUTRAL, 1: AVERSIVE}, inplace=True)
            mean_temps_samples[parse_data_files.SUBJ_ANS_IS1].replace({0: "PAS2-4", 1: "PAS1"}, inplace=True)
            for condition in condition_dfs:
                condition_cols = condition.split(" ")
                cond_data = mean_temps_samples[(mean_temps_samples[parse_data_files.SUBJ_ANS_IS1] == condition_cols[0]) & (mean_temps_samples[parse_data_files.TRIAL_STIM_VAL] == condition_cols[1])]
                cond_data.rename(columns={empatica_parser.TEMP_COL_DELTA: sub}, inplace=True)
                cond_data.drop(columns=[parse_data_files.SUBJ_ANS_IS1, parse_data_files.TRIAL_STIM_VAL], inplace=True)
                cond_data.set_index(SAMPLE_NUM, inplace=True)
                condition_dfs[condition].append(cond_data)

        for condition in condition_dfs:
            condition_df = pd.concat(condition_dfs[condition], axis=1)
            condition_df = condition_df.reset_index().rename(columns={condition_df.index.name: SAMPLE_NUM})
            result_dict[condition] = condition_df

        # save to a pickle of that dictionary
        fl = open(temp_per_samp_file, 'ab')
        pickle.dump(result_dict, fl)
        fl.close()

    for cond in result_dict:
        result_dict[cond].to_csv(os.path.join(samp_output_path, f"dT_averaged_per_sample_{cond}.csv"))

    return result_dict


def plot_cluster_analysis(dTs_for_processing, pas1_cluster_p_vals, pas24_cluster_p_vals, save_path):
    # for the significance bar, take only significant clusters
    sig_dict = dict()
    pas1_sig = pas1_cluster_p_vals[pas1_cluster_p_vals["significant"]]
    if not(pas1_sig.empty):
        pas1_sig_x = [(x1, x2) for x1, x2 in zip(list(pas1_sig["cluster_starts"]), list(pas1_sig["cluster_ends"]))]
        ymax = max(max(dTs_for_processing[f"PAS1 {NEUTRAL}"]["mean"].tolist()), max(dTs_for_processing[f"PAS1 {AVERSIVE}"]["mean"].tolist()))
        sig_dict["pas1lines"] = {"y": ymax, "x": pas1_sig_x}

    pas24_sig = pas24_cluster_p_vals[pas24_cluster_p_vals["significant"]]
    if not (pas24_sig.empty):
        pas24_sig_x = [(x1, x2) for x1, x2 in zip(list(pas24_sig["cluster_starts"]), list(pas24_sig["cluster_ends"]))]
        ymax = max(max(dTs_for_processing[f"PAS2-4 {NEUTRAL}"]["mean"].tolist()), max(dTs_for_processing[f"PAS2-4 {AVERSIVE}"]["mean"].tolist()))
        sig_dict["pas24lines"] = {"y": ymax, "x": pas24_sig_x}


    colors = {f"PAS1 {NEUTRAL}": "#00798C", f"PAS1 {AVERSIVE}": "#ED6A5A", f"PAS2-4 {NEUTRAL}": "#22395B", f"PAS2-4 {AVERSIVE}": "#88292F"}
    labels = {f"PAS1 {NEUTRAL}": f"PAS1 {NEUTRAL}", f"PAS1 {AVERSIVE}": f"PAS1 {AVERSIVE}", f"PAS2-4 {NEUTRAL}": f"PAS2-4 {NEUTRAL}", f"PAS2-4 {AVERSIVE}": f"PAS2-4 {AVERSIVE}"}
    avg_cols = {f"PAS1 {NEUTRAL}": "mean", f"PAS1 {AVERSIVE}": "mean", f"PAS2-4 {NEUTRAL}": "mean", f"PAS2-4 {AVERSIVE}": "mean"}
    se_cols = {f"PAS1 {NEUTRAL}": "se", f"PAS1 {AVERSIVE}": "se", f"PAS2-4 {NEUTRAL}": "se", f"PAS2-4 {AVERSIVE}": "se"}

    colors2 = dict()
    labels2 = dict()
    avg_cols2 = dict()
    se_cols2 = dict()

    for key in dTs_for_processing.keys():
        colors2[key] = colors[key]
        labels2[key] = labels[key]
        avg_cols2[key] = avg_cols[key]
        se_cols2[key] = se_cols[key]

    plot_title = "Evolution of change in temperature (ΔT) between conditions"
    plotter.plot_avg_line_dict(title=plot_title, trial_df_dict=dTs_for_processing,
                               avg_col_list=avg_cols2, se_col_list=se_cols2,
                               label_list=labels2, y_name=f"Average ΔT in sample", x_name="sample",
                               color_list=colors2, significance_bars_dict=sig_dict,
                               save_path=save_path, x_tick_intervals=5,
                               save_name=f"dT_averaged_per_sample")

    return


def filter_sub_data_dict_per_cond(sub_data_dict, condition):
    result_dict = dict()
    if condition == parse_data_files.UAT:
        for sub in sub_data_dict:
            result_dict[sub] = dict()
            sub_dict = sub_data_dict[sub]
            for key in sub_dict.keys():
                if key == parse_data_files.ET_DATA_NAME or key == parse_data_files.UNITY_OUTPUT_FOLDER:
                    data = sub_data_dict[sub][key]
                    result_dict[sub][key] = data[data[parse_data_files.TRIAL_NUMBER] < 40]
                else:  # key = empatica
                    empatica = sub_data_dict[sub][key]
                    result_dict[sub][key] = dict()
                    for measurement in empatica.keys():
                        if measurement != empatica_parser.EDA_EXPLORER:
                            data = sub_data_dict[sub][key][measurement]
                            result_dict[sub][key][measurement] = data[data[parse_data_files.TRIAL_NUMBER] < 40]
                        else:
                            result_dict[sub][key][measurement] = data

    else:  # cond = parse_data_files.AT
        for sub in sub_data_dict:
            result_dict[sub] = dict()
            sub_dict = sub_data_dict[sub]
            for key in sub_dict.keys():
                if key == parse_data_files.ET_DATA_NAME or key == parse_data_files.UNITY_OUTPUT_FOLDER:
                    data = sub_data_dict[sub][key]
                    result_dict[sub][key] = data[data[parse_data_files.TRIAL_NUMBER] >= 40]
                else:  # key = empatica
                    empatica = sub_data_dict[sub][key]
                    result_dict[sub][key] = dict()
                    for measurement in empatica.keys():
                        if measurement != empatica_parser.EDA_EXPLORER:
                            data = sub_data_dict[sub][key][measurement]
                            result_dict[sub][key][measurement] = data[data[parse_data_files.TRIAL_NUMBER] >= 40]
                        else:
                            result_dict[sub][key][measurement] = data
    return result_dict


def analyze_temperature(sub_data_dict, save_path):
    """
    In this analysis we will perform 2 analyses:
    1. Following Salomon et al., 2013:
    Averaged change in temperature (ΔT) between conditions (stimulus valence x experiment part).
    This code saves a dataframe per analysis, readymade to be run by JASP's RMANOVA functionality.
    2. Evolution of ΔT over time (trial): a cluster-based permutation analysis.
    The analysis is performed in Python here


    :param sub_data_dict: The dictionary containing the behavior and empatica measurements of each subject
    :param save_path: the path to which to save the results
    :return:
    """

    temp_output_path = os.path.join(save_path, TEMPERATURE)
    if not (os.path.isdir(temp_output_path)):
        os.mkdir(temp_output_path)

    conds = [parse_data_files.UAT, parse_data_files.AT]

    for condition in conds:  # separate ET analysis between attended from unattended trials
        cond_output_path = os.path.join(temp_output_path, condition)
        if not (os.path.isdir(cond_output_path)):
            os.mkdir(cond_output_path)

        sub_data_dict_cond = filter_sub_data_dict_per_cond(sub_data_dict, condition)

        # *******ANALYSIS 1*******: AVERAGED ΔT
        dT_across_trial = calculate_avg_dT_in_trial(sub_data_dict_cond, cond_output_path)


        # *******ANALYSIS 2*******: evolution of temperature over time (per sample)
        # Create a dictionary key=experimental condition, and value=a dataframe where each row is a sample,
        # and each column is a subject's average ΔT in that sample over trials of that condition

        samp_output_path = os.path.join(cond_output_path, "per_sample")
        if not (os.path.isdir(samp_output_path)):
            os.mkdir(samp_output_path)

        dT_within_trial = calculate_continuous_dT_in_trial(sub_data_dict_cond, cond_output_path, samp_output_path)
        # remove sample number column
        dTs_for_processing = dict()
        for cond in dT_within_trial.keys():
            dTs_for_processing[cond] = dT_within_trial[cond].drop(columns=SAMPLE_NUM)
            dTs_for_processing[cond] = dTs_for_processing[cond].loc[0:TEMPERATURE_NUM_SAMPLES - 1, :]  # there are only 252 samples in 1:03 minutes of temperature recoding at 4Hz!, loc is end-inclusive

        if not dTs_for_processing[f"PAS1 Neutral"].empty and not dTs_for_processing[f"PAS1 Aversive"].empty:
            pas1_cluster_p_vals, pas1_cluster_full_df, pas1_cluster_mass_list = \
                cluster_based_permutation_analysis.permutation_cluster_paired_ttest(data_cond_1=dTs_for_processing[f"PAS1 Neutral"], data_cond_2=dTs_for_processing[f"PAS1 Aversive"])
            pas1_cluster_full_df.to_csv(os.path.join(samp_output_path, f"dT_valence_cluster_based_permutation_PAS1.csv"))

        if not dTs_for_processing[f"PAS2-4 Neutral"].empty and not dTs_for_processing[f"PAS2-4 Aversive"].empty:
            pas24_cluster_p_vals, pas24_cluster_full_df, pas24_cluster_mass_list = \
                cluster_based_permutation_analysis.permutation_cluster_paired_ttest(
                    data_cond_1=dTs_for_processing[f"PAS2-4 Neutral"], data_cond_2=dTs_for_processing[f"PAS2-4 Aversive"])
            pas24_cluster_full_df.to_csv(os.path.join(samp_output_path, f"dT_valence_cluster_based_permutation_PAS2-4.csv"))

        dTs_for_plotting = dict()
        for cond in dTs_for_processing.keys():
            if not dTs_for_processing[cond].empty:
                dTs_for_processing[cond].loc[:, 'mean'] = dTs_for_processing[cond].iloc[:, :].mean(axis=1)
                dTs_for_processing[cond].loc[:, 'std'] = dTs_for_processing[cond].iloc[:, :].std(axis=1)
                dTs_for_processing[cond].loc[:, 'se'] = dTs_for_processing[cond].loc[:, 'std'] / np.sqrt(dTs_for_processing[cond].iloc[:, :].shape[1])
                dTs_for_processing[cond].to_csv(os.path.join(samp_output_path, f"dT_averaged_per_sample_{cond}.csv"))  # update the saved data
                dTs_for_plotting[cond] = dTs_for_processing[cond]

        plot_cluster_analysis(dTs_for_plotting, pas1_cluster_p_vals, pas24_cluster_p_vals, cond_output_path)

    # Prepare UAT and AT data for comparison
    comp_output_path = os.path.join(temp_output_path, "comparison")
    if not (os.path.isdir(comp_output_path)):
        os.mkdir(comp_output_path)

    avg_temp = {parse_data_files.UAT: None, parse_data_files.AT: None}

    relevant_cols = [analysis_manager.SUB, parse_data_files.TRIAL_NUMBER,
                   parse_data_files.SUBJ_ANS, parse_data_files.SUBJ_ANS_IS1, parse_data_files.SUBJ_ANS_IS12,
                   parse_data_files.TRIAL_STIM_VAL, empatica_parser.TEMP_COL_DELTA]

    for condition in avg_temp.keys():  # separate ET analysis between attended from unattended trials
        cond_output_path = os.path.join(temp_output_path, condition)
        avg_temp_in_cond = pd.read_csv(os.path.join(cond_output_path, f"dT_averaged_per_sub_long.csv"))
        avg_temp[condition] = avg_temp_in_cond.loc[:, relevant_cols]

    unified_df = gaze_analysis.unify_for_comparison(avg_temp, comp_output_path, f"dT_averaged_per_sub_long.csv", "condition")
    return


def plot_peripheral(df, columns, name, path, plot_title, plot_x_name, plot_y_name, x_col_color_order):

    long_df = pd.melt(df, id_vars=columns[0], value_vars=columns[1:])  # id column, "variable" column, and "value" column
    # the conditions will be plotted as separate columns by their alphabetical order, and so their labels need to change
    variable_labels = sorted(long_df["variable"].unique())
    variable_conversion_dict = {variable_labels[i]: i for i in range(len(variable_labels))}  # the values
    long_df["variable"].replace(variable_conversion_dict, inplace=True)
    group_name_mapping = {v: k for k, v in variable_conversion_dict.items()}  # conversion back

    y_min_val = np.nanmin(long_df["value"])
    y_max_val = np.nanmax(long_df["value"])
    y_tick_skip = 0.5

    plotter.plot_raincloud(df=long_df, x_col_name="variable",
                           y_col_name="value",
                           plot_title=plot_title,
                           plot_x_name=plot_x_name,
                           plot_y_name=plot_y_name,
                           save_path=path, save_name=name,
                           y_tick_interval=y_tick_skip, y_tick_min=y_min_val, y_tick_max=y_max_val,
                           x_axis_names=[group_name_mapping[i] for i in sorted(long_df["variable"].unique())],
                           y_tick_names=None,
                           group_col_name="variable", group_name="Condition",
                           group_name_mapping=group_name_mapping,
                           x_col_color_order=x_col_color_order,
                           x_values=sorted(long_df["variable"].unique()), alpha_step=0, valpha=0.9)
    return


def analyze_eda(sub_data_dict, save_path):

    eda_output_path = os.path.join(save_path, EDA)
    if not (os.path.isdir(eda_output_path)):
        os.mkdir(eda_output_path)

    conds = [parse_data_files.UAT, parse_data_files.AT]
    for condition in conds:  # separate ET analysis between attended from unattended trials
        cond_output_path = os.path.join(eda_output_path, condition)
        if not (os.path.isdir(cond_output_path)):
            os.mkdir(cond_output_path)

        sub_data_dict_cond = filter_sub_data_dict_per_cond(sub_data_dict, condition)

        subjects = list(sub_data_dict_cond.keys())
        peak_num_list = list()
        peak_amp_list = list()

        peak_list_long = list()

        for sub in subjects:
            beh = sub_data_dict_cond[sub][parse_data_files.UNITY_OUTPUT_FOLDER]
            # FILTER OUT trials where motion was present (as labeled by EDA explorer as invalid)
            new_beh = beh[beh[empatica_parser.BEH_EDA_NOISE_COL] == 0]  # 1 is noise, 0 otherwise

            new_beh_long = new_beh.groupby([parse_data_files.TRIAL_NUMBER, parse_data_files.SUBJ_ANS, parse_data_files.TRIAL_STIM_VAL]).mean().reset_index()
            new_beh_long.loc[:, analysis_manager.SUB] = sub
            relevant_cols = [analysis_manager.SUB, parse_data_files.TRIAL_NUMBER,
                             parse_data_files.SUBJ_ANS, parse_data_files.SUBJ_ANS_IS1, parse_data_files.SUBJ_ANS_IS12,
                             parse_data_files.TRIAL_STIM_VAL, empatica_parser.BEH_EDA_PEAK_AMP, empatica_parser.BEH_EDA_PEAK_CNT]
            new_beh_long = new_beh_long.loc[:, relevant_cols]
            peak_list_long.append(new_beh_long)

            # average peaks across trials of different types
            mean_peak_num = new_beh.groupby([parse_data_files.SUBJ_ANS_IS1, parse_data_files.TRIAL_STIM_VAL])[empatica_parser.BEH_EDA_PEAK_CNT].mean().reset_index()
            mean_peak_num.loc[:, parse_data_files.TRIAL_STIM_VAL] = mean_peak_num[parse_data_files.TRIAL_STIM_VAL].map({0: NEUTRAL, 1: AVERSIVE})
            peak_num_row = [sub]
            mean_peak_amp = new_beh.groupby([parse_data_files.SUBJ_ANS_IS1, parse_data_files.TRIAL_STIM_VAL])[empatica_parser.BEH_EDA_PEAK_AMP].mean().reset_index()
            mean_peak_amp.loc[:, parse_data_files.TRIAL_STIM_VAL] = mean_peak_amp[parse_data_files.TRIAL_STIM_VAL].map({0: NEUTRAL, 1: AVERSIVE})
            peak_amp_row = [sub]
            for col in columns[1:]:
                # for each column name, add the averaged # of peaks and amp for this subject in this condition (column)
                col_info = col.split(" ")
                ispas1 = 1 if col_info[0] == "PAS1" else 0
                try:
                    peak_num = mean_peak_num.loc[(mean_peak_num[parse_data_files.SUBJ_ANS_IS1] == ispas1) & (mean_peak_num[parse_data_files.TRIAL_STIM_VAL] == col_info[1])][empatica_parser.BEH_EDA_PEAK_CNT].tolist()[0]
                except IndexError:  # this means that there are NO TRIALS WITH THIS CONDITION FOR THIS SUBJECT (IN UAT/AT)
                    peak_num = np.nan
                peak_num_row.append(peak_num)
                try:
                    peak_amp = mean_peak_amp.loc[(mean_peak_amp[parse_data_files.SUBJ_ANS_IS1] == ispas1) & (mean_peak_amp[parse_data_files.TRIAL_STIM_VAL] == col_info[1])][empatica_parser.BEH_EDA_PEAK_AMP].tolist()[0]
                except IndexError:  # this means that there are NO TRIALS WITH THIS CONDITION FOR THIS SUBJECT (IN UAT/AT)
                    peak_amp = np.nan
                peak_amp_row.append(peak_amp)

            peak_num_list.append(peak_num_row)
            peak_amp_list.append(peak_amp_row)

        peak_num_df = pd.DataFrame(peak_num_list, columns=columns)
        peak_num_df.to_csv(os.path.join(cond_output_path, "peak_num_averaged_per_sub.csv"))

        peak_amp_df = pd.DataFrame(peak_amp_list, columns=columns)
        peak_amp_df.to_csv(os.path.join(cond_output_path, "peak_amp_averaged_per_sub.csv"))

        plot_peripheral(df=peak_num_df, columns=columns, name="peak_num_averaged_per_sub", path=cond_output_path,
                        plot_title="Average Number of EDA Peaks in Condition", plot_x_name="Condition",
                        plot_y_name="Average Number of SCR Peaks in Trial",
                        x_col_color_order=[["#88292F", "#22395B", "#ED6A5A", "#00798C"],
                                           ["#88292F", "#22395B", "#ED6A5A", "#00798C"],
                                           ["#88292F", "#22395B", "#ED6A5A", "#00798C"],
                                           ["#88292F", "#22395B", "#ED6A5A", "#00798C"]])

        result_df_long = pd.concat(peak_list_long)
        result_df_long.to_csv(os.path.join(cond_output_path, "peak_averaged_per_sub_long.csv"), index=False)

    comp_output_path = os.path.join(eda_output_path, "comparison")
    if not (os.path.isdir(comp_output_path)):
        os.mkdir(comp_output_path)

    avg_peak = {parse_data_files.UAT: None, parse_data_files.AT: None}

    for condition in avg_peak.keys():  # separate peak analysis between attended from unattended trials
        cond_output_path = os.path.join(eda_output_path, condition)
        avg_peak_in_cond = pd.read_csv(os.path.join(cond_output_path, "peak_averaged_per_sub_long.csv"))
        avg_peak[condition] = avg_peak_in_cond

    unified_df = gaze_analysis.unify_for_comparison(avg_peak, comp_output_path, f"avg_gaze_per_pas_long.csv", "condition")

    return


def analyze_hr(sub_data_dict, save_path):
    hr_output_path = os.path.join(save_path, HR)
    if not (os.path.isdir(hr_output_path)):
        os.mkdir(hr_output_path)
    return


def filter_out_noisy_trials(sub_data_dict):
    """
    Filter out trials by EDA-Explorer's noise indicator: in the behavioral dataframe we created a column denoting whether
    a noise event was registered in this trial. If it was - this trial is being filtered out.
    :param sub_data_dict: dictionary with all the behavior, et and empatica measurement
    :return: THE SAME DICT with all the empatica measurement dataframes filtered s.t. they contain ZERO samples out of
    trials that were marked as noisy
    """
    for subject in sub_data_dict.keys():
        sub_data = sub_data_dict[subject]
        sub_beh = sub_data[parse_data_files.UNITY_OUTPUT_FOLDER]
        sub_empatica = sub_data[empatica_parser.EMPATICA_OUTPUT_FOLDER]
        # EXTRACT ALL TRIALS THAT **DID** HAVE EMPATICA NOISE IN THEM
        beh_noisy_trials = sub_beh[sub_beh[empatica_parser.BEH_EDA_NOISE_COL] == 1][parse_data_files.TRIAL_NUMBER].tolist()
        new_measurements = dict()
        for measurement in sub_empatica.keys():
            if measurement != empatica_parser.EDA_EXPLORER:
                new_measurements[measurement] = sub_empatica[measurement].loc[~sub_empatica[measurement][parse_data_files.TRIAL_NUMBER].isin(beh_noisy_trials), :]
            else:
                new_measurements[measurement] = sub_empatica[measurement]
        sub_data_dict[subject][empatica_parser.EMPATICA_OUTPUT_FOLDER] = new_measurements
    return sub_data_dict


def peripheral_analysis(sub_data_dict, save_path):

    sub_data_dict = filter_out_noisy_trials(sub_data_dict)

    analyze_temperature(sub_data_dict, save_path)

    analyze_eda(sub_data_dict, save_path)

    #analyze_hr(sub_data_dict, save_path)
    return