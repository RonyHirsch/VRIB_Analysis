import pandas as pd
import os
import datetime as dt
import pickle
import numbers
import re
import math
import numpy as np
import itertools
import seaborn as sns
import plotter
import parse_data_files
import empatica_parser
import analysis_manager

IS_REPLAY = "isReplay"
CONDITION = "condition"
SAMPLE_NUM = "sampleNum"
DELTA_TEMP = "deltaTemp"
REPLAY = "Attended"
GAME = "Unattended"
AVERSIVE = "Aversive"
NEUTRAL = "Neutral"

TEMPERATURE = "temperature"
TEMPERATURE_NUM_SAMPLES = 252  # a trial's length is always 63 seconds, and empatica E4's sampling rate is 4Hz

EDA = "eda"
EDA_PEAK_FILENAME = "_PeakFeatures_Lit Review Based.csv"  # this is the name of the filter I used in EDA-Explorer
EDA_NOISE_FILENAME = "_NoiseLabels_Binary.csv"
EDA_NOISE_COLNAME = "BinaryLabels"
EDA_NOISE_START = "StartTime"
EDA_NOISE_END = "EndTime"
EDA_PEAKS_AMP_COL = "amp"
BEH_EDA_PEAK_CNT = "numOfPeaks"
BEH_EDA_PEAK_AMP = "avgAmplitude"

BEH_EDA_NOISE_COL = "edaIsNoise"

columns = [analysis_manager.SUB, f"{GAME} {NEUTRAL}", f"{GAME} {AVERSIVE}", f"{REPLAY} {NEUTRAL}", f"{REPLAY} {AVERSIVE}"]


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


def analyze_temperature(sub_data_dict, save_path):
    """
    In this analysis we are following 2 analyses types performed by Salomon et al., 2013:
    1. Averaged change in temperature (ΔT) between conditions (stimulus valence x experiment part).
    2. Evolution of ΔT over time (trial).
    This code saves a dataframe per analysis, readymade to be run by JASP's RMANOVA functionality.

    :param sub_data_dict: The dictionary containing the behavior and empatica measurements of each subject
    :param save_path: the path to which to save the results
    :return:
    """

    temp_output_path = os.path.join(save_path, TEMPERATURE)
    if not (os.path.isdir(temp_output_path)):
        os.mkdir(temp_output_path)

    # *******ANALYSIS 1*******: AVERAGED ΔT
    subjects = list(sub_data_dict.keys())
    result_list = list()
    for sub in subjects:
        temp = sub_data_dict[sub][empatica_parser.EMPATICA_OUTPUT_FOLDER][empatica_parser.TEMP]
        temp_trials = temp[temp[parse_data_files.TRIAL_NUMBER] != -1]  # just real trials
        # mark experiment conditions in column
        temp_trials.loc[temp_trials[parse_data_files.TRIAL_NUMBER] >= empatica_parser.REPLAY_TRIAL, CONDITION] = REPLAY
        temp_trials.loc[temp_trials[parse_data_files.TRIAL_NUMBER] < empatica_parser.REPLAY_TRIAL, CONDITION] = GAME
        mean_temps = temp_trials.groupby([CONDITION, parse_data_files.TRIAL_STIM_VAL])[empatica_parser.TEMP_COL_DELTA].mean().reset_index()
        mean_temps.loc[:, parse_data_files.TRIAL_STIM_VAL] = mean_temps[parse_data_files.TRIAL_STIM_VAL].map({0: NEUTRAL, 1: AVERSIVE})
        row = [sub]
        for col in columns[1:]:
            # for each column name, add the averaged temperature for this subject in this condition (column)
            col_info = col.split(" ")
            temp = mean_temps.loc[(mean_temps[CONDITION] == col_info[0]) & (mean_temps[parse_data_files.TRIAL_STIM_VAL] == col_info[1])][empatica_parser.TEMP_COL_DELTA].tolist()[0]
            row.append(temp)
        result_list.append(row)
    result_df = pd.DataFrame(result_list, columns=columns)
    result_df.to_csv(os.path.join(temp_output_path, "dT_averaged_per_sub.csv"))

    # *******ANALYSIS 2*******: evolution of temperature over time (per sample)
    # Create a dictionary (samp_dict_df) where key=sample number, and value=a dataframe where each row is a subject,
    # each column is a condition, and each cell contains the averaged ΔT for that subject in that condition IN THIS SAMPLE!!

    temp_per_samp_file = os.path.join(temp_output_path, "dT_averaged_per_sample.pickle")
    if os.path.exists(temp_per_samp_file):
        fl = open(temp_per_samp_file, 'rb')
        samp_dict_df = pickle.load(fl)
        fl.close()
    else:
        # prepare a list samples per trial
        samples = [samp for samp in range(TEMPERATURE_NUM_SAMPLES)]
        samps_per_trial_nested = [samples for i in range(parse_data_files.NUM_TRIALS)]
        samps_per_trial = [x for xs in samps_per_trial_nested for x in xs]

        # prepare a dict with df for each timepoint
        samp_dict = {k: list() for k in samples}
        samp_dict_df = dict()

        for sub in subjects:
            temp = sub_data_dict[sub][empatica_parser.EMPATICA_OUTPUT_FOLDER][empatica_parser.TEMP]
            temp_trials = temp[temp[parse_data_files.TRIAL_NUMBER] != -1]  # just real trials
            # mark experiment conditions in column
            temp_trials.loc[temp_trials[parse_data_files.TRIAL_NUMBER] >= empatica_parser.REPLAY_TRIAL, CONDITION] = REPLAY
            temp_trials.loc[temp_trials[parse_data_files.TRIAL_NUMBER] < empatica_parser.REPLAY_TRIAL, CONDITION] = GAME
            # mark samples in each trial to create a sample column (SAMPLE_NUM)
            temp_trials = manual_sample_assignment(sub, temp_trials)
            mean_temps_samples = temp_trials.groupby([SAMPLE_NUM, CONDITION, parse_data_files.TRIAL_STIM_VAL])[empatica_parser.TEMP_COL_DELTA].mean().reset_index()
            mean_temps_samples[parse_data_files.TRIAL_STIM_VAL].replace({0: NEUTRAL, 1: AVERSIVE}, inplace=True)
            for samp in samples:
                mean_temps_samp = mean_temps_samples[mean_temps_samples[SAMPLE_NUM] == samp]
                row = [sub]
                for col in columns[1:]:
                    # for each column name, add the averaged temperature for this subject in this condition (column)
                    col_info = col.split(" ")
                    temp = mean_temps_samp.loc[(mean_temps_samp[CONDITION] == col_info[0]) & (mean_temps_samp[parse_data_files.TRIAL_STIM_VAL] == col_info[1])][empatica_parser.TEMP_COL_DELTA].tolist()[0]
                    row.append(temp)
                samp_dict[samp].append(row)

        for s in samp_dict.keys():
            samp_dict_df[s] = pd.DataFrame(samp_dict[s], columns=columns)

        # save to a pickle of that dictionary
        fl = open(temp_per_samp_file, 'ab')
        pickle.dump(samp_dict_df, fl)
        fl.close()

    samp_output_path = os.path.join(temp_output_path, "per_sample")
    if not (os.path.isdir(samp_output_path)):
        os.mkdir(samp_output_path)

    for s in samp_dict_df:
        samp_dict_df[s].to_csv(os.path.join(samp_output_path, f"dT_averaged_per_sample_{s}.csv"))

    return samp_dict_df


def convert_cols_to_time(df, cols):
    for col in cols:
        for ind, row in df.iterrows():
            try:
                new_cell = dt.datetime.strptime(row[col], '%Y-%m-%d %H:%M:%S.%f').time()
            except ValueError:
                new_cell = dt.datetime.strptime(row[col], '%Y-%m-%d %H:%M:%S').time()
            df.at[ind, col] = new_cell
    return df


def convert_utc_to_ist(df, cols):
    for col in cols:
        df.loc[:, col] = df[col].apply(lambda x: dt.datetime.combine(dt.date(1, 1, 1), x))  # convert to be able to add hours
        df.loc[:, col] = df[col].apply(lambda x: (x + dt.timedelta(hours=3)).time())  # add the time and convert back to time
    return df


def load_preprocessed_eda(sub, sub_eda_file_path):
    noise_file_path = os.path.join(sub_eda_file_path, f"{sub}{EDA_NOISE_FILENAME}")
    noise_file = pd.read_csv(noise_file_path)
    peak_file_path = os.path.join(sub_eda_file_path, f"{sub}{EDA_PEAK_FILENAME}")
    peak_file = pd.read_csv(peak_file_path)
    # NOTE: the timezone of Israel is 3 hours ahead of UTC which is the empatica E4 timestamp.
    # This is why we need to add 3 hours to ALL timestamps in these files!

    # Turn into time:
    noise_df = convert_cols_to_time(noise_file, cols=[EDA_NOISE_START, EDA_NOISE_END])
    peak_df = convert_cols_to_time(peak_file, cols=[peak_file.columns[0]])
    # Convert UTC time to IST : add 3 hours
    noise_df = convert_utc_to_ist(noise_df, cols=[EDA_NOISE_START, EDA_NOISE_END])
    peak_df = convert_utc_to_ist(peak_df, cols=[peak_file.columns[0]])
    return peak_df, noise_df


def noise_based_trial_exclusion(beh_df, eda_noise_df):
    """

    :param beh_df: the trial behavior + ET data (as extracted from the Unity experiment)
    :param eda_noise_df: a dataframe that is the output of "EDA Explorer" processing (ref below). In this dataframe,
    the first column contains time-stamps that break the raw data into 5-second epochs.
    The second column contains either a -1, 1, or 0 in each row, representing whether the corresponding epoch is a
    noise (-1), is clean (1), or is questionable (0).
    See:
    Taylor, S., Jaques, N., Chen, W., Fedor, S., Sano, A., and Picard, R.
    Automatic Identification of Artifacts in Electrodermal Activity Data" In Proc.
    International Conference of the IEEE Engineering in Medicine and Biology Society (EMBC), Milan, Italy, August 2015.
    :return:
    """
    beh_df.loc[:, BEH_EDA_NOISE_COL] = 0  # start with all trials being valid
    eda_noise_df_noise = eda_noise_df[eda_noise_df[EDA_NOISE_COLNAME] != 1]  # all the invalied 5-sec time windows
    for ind, row in eda_noise_df_noise.iterrows():  # for each noise row
        row_start = row[EDA_NOISE_START]
        row_end = row[EDA_NOISE_END]
        for trial_ind, beh_row in beh_df.iterrows():
            # if there is noise within the trial
            if (beh_row[parse_data_files.TRIAL_START] <= row_start <= beh_row[parse_data_files.TRIAL_END]) or \
                    (beh_row[parse_data_files.TRIAL_START] <= row_end <= beh_row[parse_data_files.TRIAL_END]):
                beh_df.at[trial_ind, BEH_EDA_NOISE_COL] = 1
    return beh_df


def count_eda_peaks_per_trial(beh_df, eda_peaks):

    beh_df.loc[:, BEH_EDA_PEAK_CNT] = 0  # start with having no peaks
    beh_df.loc[:, BEH_EDA_PEAK_AMP] = float(0)
    for ind, row in eda_peaks.iterrows():  # for each peak row
        peak_time = row[eda_peaks.columns[0]]
        peak_amp = row[EDA_PEAKS_AMP_COL]
        for trial_ind, beh_row in beh_df.iterrows():
            # if there is a peak within the trial
            if (beh_row[parse_data_files.TRIAL_START] <= peak_time <= beh_row[parse_data_files.TRIAL_END]):
                beh_df.at[trial_ind, BEH_EDA_PEAK_CNT] += 1
                beh_df.at[trial_ind, BEH_EDA_PEAK_AMP] += peak_amp  # this is currently the sum. At the end, we'll divide the sum by BEH_EDA_PEAK_CNT to get the average

    # the BEH_EDA_PEAK_AMP should be the AVERAGE amplitude: divide the sum by the number of peaks leading to it.
    beh_df.loc[:, BEH_EDA_PEAK_AMP] = beh_df[BEH_EDA_PEAK_AMP] / beh_df[BEH_EDA_PEAK_CNT]
    return beh_df


def analyze_eda(sub_data_dict, save_path):

    eda_output_path = os.path.join(save_path, EDA)
    if not (os.path.isdir(eda_output_path)):
        os.mkdir(eda_output_path)

    subjects = list(sub_data_dict.keys())
    excluded = 0
    peak_num_list = list()
    peak_amp_list = list()
    for sub in subjects:
        beh = sub_data_dict[sub][parse_data_files.UNITY_OUTPUT_FOLDER]
        eda_preprocessed_path = os.path.join(save_path, parse_data_files.PER_SUBJECT, sub, EDA)
        if not os.path.isdir(eda_preprocessed_path):
            print(f"ERROR: subject {sub} does not have pre-processed EDA data. This subject is SKIPPED during analysis")
            excluded += 1
            continue
        eda_peaks, eda_noise = load_preprocessed_eda(sub, eda_preprocessed_path)
        # First thing first: mark trials where there was noise as measured in the noise file (BEH_EDA_NOISE_COL col)
        beh_eda_noise = noise_based_trial_exclusion(beh, eda_noise)
        beh_eda_peaks = count_eda_peaks_per_trial(beh_eda_noise, eda_peaks)
        # resave subject trial data:
        beh_eda_peaks.to_csv(os.path.join(os.path.join(save_path, parse_data_files.PER_SUBJECT, sub), "sub_trial_data.csv"), index=False)

        # FILTER OUT trials where motion was present (as labeled by EDA explorer as invalid)
        new_beh = beh_eda_peaks[beh_eda_peaks[BEH_EDA_NOISE_COL] == 0]  # 1 is noise, 0 otherwise
        # average peaks across trials of different types
        mean_peak_num = new_beh.groupby([CONDITION, parse_data_files.TRIAL_STIM_VAL])[BEH_EDA_PEAK_CNT].mean().reset_index()
        mean_peak_num.loc[:, parse_data_files.TRIAL_STIM_VAL] = mean_peak_num[parse_data_files.TRIAL_STIM_VAL].map({0: NEUTRAL, 1: AVERSIVE})
        peak_num_row = [sub]
        mean_peak_amp = new_beh.groupby([CONDITION, parse_data_files.TRIAL_STIM_VAL])[BEH_EDA_PEAK_AMP].mean().reset_index()
        mean_peak_amp.loc[:, parse_data_files.TRIAL_STIM_VAL] = mean_peak_amp[parse_data_files.TRIAL_STIM_VAL].map({0: NEUTRAL, 1: AVERSIVE})
        peak_amp_row = [sub]
        for col in columns[1:]:
            # for each column name, add the averaged # of peaks and amp for this subject in this condition (column)
            col_info = col.split(" ")
            peak_num = mean_peak_num.loc[(mean_peak_num[CONDITION] == col_info[0]) & (mean_peak_num[parse_data_files.TRIAL_STIM_VAL] == col_info[1])][BEH_EDA_PEAK_CNT].tolist()[0]
            peak_num_row.append(peak_num)

            peak_amp = mean_peak_amp.loc[(mean_peak_amp[CONDITION] == col_info[0]) & (mean_peak_amp[parse_data_files.TRIAL_STIM_VAL] == col_info[1])][BEH_EDA_PEAK_AMP].tolist()[0]
            peak_amp_row.append(peak_amp)

        peak_num_list.append(peak_num_row)
        peak_amp_list.append(peak_amp_row)

    peak_num_df = pd.DataFrame(peak_num_list, columns=columns)
    peak_num_df.to_csv(os.path.join(eda_output_path, "peak_num_averaged_per_sub.csv"))

    peak_amp_df = pd.DataFrame(peak_amp_list, columns=columns)
    peak_amp_df.to_csv(os.path.join(eda_output_path, "peak_amp__averaged_per_sub.csv"))

    return


def peripheral_analysis(sub_data_dict, save_path):
    #behavioral_data = pd.concat([sub_data_dict[sub][parse_data_files.UNITY_OUTPUT_FOLDER] for sub in sub_data_dict],
    #                            keys=sub_data_dict.keys(), names=[analysis_manager.SUB, None]).reset_index(level=analysis_manager.SUB)
    # temperature analysis
    #temperature_data = pd.concat([sub_data_dict[sub][empatica_parser.EMPATICA_OUTPUT_FOLDER][empatica_parser.TEMP] for sub in sub_data_dict],
    #                             keys=sub_data_dict.keys(), names=[analysis_manager.SUB, None]).reset_index(level=analysis_manager.SUB)

    #analyze_temperature(sub_data_dict, save_path)

    analyze_eda(sub_data_dict, save_path)
    return