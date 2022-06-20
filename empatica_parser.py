import os
import pandas as pd
from pandas.api.types import is_string_dtype
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
from dateutil import tz
import parse_data_files

ACC = "ACC"
BVP = "BVP"
EDA = "EDA"
EDA_EXPLORER = "EDA_EXPLORER"
HR = "HR"
IBI = "IBI"
TEMP = "TEMP"
TIME = "time"
TIME_NORM = "time (normalized)"
TEMP_COL = "degrees °C"
BILLBOARD_GAZE = "billboardGaze"
IS_GAZE = "isGaze"
DATA_TYPE_NAMES = [ACC, BVP, EDA, HR, IBI, TEMP]
IBI_1 = "time of detected inter-beat interval (sec) with respect to start time"
IBI_2 = "IBI duration (sec, distance in seconds from the previous beat)"
DATA_TYPE_COLUMNS = {ACC: ["x", "y", "z"], BVP: [BVP], EDA: [f"{EDA} (μS)"],
                     HR: [HR], IBI: [IBI_1, IBI_2],
                     TEMP: [TEMP_COL]}  # derived from the empatica "info" file
TEMP_COL_DELTA = "temperature delta from trial start (degrees °C)"
SAMPLE = "sample"
SAMPLE_RATE = "sample rate (Hz)"
TIME_DELTA_MS = "time delta (ms)"
TIMEZONE = tz.gettz('Israel')
EMPATICA_OUTPUT_FOLDER = "empatica"
EDA_EXPLORER = "eda_explorer"
EDA_PEAK_FILENAME = "_PeakFeatures_Lit Review Based.csv"  # this is the name of the filter I used in EDA-Explorer
EDA_NOISE_FILENAME = "_NoiseLabels_Binary.csv"
EDA_NOISE_START = "StartTime"
EDA_NOISE_END = "EndTime"
EDA_NOISE_COLNAME = "BinaryLabels"
EDA_PEAKS_AMP_COL = "amp"
BEH_EDA_PEAK_CNT = "numOfPeaks"
BEH_EDA_PEAK_AMP = "avgAmplitude"
BEH_EDA_NOISE_COL = "edaIsNoise"

REPLAY_TRIAL = 40


def load_empatica_data(sub_empatica_path):
    empatica_data = dict()
    for dtype in DATA_TYPE_NAMES:
        data_file = [f for f in os.listdir(sub_empatica_path) if dtype in f][0]
        data = pd.read_csv(os.path.join(sub_empatica_path, data_file), header=None)
        data.columns = DATA_TYPE_COLUMNS[dtype]
        time_from_stamp = dt.datetime.fromtimestamp(data.iloc[0, 0], tz=TIMEZONE) # The first row is the initial time of the session expressed as unix timestamp in UTC
        if dtype != IBI:
            sample_rate = data.iloc[1, 0]  # The second row is the sample rate expressed in Hz
            actual_data = data.iloc[2:, :]  # The rest are the data recordings
            actual_data.loc[:, SAMPLE_RATE] = sample_rate
            actual_data.loc[:, SAMPLE] = list(range(0, actual_data.shape[0]))
            actual_data.loc[:, TIME_DELTA_MS] = [s*(1/sample_rate)*1000 for s in list(actual_data[SAMPLE])]
            actual_data.loc[:, TIME] = [(time_from_stamp + dt.timedelta(milliseconds=ms)).time() for ms in actual_data[TIME_DELTA_MS].tolist()]

        else:  # IBI has no sample rate
            actual_data = data.iloc[1:, :]
            actual_data.loc[:, SAMPLE_RATE] = np.nan
            actual_data.loc[:, SAMPLE] = np.nan
            actual_data.loc[:, TIME_DELTA_MS] = [float(d)*1000 for d in actual_data[IBI_2].tolist()]
            actual_data.loc[:, TIME] = [(time_from_stamp + dt.timedelta(seconds=s)).time() for s in actual_data[IBI_1].tolist()]

        actual_data.loc[:, parse_data_files.DATE] = f"{time_from_stamp.year}-{time_from_stamp.month}-{time_from_stamp.day}"
        col_order = [SAMPLE_RATE, parse_data_files.DATE, SAMPLE, TIME_DELTA_MS, TIME] + DATA_TYPE_COLUMNS[dtype]
        actual_data = actual_data[col_order]
        empatica_data[dtype] = actual_data
    return empatica_data


def time_in_range(start, end, t):
    """Return true if x is in the range [start, end]"""
    if start <= end:
        return start <= t <= end
    else:
        return start <= t or t <= end


def parse_empatica_trial_data(empatica_data, sub_trial_data, sub_output_path):
    """
    :param empatica_data: dict of empatica data dataframes, output of load_empatica_data
    :param sub_trial_data: dataframe with all the beh + ET data of the subject, from parse_data_files
    :return:
    """

    for col in [parse_data_files.TRIAL_START, parse_data_files.TRIAL_END]:
        if is_string_dtype(sub_trial_data[col].tolist()[0]):
            sub_trial_data[col] = [dt.datetime.strptime(s, "%H:%M:%S.%f").time() for s in sub_trial_data[col].tolist()]

    empatica_result_file = os.path.join(sub_output_path, "empatica_data_dict.pickle")
    if os.path.exists(empatica_result_file):
        fl = open(empatica_result_file, 'rb')
        empatica_data = pickle.load(fl)
        fl.close()

    else:
        for data_type in empatica_data:  # set a trial column in each data type
            data = empatica_data[data_type]
            data[parse_data_files.TRIAL_NUMBER] = -1  # set a default value if sample is not in any trial

        for index, trial in sub_trial_data.iterrows():  # iterate and mark trials
            trial_number = trial[parse_data_files.TRIAL_NUMBER]
            trial_start_time = trial[parse_data_files.TRIAL_START]  # datetime.time
            trial_end_time = trial[parse_data_files.TRIAL_END]  # datetime.time
            for data_type in empatica_data:
                data = empatica_data[data_type]
                for jndex, sample in data.iterrows():
                    sample_time = sample[TIME]
                    if time_in_range(start=trial_start_time, end=trial_end_time, t=sample_time):
                        data.at[jndex, parse_data_files.TRIAL_NUMBER] = trial_number
        # save to a pickle of that dictionary
        fl = open(empatica_result_file, 'ab')
        pickle.dump(empatica_data, fl)
        fl.close()

    return empatica_data


def normalize_time(time_to_normalize, first_time):
    date = dt.date(1, 1, 1)
    datetime1 = dt.datetime.combine(date, first_time)
    datetime2 = dt.datetime.combine(date, time_to_normalize)
    time_elapsed = datetime2 - datetime1  # this is a "deltatime" object
    # (dt.datetime.min + time_elapsed).time()  # return THIS to return a "datetime.time" object instead
    return time_elapsed.total_seconds() * 1000  # returns result as difference in milliseconds


def parse_empatica_valence_data(empatica_data, sub_trial_data):
    trials = sub_trial_data[parse_data_files.TRIAL_NUMBER].tolist()
    dtypes = list(empatica_data.keys())
    for data_type in dtypes:
        data = empatica_data[data_type]
        data[parse_data_files.TRIAL_STIM_VAL] = -1  # dummy value to be filled
        for trial_number in trials:
            trial_valence = sub_trial_data[sub_trial_data[parse_data_files.TRIAL_NUMBER] == trial_number][parse_data_files.TRIAL_STIM_VAL].tolist()[0]
            data.loc[data[parse_data_files.TRIAL_NUMBER] == trial_number, [parse_data_files.TRIAL_STIM_VAL]] = trial_valence
        empatica_data[data_type] = data
    return empatica_data


def unify_empatica_data(empatica_data_dict):
    merged = list()
    for data_type in empatica_data_dict:
        alt_data = empatica_data_dict[data_type].drop(columns=[parse_data_files.DATE, SAMPLE, SAMPLE_RATE, TIME_DELTA_MS], inplace=False)
        merged.append(alt_data)
    all_empatica = pd.concat(merged, ignore_index=True).sort_values(TIME)
    all_empatica = all_empatica.groupby(TIME).mean().reset_index(inplace=False, drop=False)
    all_empatica[TIME_NORM] = all_empatica.apply(lambda row: normalize_time(row[TIME], all_empatica.loc[0, TIME]), axis=1)
    return all_empatica


def plot_empatica_data(sub_code, empatica_df, sub_trial_data, save_path):
    x_axis = empatica_df[TIME_NORM]

    DTYPE = "data"
    COLOR = "color"
    DTYPE_NAME = "name"
    data_dict = {
        0: {DTYPE_NAME: BVP, DTYPE: DATA_TYPE_COLUMNS[BVP], COLOR: "#08808E"},
        1: {DTYPE_NAME: HR, DTYPE: DATA_TYPE_COLUMNS[HR], COLOR: "#7AC9D0"},
        2: {DTYPE_NAME: EDA, DTYPE: DATA_TYPE_COLUMNS[EDA], COLOR: "#FE9A84"},
        3: {DTYPE_NAME: TEMP, DTYPE: DATA_TYPE_COLUMNS[TEMP], COLOR: "#E74632"}
            }

    # identify the neutral and aversive trials, and mark those samples
    negative_trial_nums = sub_trial_data[sub_trial_data[parse_data_files.TRIAL_STIM_VAL] == 1][parse_data_files.TRIAL_NUMBER].tolist()
    neutral_trial_nums = sub_trial_data[sub_trial_data[parse_data_files.TRIAL_STIM_VAL] != 1][parse_data_files.TRIAL_NUMBER].tolist()

    negative_trial_samples = empatica_df[empatica_df[parse_data_files.TRIAL_NUMBER].isin(negative_trial_nums)][TIME_NORM]
    neutral_trial_samples = empatica_df[empatica_df[parse_data_files.TRIAL_NUMBER].isin(neutral_trial_nums)][TIME_NORM]

    valence_dict = {0: {DTYPE: neutral_trial_samples, COLOR: "#1C1C1C", "alpha": 0.25},
                    1: {DTYPE: negative_trial_samples, COLOR: "#1C1C1C", "alpha": 0.45}}

    # identify the first timestamp in the replay and mark it
    replay_begin = empatica_df[empatica_df[parse_data_files.TRIAL_NUMBER] == REPLAY_TRIAL].loc[:, TIME_NORM].tolist()[0]

    # Let's plot the results
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(nrows=len(data_dict.keys()), sharex=True, figsize=(20, 10))
    fig.suptitle(f"{sub_code} RAW Peripheral Data")

    for i in range(len(data_dict.keys())):
        t = data_dict[i][DTYPE]
        # plot the data
        axes[i].plot(x_axis, empatica_df[t], color=data_dict[i][COLOR], marker='.', markersize=2)
        axes[i].set(ylabel=data_dict[i][DTYPE_NAME])
        # add a line separating the game from replay
        axes[i].axvline(x=replay_begin, color="#947EB0", lw=5)
        # mark the negative and neutral trials
        ymin = empatica_df[t].min()
        ymax = empatica_df[t].max()
        for j in range(len(valence_dict.keys())):
            temp_set = set(valence_dict[j][DTYPE].tolist())
            x_cond = [1 if x in temp_set else 0 for x in x_axis]  # color the trials
            axes[i].fill_between(x_axis, ymin, ymax, where=x_cond, color=valence_dict[j][COLOR], alpha=valence_dict[j]["alpha"])

    figure = plt.gcf()  # get current figure
    plt.savefig(os.path.join(save_path, f"empatica.png"))
    plt.clf()
    plt.close()
    return


def preprocess_temperature_delta(empatica_data_dict):
    temperature_data = empatica_data_dict[TEMP]
    # take only actual trial samples
    """
    Calculate temperature delta according to Salomon, et al.,2013's method: 
    Subtract the skin temperature at the first time point from all subsequent time points for each trial. 
    """
    temperature_data[TEMP_COL_DELTA] = temperature_data[TEMP_COL]  # start by copying the temperature data
    trials = temperature_data[parse_data_files.TRIAL_NUMBER].unique()
    for trial in trials:
        trial_data = temperature_data[temperature_data[parse_data_files.TRIAL_NUMBER] == trial]
        first_trial_temperature = trial_data.loc[:, TEMP_COL].tolist()[0]
        temperature_data.loc[temperature_data[parse_data_files.TRIAL_NUMBER] == trial, [TEMP_COL_DELTA]] -= first_trial_temperature
    # nullify the delta-temperature for time inbetween trials ("-1" trials), as this column is NOT a right calculation on them
    temperature_data.loc[temperature_data[parse_data_files.TRIAL_NUMBER] == -1, [TEMP_COL_DELTA]] = np.nan
    empatica_data_dict[TEMP] = temperature_data
    return empatica_data_dict


def add_range_data_column(data, additional_col_name, additional_df, start_col, end_col, time_col, unifying_col):
    """
    :param data: The dataframe to add an extra information column to
    :param additional_col_name: the extra information column name in "data"
    :param additional_df: the source of the extra information we want to add - information about range
    :param start_col: column name in "additional_df" denoting the start of a range
    :param end_col: column name in "additional_df" denoting the end of a range
    :param unifying_col: for example, trial number. A column that exists in both dataframes
    :return: data, with an additional column denoting for each row whether it is within a range.
    """
    unifying_col_range = data[unifying_col].unique().tolist()
    for val in unifying_col_range:
        additional_df_trial = additional_df[additional_df[unifying_col] == val]
        data_trial = data[data[unifying_col] == val]
        for ind, row in additional_df_trial.iterrows():
            data.at[data_trial[(row[start_col] <= data_trial[time_col]) & (row[end_col] >= data_trial[time_col])].index, additional_col_name] = 1

    return data


def preprocess_temperature_data(sub_trial_data, empatica_data, sub_busstop_gaze_data, sub_output_path):
    """
        *** TEMPERATURE ***
        --> METHOD : Salomon et al.,2013 (Frontiers in behavioral neuroscience)
        With this method, for analysis of the evolution of temperature over the course of the trial, changes in temperature
        are calculated by subtracting the temperature at the start of the trial from all subsequent time points in that trial
        """
    empatica_data_temp_updated = preprocess_temperature_delta(empatica_data)
    temperature_data = empatica_data_temp_updated[TEMP]

    """
    *** GAZE ***
    We would like to cross the information of skin temperature with the information about the stimulus being in 
    the fovea of the subject's gaze
    """
    # Take only the gaze on the intact stimuli - gaze on scrambled bus stops is currently not interesting
    sub_busstop_gaze_data_intact = sub_busstop_gaze_data[sub_busstop_gaze_data[parse_data_files.IS_INTACT] == True]

    # mark the temperature data samples with the valence of the trial
    temperature_data.loc[:, parse_data_files.SUBJ_ANS_IS1] = -1  # default value
    temperature_data.loc[:, parse_data_files.SUBJ_ANS_IS12] = -1  # default value

    # mark in the temperature data samples that occurred when subject's gaze was on an intact stimulus
    trials = temperature_data[parse_data_files.TRIAL_NUMBER].unique().tolist()
    trials.remove(-1)  # -1 is not a real trial, but a default value
    for val in trials:
        additional_df_trial = sub_busstop_gaze_data_intact[sub_busstop_gaze_data_intact[parse_data_files.TRIAL_NUMBER] == val]
        data_trial = temperature_data[temperature_data[parse_data_files.TRIAL_NUMBER] == val]
        # ADD TRIAL DATA TO TEMPERATURE DATA
        beh_trial = sub_trial_data[sub_trial_data[parse_data_files.TRIAL_NUMBER] == val]
        temperature_data.at[temperature_data[parse_data_files.TRIAL_NUMBER] == val, parse_data_files.SUBJ_ANS_IS1] = beh_trial[parse_data_files.SUBJ_ANS_IS1].tolist()[0]
        temperature_data.at[temperature_data[parse_data_files.TRIAL_NUMBER] == val, parse_data_files.SUBJ_ANS_IS12] = beh_trial[parse_data_files.SUBJ_ANS_IS12].tolist()[0]

        for bustop in additional_df_trial[parse_data_files.BUSSTOP].unique().tolist():
            bustop_df_trial = additional_df_trial[additional_df_trial[parse_data_files.BUSSTOP] == bustop]
            for ind, row in bustop_df_trial.iterrows():
                gazed_billboards = data_trial[(row[parse_data_files.GAZE_START] <= data_trial[TIME]) & (
                            row[parse_data_files.GAZE_END] >= data_trial[TIME])]
                temperature_data.at[gazed_billboards.index, parse_data_files.BUSSTOP] = bustop
                temperature_data.at[gazed_billboards.index, IS_GAZE] = 1

    empatica_data_temp_updated[TEMP] = temperature_data
    # save to csv
    temperature_data.to_csv(os.path.join(sub_output_path, "sub_temperature_data.csv"), index=False)
    return empatica_data_temp_updated


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


def preprocess_eda_data(sub_trial_data, general_output_path, sub_code):
    """
    EDA is processed using EDA-EXPLORER
    Taylor, S., Jaques, N., Chen, W., Fedor, S., Sano, A., and Picard, R.
    Automatic Identification of Artifacts in Electrodermal Activity Data" In Proc.
    International Conference of the IEEE Engineering in Medicine and Biology Society (EMBC), Milan, Italy, August 2015.
    Here we parse the information from EDA-EXPLORER, correct the time shift (UTC TO IST) and add this information.
    :param sub_trial_data:
    :return:
    """
    eda_preprocessed_path = os.path.join(general_output_path, EDA_EXPLORER, sub_code)
    if not os.path.isdir(eda_preprocessed_path):
        print(f"ERROR: subject {sub_code} does not have pre-processed EDA data. This subject is SKIPPED during analysis")
        return None
    eda_peaks, eda_noise = load_preprocessed_eda(sub_code, eda_preprocessed_path)
    # Mark trials where there was noise as measured in the noise file (BEH_EDA_NOISE_COL col)
    sub_trial_data_eda_noise = noise_based_trial_exclusion(sub_trial_data, eda_noise)
    sub_trial_data_eda_peaks = count_eda_peaks_per_trial(sub_trial_data_eda_noise, eda_peaks)
    return sub_trial_data_eda_peaks, eda_peaks, eda_noise


def preprocess_empatica_data(empatica_data, sub_busstop_gaze_data, sub_trial_data, sub_output_path, general_output_path, sub_code):
    # preprocess the temperature data
    empatica_data_temp_updated = preprocess_temperature_data(sub_trial_data, empatica_data, sub_busstop_gaze_data, sub_output_path)
    # preprocess the EDA data and re-save sub trial data with it
    sub_trial_data, eda_peaks, eda_noise = preprocess_eda_data(sub_trial_data, general_output_path, sub_code)
    sub_trial_data.to_csv(os.path.join(sub_output_path, "sub_trial_data.csv"), index=False)
    empatica_data_temp_updated[EDA_EXPLORER] = [eda_peaks, eda_noise]
    return sub_trial_data, empatica_data_temp_updated


def load_sub_peripheral_data(sub_path, sub_trial_data, sub_busstop_gaze_data, sub_output_path, sub_code, general_output_path):
    sub_empatica_path = os.path.join(sub_path, EMPATICA_OUTPUT_FOLDER)  # path to raw empatica data
    empatica_data = load_empatica_data(sub_empatica_path)  # load dict of empatica data
    # mark trials in each empatica data table: add a trial column
    empatica_data = parse_empatica_trial_data(empatica_data, sub_trial_data, sub_output_path)
    # add columns
    processed_empatica = os.path.join(sub_output_path, "empatica_data_dict_processed.pickle")
    if os.path.exists(processed_empatica):
        fl = open(processed_empatica, 'rb')
        empatica_data = pickle.load(fl)
        fl.close()
    else:
        # add a column depicting if the stimulus in this trial was negative or neutral
        empatica_data = parse_empatica_valence_data(empatica_data, sub_trial_data)
        # unify all empatica data types into a single dataframe
        unified_empatica_data = unify_empatica_data(empatica_data)
        plot_empatica_data(sub_code, unified_empatica_data, sub_trial_data, sub_output_path)
        # Pre-process the raw empatica data
        sub_trial_data, empatica_data = preprocess_empatica_data(empatica_data, sub_busstop_gaze_data, sub_trial_data, sub_output_path, general_output_path, sub_code)
        # save to a pickle of that dictionary
        fl = open(processed_empatica, 'ab')
        pickle.dump(empatica_data, fl)
        fl.close()

    return sub_trial_data, empatica_data
