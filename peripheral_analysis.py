import pandas as pd
import os
import pickle
import parse_data_files
import empatica_parser
import analysis_manager

IS_REPLAY = "isReplay"
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
        mean_temps = temp_trials.groupby([parse_data_files.SUBJ_ANS_IS1, parse_data_files.TRIAL_STIM_VAL])[empatica_parser.TEMP_COL_DELTA].mean().reset_index()
        mean_temps.loc[:, parse_data_files.TRIAL_STIM_VAL] = mean_temps[parse_data_files.TRIAL_STIM_VAL].map({0: NEUTRAL, 1: AVERSIVE})
        row = [sub]
        for col in columns[1:]:
            # for each column name, add the averaged temperature for this subject in this condition (column)
            col_info = col.split(" ")
            ispas1 = 1 if col_info[0] == "PAS1" else 0
            temp = mean_temps.loc[(mean_temps[parse_data_files.SUBJ_ANS_IS1] == ispas1) & (mean_temps[parse_data_files.TRIAL_STIM_VAL] == col_info[1])][empatica_parser.TEMP_COL_DELTA].tolist()[0]
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
            # mark samples in each trial to create a sample column (SAMPLE_NUM)
            temp_trials = manual_sample_assignment(sub, temp_trials)
            mean_temps_samples = temp_trials.groupby([SAMPLE_NUM, parse_data_files.SUBJ_ANS_IS1, parse_data_files.TRIAL_STIM_VAL])[empatica_parser.TEMP_COL_DELTA].mean().reset_index()
            mean_temps_samples[parse_data_files.TRIAL_STIM_VAL].replace({0: NEUTRAL, 1: AVERSIVE}, inplace=True)
            for samp in samples:
                mean_temps_samp = mean_temps_samples[mean_temps_samples[SAMPLE_NUM] == samp]
                row = [sub]
                for col in columns[1:]:
                    # for each column name, add the averaged temperature for this subject in this condition (column)
                    col_info = col.split(" ")
                    ispas1 = 1 if col_info[0] == "PAS1" else 0
                    temp = mean_temps_samp.loc[(mean_temps_samp[parse_data_files.SUBJ_ANS_IS1] == ispas1) & (mean_temps_samp[parse_data_files.TRIAL_STIM_VAL] == col_info[1])][empatica_parser.TEMP_COL_DELTA].tolist()[0]
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
        # FILTER OUT trials where motion was present (as labeled by EDA explorer as invalid)
        new_beh = beh[beh[empatica_parser.BEH_EDA_NOISE_COL] == 0]  # 1 is noise, 0 otherwise
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
            peak_num = mean_peak_num.loc[(mean_peak_num[parse_data_files.SUBJ_ANS_IS1] == ispas1) & (mean_peak_num[parse_data_files.TRIAL_STIM_VAL] == col_info[1])][empatica_parser.BEH_EDA_PEAK_CNT].tolist()[0]
            peak_num_row.append(peak_num)

            peak_amp = mean_peak_amp.loc[(mean_peak_amp[parse_data_files.SUBJ_ANS_IS1] == ispas1) & (mean_peak_amp[parse_data_files.TRIAL_STIM_VAL] == col_info[1])][empatica_parser.BEH_EDA_PEAK_AMP].tolist()[0]
            peak_amp_row.append(peak_amp)

        peak_num_list.append(peak_num_row)
        peak_amp_list.append(peak_amp_row)

    peak_num_df = pd.DataFrame(peak_num_list, columns=columns)
    peak_num_df.to_csv(os.path.join(eda_output_path, "peak_num_averaged_per_sub.csv"))

    peak_amp_df = pd.DataFrame(peak_amp_list, columns=columns)
    peak_amp_df.to_csv(os.path.join(eda_output_path, "peak_amp_averaged_per_sub.csv"))

    return


def analyze_hr(sub_data_dict, save_path):
    hr_output_path = os.path.join(save_path, HR)
    if not (os.path.isdir(hr_output_path)):
        os.mkdir(hr_output_path)
    return


def peripheral_analysis(sub_data_dict, save_path):
    analyze_temperature(sub_data_dict, save_path)

    analyze_eda(sub_data_dict, save_path)

    #analyze_hr(sub_data_dict, save_path)
    return