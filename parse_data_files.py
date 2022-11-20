import os
import pandas as pd
import numpy as np
import datetime as dt
import empatica_parser

UNITY_OUTPUT_FOLDER = "beh"
ET_DATA_NAME = "et"
TRIAL_NUMBER = "trialNumber"
TRIAL_STIM_ID = "stimIndex"
TRIAL_STIM_NAME = "stimPicName"
TRIAL_STIM_VAL = "stimValence"
TRIAL_SCRAMBLED_LOCS = "scrambledLocations"
CLUES_TAKEN = "cluesTaken"
BEE_ANS = "beeScore"
BEE_SELECT_LOC = "beeSelectedLocation"
TRIAL_MONEY = "trialScore"
OBJ_ANS = "objectiveIsCorrect"
OBJ_TIME = "objectiveQStartTime"
OBJ_TIME_SELECTED = "objectiveQSelectedTime"
OBJ_TARGET_LOC = "objectiveTargetLoc"
OBJ_RT = "objectiveRTSec"
OBJ_BUSSTOP = "objectiveTimeFromLastIntactSec"
SUBJ_ANS = "subjectiveAwareness"
SUBJ_TIME = "subjectiveQStartTime"
SUBJ_TIME_SELECTED = "subjectiveQSelectedTime"
SUBJ_RT = "subjectiveRTSec"
SUBJ_BUSSTOP = "subjectiveTimeFromLastIntactSec"
SUBJ_ANS_IS1 = "isPAS1"
SUBJ_ANS_IS12 = "isPAS12"
VAL_ANS = "valenceJudgement"
VAL_ANS_CORRECT = "isCorrectInValenceJudgement"
BUSSTOP_GAZE_DUR = "busstopGazeDuration"
BUSSTOP_GAZE_NUM = "busstopGazeNumOfPeriods"
BUSSTOP_GAZE_DUR_TOTAL = "busstopGazeDurTotalSum"
BUSSTOP_GAZE_DUR_AVG_TOTAL = "busstopGazeDurTotalAvg"
BUSSTOP_GAZE_DUR_STD_TOTAL = "busstopGazeDurTotalStd"
BUSSTOP_GAZE_DUR_INTACT = "busstopGazeDurIntactSum"
BUSSTOP_GAZE_DUR_AVG_INTACT = "busstopGazeDurIntactAvg"
BUSSTOP_GAZE_DUR_STD_INTACT = "busstopGazeDurIntactStd"
BUSSTOP_GAZE_DUR_MIN_INTACT = "busstopGazeDurIntactMin"
BUSSTOP_GAZE_DUR_MAX_INTACT = "busstopGazeDurIntactMax"
IS_INTACT = "isIntact"
BUSSTOP_GAZE_DUR_SCRAMBLED = "busstopGazeDurScrambledSum"
BUSSTOP_GAZE_DUR_AVG_SCRAMBLED = "busstopGazeDurScrambledAvg"
BUSSTOP_GAZE_DUR_STD_SCRAMBLED = "busstopGazeDurScrambledStd"
BUSSTOP_GAZE_DUR_MIN_SCRAMBLED = "busstopGazeDurScrambledMin"
BUSSTOP_GAZE_DUR_MAX_SCRAMBLED = "busstopGazeDurScrambledMax"
BUSSTOP = "busStop"
# file and file content data
FILE_RANDOMIZE_TRIALS = "Block.randomizeTrials_"
FILE_MOTION_PIVOT = "MotionPivot"
MOTION_BILLBOARD = "ClosestBillboardPassingNow"
FILE_STIM_ORDER = "stimOrder"
CHOSEN_PIC_ORDER = "chosenPicsOrder"
CHOSEN_PIC_VAL = "chosenPicsValence"
START_TIME = "startTime"
END_TIME = "endTime"
PIC_AVERSIVE_IND = list(range(0, 20))
PIC_NEUTRAL_IND = list(range(20, 40))
SCRAMBELOO = "scrambeloo"
# scoreCalc
FILE_SCORE_CALC = "ScoreCalc.ScoreCalc_"
SCORE_GAIN = "PointsWhenCorrect"
SCORE_LOSS = "PointsWhenWrong"
SCORE_CLUE = "PointsWhenClue"
SCORE_REPORT = "CurrPoints"
SCORE_START = "startMoney"
FILE_SCORE_CALC_ALTERNATIVE = "trialScoreData"
# target bee
FILE_TARGET_BEE = "TagertBeeSelection.controllerProjection."
BEE_CORRECT = "isCorrectInBeeSelection"
SELECTED_LOC = "selectedTargetInd"
BEE_VALUE = 2
# randomizeStim (randomization of scrambled/intact locations)
FILE_RANDOMIZE_STIM = "ExpStimRandomizer.randomizeStim_"
SCRAMBLED_LOCS = "ScrambledLocationsInTrial"
LAST_INTACT_LOC = "LastIntactLocationInTrial"
LAST_INTACT_TIME = "LastIntactTimeInTrial"
# objective question
OBJ_Q = "ObjTrialAwareness."
OBJ_Q_RANDOMIZE_OPTIONS = "ObjTrialAwareness.randomizeOptions_"
OBJ_CORRECT_LOC = ".correctAnsLoc"
OBJ_SELECTED_LOC = ".selectedTargetInd"
OBJ_IS_CORRECT = ".isCorrectAnswer"
OBJ_TIMINGS = ".objectiveQProjection_"
OBJ_Q_START = "RealObjStart"
OBJ_Q_SELECTED = "RealObjSelected"
IN = "IN"
OUT = "OUT"
TARGET = "TARGET"
# subjective question
SUBJ_Q = "SubjTrialAwareness.subjectiveQProjection.selectedTargetInd"
SUBJ_SELECTED_LOC = "selectedTargetInd"
SUBJ_TIMINGS = ".subjectiveQProjection_"
SUBJ_Q_START = "RealSubjStart"
SUBJ_Q_SELECTED = "RealSubjSelected"
# velance question
VALENCE_Q = "ValTrialAwareness.valenceQProjection.selectedTargetInd"
# ET data: images' recordAtGaze
IMAGE_RECORD = "Tobii.XR.Examples.RecordAtGaze"
IMAGE_RECORD_BACKUP = "RecordAtGazeBackup" # backup file
IMAGE = "Image"
BUSSTOP_FIRST = 2
BUSSTOP_LAST = 11
NUM_BUSSTOPS_IN_TRIAL = 10
GAZE_DURATION = "GazeDuration"  # STOPPED RELYING ON THIS COLUMN: CALCULATING ENDTIME - STARTTIME
GAZE_START = "startGazeTime"
GAZE_END = "endGazeTime"
ET_RECORD_RAW = "ETRecord"
REPLAY_LEVELS = [x for x in range(40, 50)]
NUM_TRIALS = 50
# trial start and end (ride): bus motion information
FILE_BUS_MOTION = "BusBeesAndPlayer.BusMotion"
DATE = "date"
TRIAL_START_TIME = "RealTrialStart"  # datetime.timestamp!
TRIAL_END_TIME = "RealBusStopTime"  # datetime.timestamp!
TRIAL_START = "TrialStart"  # datetime.time
TRIAL_END = "TrialEnd"  # datetime.time
TRIAL_DUR_SEC = "TrialDurSec"
CONDITION = "condition"
REPLAY = "Attended"
GAME = "Unattended"
# folders and file names
PER_SUBJECT = "per_subject"
UAT = "unattended"
AT = "attended"


def label_valence(stim_number_row, col=TRIAL_STIM_ID):
    if stim_number_row[col] in PIC_AVERSIVE_IND:
        return 1
    else:
        return 0


def load_scrambeloo(sub_path):
    scrambeloo_file_names = [f for f in os.listdir(sub_path) if SCRAMBELOO in f]
    stim_id_list = list()
    trial_scrambled_list = list()
    for f in scrambeloo_file_names: # each file is a stimulus file (i.e., not trial order)
        f_data = open(os.path.join(sub_path, f), "r")
        content = f_data.read()
        f_data.close()
        data_list = content.split(" ")
        stim_id = int(data_list[0])  # the first number in each file is the STIMULUS ID
        scrambled = data_list[1:]
        scrambled_locs = [int(x) for x in scrambled]
        scrambled_locs_string = ';'.join(map(str, scrambled_locs))
        stim_id_list.append(stim_id)
        trial_scrambled_list.append(scrambled_locs_string)

    trial_scrambled_dict = {TRIAL_STIM_ID: stim_id_list, TRIAL_SCRAMBLED_LOCS: trial_scrambled_list}
    trial_scrambled_df = pd.DataFrame.from_dict(trial_scrambled_dict)
    return trial_scrambled_df


def last_intact_in_trial(row):
    if isinstance(row[TRIAL_SCRAMBLED_LOCS], str):
        scrambled = sorted(row[TRIAL_SCRAMBLED_LOCS].split(";"))
        intact = [int(x) for x in range(NUM_BUSSTOPS_IN_TRIAL) if not str(x) in scrambled]
        last_intact = max(intact)
        return last_intact
    return np.nan


def mark_motion_pivot_trials(motion_pivot):
    motion_pivot.reset_index(drop=True, inplace=True)
    trial_number = 0
    last_billboard = 0
    motion_pivot.loc[:, TRIAL_NUMBER] = 0
    for ind, row in motion_pivot.iterrows():
        if int(row[2]) > last_billboard:  # increasing billboard serial number, within the same trial
            last_billboard = int(row[2])
        else:
            last_billboard = int(row[2])
            trial_number += 1
        motion_pivot.at[ind, TRIAL_NUMBER] = trial_number
    return motion_pivot


def mark_last_busstop_time(trial_data, motion_pivot):
    trial_data.loc[:, LAST_INTACT_TIME] = np.nan
    for ind, row in trial_data.iterrows():
        trial = row[TRIAL_NUMBER]
        if np.isnan(row[LAST_INTACT_LOC]):
            trial_data.loc[ind, LAST_INTACT_TIME] = np.nan
        else:
            loc = str(int(row[LAST_INTACT_LOC]))
            trial_data.loc[ind, LAST_INTACT_TIME] = dt.datetime.strptime(motion_pivot[(motion_pivot[TRIAL_NUMBER] == trial) & (motion_pivot[2] == loc)][3].tolist()[0], '%Y-%m-%d %H:%M:%S.%f')
    return trial_data


def load_trial_stim_info(sub_path):
    # load stimulus index and valence
    rand_trials_name = [f for f in os.listdir(sub_path) if FILE_RANDOMIZE_TRIALS in f][0]
    rand_trials_data = pd.read_csv(os.path.join(sub_path, rand_trials_name), sep="\t")
    rand_trials_data = rand_trials_data.drop(columns=[rand_trials_data.columns[0]])
    block_data = rand_trials_data[rand_trials_data.apply(lambda row: row.astype(str).str.contains(CHOSEN_PIC_ORDER, case=False).any(), axis=1)]
    if not(block_data.empty):
        trial_data_dict = {TRIAL_NUMBER: [i for i in range(len(rand_trials_data.loc[0, rand_trials_data.columns[1]].split(";")))],
                           rand_trials_data.loc[0, rand_trials_data.columns[0]]: rand_trials_data.loc[0, rand_trials_data.columns[1]].split(";"),
                           rand_trials_data.loc[1, rand_trials_data.columns[0]]: rand_trials_data.loc[1, rand_trials_data.columns[1]].split(";")}
        trial_data = pd.DataFrame.from_dict(trial_data_dict)
        trial_data.rename(columns={CHOSEN_PIC_ORDER: TRIAL_STIM_ID, CHOSEN_PIC_VAL: TRIAL_STIM_VAL}, inplace=True)
        trial_data.loc[:, TRIAL_NUMBER] = trial_data[TRIAL_NUMBER].astype(int)
        trial_data.loc[:, TRIAL_STIM_VAL] = trial_data[TRIAL_STIM_VAL].astype(int)
        trial_data.loc[:, TRIAL_STIM_ID] = trial_data[TRIAL_STIM_ID].astype(int)
    else:  # read the stimulus order from a file that contains a list of all stimuli in the experiment in order
        stim_order_name = [f for f in os.listdir(sub_path) if FILE_STIM_ORDER in f][0]
        trial_data = pd.read_csv(os.path.join(sub_path, stim_order_name), sep="\t", header=None)
        trial_data.rename(columns={0: TRIAL_STIM_ID}, inplace=True)
        trial_data[TRIAL_NUMBER] = trial_data.index
        trial_data = trial_data[[TRIAL_NUMBER, TRIAL_STIM_ID]]  # rearrange columns to make sense
        trial_data[TRIAL_STIM_VAL] = trial_data.apply(lambda row: label_valence(row), axis=1)

    # load stimulus scrambled locations
    rand_stim_name = [f for f in os.listdir(sub_path) if FILE_RANDOMIZE_STIM in f][0]
    rand_stim_data = pd.read_csv(os.path.join(sub_path, rand_stim_name), sep="\t", header=None)
    scrambled_loc_data = rand_stim_data[rand_stim_data.apply(lambda row: row.astype(str).str.contains(SCRAMBLED_LOCS, case=False).any(), axis=1)]
    if not(scrambled_loc_data.empty):
        rand_stim_data = rand_stim_data[rand_stim_data[1] == SCRAMBLED_LOCS].reset_index(drop=True)
        trial_data[TRIAL_SCRAMBLED_LOCS] = rand_stim_data[[2]].copy()
    else:  # we need to read the "scrambeloo" files, as the FILE_RANDOMIZE_STIM documentation didn't work
        trial_scrambled_info = load_scrambeloo(sub_path)
        trial_data = pd.merge(trial_data, trial_scrambled_info, on=TRIAL_STIM_ID)

    # get the last intact billboard
    trial_data.loc[:, LAST_INTACT_LOC] = trial_data.apply(lambda row: last_intact_in_trial(row), axis=1)

    # load intact stimuli and their times: mark the TIME of the LAST INTACT STIMULUS appearing in the trial
    motion_pivot_file = [f for f in os.listdir(sub_path) if FILE_MOTION_PIVOT in f][0]
    motion_pivot = pd.read_csv(os.path.join(sub_path, motion_pivot_file), sep="\t", header=None, names=[i for i in range(13)])
    motion_pivot = motion_pivot[motion_pivot[1] == MOTION_BILLBOARD].drop(columns=[i for i in range(4, 13)])  # only rows indicating passing a billboard
    if not motion_pivot.empty:
        motion_pivot = mark_motion_pivot_trials(motion_pivot)
        trial_data = mark_last_busstop_time(trial_data, motion_pivot)
    else:
        trial_data.loc[:, LAST_INTACT_TIME] = np.nan

    return trial_data


def enumerate_trials(df):
    start_flag = False
    end_flag = False
    curr_trial = -1
    df[3] = 0
    for ind, row in df.iterrows():
        if row[0] == START_TIME:
            start_flag = True
            end_flag = False
            curr_trial += 1
            df.at[ind, 3] = curr_trial
        elif row[0] == END_TIME:
            end_flag = True
            start_flag = False
            df.at[ind, 3] = curr_trial
        else:
            if start_flag and not (end_flag):
                df.at[ind, 3] = curr_trial
    return df


def helper_mark_trial(df):
    df = enumerate_trials(df)  # mark trial numbers
    if SCORE_START in df[1].unique():
        return df  # skip the following check below, as the file to be checked below is an ET file and this is not it.
    else:  # sanity check for the file, as subject 393928 had a couple of interruptions during the game which messed up the ET file a bit
        for trial in df[3].unique():
            trial_data = df[df[3] == trial]
            if (GAZE_START in trial_data[1].unique() and GAZE_END not in trial_data[1].unique()) \
                    or (GAZE_START not in trial_data[1].unique() and GAZE_END in trial_data[1].unique()):
                # there are start times without end times // end times without start times - this trial is interrupted hence should be dropped
                df.iloc[list(np.where(df[3] == trial)), 3] = -1  # invalidate this trial
        df = df.drop(df[df[3] == -1].index, inplace=False).reset_index(drop=True, inplace=False)
        df = enumerate_trials(df)  # repeat w/o the partial (interrupted) trials
    return df


def trial_to_score_data(score_df):
    # add a column denoting which trial number it is
    score_df = helper_mark_trial(score_df)

    # count clues and gain/loss
    # get the first trial's (score_df[3] == 0) start money (score_df[1] == SCORE_START) from column 2
    curr_score = score_df[(score_df[3] == 0) & (score_df[1] == SCORE_START)].reset_index(drop=True).loc[0, 2]
    score_df = score_df[score_df[1] == SCORE_REPORT]
    num_clues = list()
    bee_score = list()
    trial_score = list()

    for trial in score_df[3].unique():
        trial_score_change = score_df[score_df[3] == trial]
        last_score = trial_score_change.iloc[-1, 2]  # the last/only score-update row is about answering the bee
        # check clues and score in bee
        if trial_score_change.shape[0] > 1:  # if there is more than 1 score-update in a trial
            # this means that clues were taken during the trial.
            trial_num_clues = trial_score_change.shape[0] - 1
            trial_bee_score = last_score - trial_score_change.iloc[-2, 2]  # correct/incorrect in bee
        else:
            trial_num_clues = 0
            trial_bee_score = last_score - curr_score
        num_clues.append(trial_num_clues)
        bee_score.append(trial_bee_score)
        trial_score.append(last_score)
        curr_score = last_score

    trial_scores = pd.DataFrame.from_dict({TRIAL_NUMBER: [i for i in range(len(score_df[3].unique()))],
                                           CLUES_TAKEN: num_clues, BEE_ANS: bee_score,
                                           TRIAL_MONEY: trial_score})
    return trial_scores


def label_score(row, col=BEE_CORRECT):
    if row[col] == True:
        return BEE_VALUE
    else:
        return -BEE_VALUE


def trial_to_score_data_alternative(score_df):
    score_df[TRIAL_NUMBER] = score_df.index
    score_df.rename(columns={"hints": CLUES_TAKEN, "score": TRIAL_MONEY, "correct": BEE_CORRECT}, inplace=True)
    score_df[BEE_ANS] = score_df.apply(lambda row: label_score(row), axis=1)
    score_df = score_df[[TRIAL_NUMBER, CLUES_TAKEN, BEE_ANS, TRIAL_MONEY, BEE_CORRECT]]
    return score_df


def load_trial_bee_info(sub_path, trial_df):
    # score
    score_calc_name = [f for f in os.listdir(sub_path) if FILE_SCORE_CALC in f][0]
    score_calc_data = pd.read_csv(os.path.join(sub_path, score_calc_name), sep="\t", header=None)
    trial_score_data = score_calc_data[score_calc_data.apply(lambda row: row.astype(str).str.contains(SCORE_START, case=False).any(), axis=1)]
    if trial_score_data.shape[0] > 1:
        score_calc_data = trial_to_score_data(score_calc_data)
    else:
        score_calc_name = [f for f in os.listdir(sub_path) if FILE_SCORE_CALC_ALTERNATIVE in f][0]
        score_calc_data = pd.read_csv(os.path.join(sub_path, score_calc_name), sep="\t", header=None)
        score_calc_data.rename(columns={1: score_calc_data.iloc[0, 0], 3: score_calc_data.iloc[0, 2], 5: score_calc_data.iloc[0, 4]}, inplace=True)
        score_calc_data.drop(columns=[0, 2, 4], inplace=True)
        score_calc_data = trial_to_score_data_alternative(score_calc_data)
    trial_df = pd.merge(trial_df, score_calc_data, on=TRIAL_NUMBER)

    # information from target bee
    target_bee_files = [f for f in os.listdir(sub_path) if FILE_TARGET_BEE in f]
    if BEE_CORRECT not in trial_df.columns:  # if we didn't already achieve this information
        bee_correct = pd.read_csv(os.path.join(sub_path, [f for f in target_bee_files if BEE_CORRECT in f][0]), sep="\t", header=None, names=[i for i in range(4)])
        bee_correct = bee_correct[bee_correct[2] == BEE_CORRECT].reset_index(drop=True)
        trial_df.loc[:, BEE_CORRECT] = bee_correct[3]

    bee_selected_loc = pd.read_csv(os.path.join(sub_path, [f for f in target_bee_files if SELECTED_LOC in f][0]), sep="\t", header=None, names=[i for i in range(4)])
    target_loc_data = bee_selected_loc[bee_selected_loc.apply(lambda row: row.astype(str).str.contains(SELECTED_LOC, case=False).any(), axis=1)]
    if target_loc_data.shape[0] >= NUM_TRIALS:
        bee_selected_loc = bee_selected_loc[bee_selected_loc[2] == SELECTED_LOC].reset_index(drop=True)
        trial_df.loc[:, BEE_SELECT_LOC] = bee_selected_loc[3]
    else:
        trial_df.loc[:, BEE_SELECT_LOC] = None  # can be extracted from bee location files per trial; currently has no use
    return trial_df


def get_obj_correct(correct_loc, selected_loc, is_correct):
    # remain only with the actual response lines
    correct_loc = correct_loc[correct_loc[1] == IN].reset_index(drop=True)
    selected_loc = selected_loc[selected_loc[1] == OUT].reset_index(drop=True)
    is_correct = is_correct[is_correct[1] == OUT].reset_index(drop=True)
    comparison_df = pd.DataFrame.from_dict({"ans": correct_loc[3], "chose": selected_loc[3], "isCorrect?": is_correct[3]})
    comparison_df.loc[:, "isCorrectComparison"] = np.where((comparison_df['ans'] == comparison_df['chose']), True, False)
    comparison_df.loc[:, "FalseForMistake"] = np.where((comparison_df['isCorrect?'] == comparison_df['isCorrectComparison']), True, False)
    if not all(comparison_df["FalseForMistake"]):
        print("OBJ Q ERROR: INCONSISTENCY BETWEEN LOGS REGARDING SUBJECT'S OBJECTIVE Q ANSWER")
        return None
    else:  # return both whether correct, and the TIMESTAMP of the answer!
        if comparison_df.shape[0] < NUM_TRIALS:
            missing_trials = NUM_TRIALS - comparison_df.shape[0]
            new_comparison_df = comparison_df.append([[] for _ in range(missing_trials)], ignore_index=True)
            comparison_df = new_comparison_df
        comp_result = comparison_df["isCorrectComparison"]
        return comp_result


def get_obj_time_loc(obj_correct_location, obj_q_timings):
    if obj_q_timings is not None:
        start_time_rows = obj_q_timings[obj_q_timings[1] == OBJ_Q_START]
        obj_start_times = [dt.datetime.strptime(t, '%Y-%m-%d %H:%M:%S.%f') for t in start_time_rows[2].tolist()]
        ans_time_rows = obj_q_timings[obj_q_timings[1] == OBJ_Q_SELECTED]
        obj_selected_times = [dt.datetime.strptime(t, '%Y-%m-%d %H:%M:%S.%f') for t in ans_time_rows[2].tolist()]
    else:
        obj_start_times = np.nan
        obj_selected_times  = np.nan

    # remain only with the TARGET location rows
    target_rows = obj_correct_location[obj_correct_location[5] == TARGET]
    if target_rows.shape[0] < NUM_TRIALS:
        print(f"Objective Q logs: missing trials = {NUM_TRIALS - target_rows.shape[0]}")
        missing_trials = NUM_TRIALS - target_rows.shape[0]
        target_rows = target_rows.append([[] for _ in range(missing_trials)], ignore_index=True)
        obj_start_times.extend([np.nan for _ in range(missing_trials)])
        obj_selected_times.extend([np.nan for _ in range(missing_trials)])

    obj_picname = [int(x) if not(pd.isna(x)) else x for x in target_rows[4].tolist()]
    obj_locs = target_rows[2].tolist()
    return obj_start_times, obj_selected_times, obj_picname, obj_locs


def load_trial_objQ_info(sub_path, trial_df):
    objQ_files = [f for f in os.listdir(sub_path) if OBJ_Q in f]
    correct_ans_loc = pd.read_csv(os.path.join(sub_path, [f for f in objQ_files if OBJ_CORRECT_LOC in f][0]), sep="\t", header=None, names=[i for i in range(4)])
    selected_ans_loc = pd.read_csv(os.path.join(sub_path, [f for f in objQ_files if OBJ_SELECTED_LOC in f][0]), sep="\t", header=None, names=[i for i in range(4)])
    is_correct = pd.read_csv(os.path.join(sub_path, [f for f in objQ_files if OBJ_IS_CORRECT in f][0]), sep="\t", header=None, names=[i for i in range(4)])
    obj_correct_location = pd.read_csv(os.path.join(sub_path, [f for f in objQ_files if OBJ_Q_RANDOMIZE_OPTIONS in f][0]), sep="\t", header=None, names=[i for i in range(6)])
    # timings file: version 22-07-30 onward
    if len([f for f in objQ_files if OBJ_TIMINGS in f]) > 0:
        obj_q_timings = pd.read_csv(os.path.join(sub_path, [f for f in objQ_files if OBJ_TIMINGS in f][0]), sep="\t", header=None, names=[i for i in range(3)])
    else:
        obj_q_timings = None
    trial_df[OBJ_TIME], trial_df[OBJ_TIME_SELECTED], trial_df[TRIAL_STIM_NAME], trial_df[OBJ_TARGET_LOC] = get_obj_time_loc(obj_correct_location, obj_q_timings)
    trial_df.loc[:, OBJ_RT] = trial_df[OBJ_TIME_SELECTED] - trial_df[OBJ_TIME]
    trial_df.loc[:, OBJ_BUSSTOP] = trial_df[OBJ_TIME] - trial_df[LAST_INTACT_TIME]  # time between the appearance of the last intact instance of the stimulus, and the objective question
    trial_df.loc[:, LAST_INTACT_TIME] = pd.to_datetime(trial_df[LAST_INTACT_TIME])  # convert "nan"s to "NaT"s
    trial_df.loc[:, OBJ_ANS] = get_obj_correct(correct_ans_loc, selected_ans_loc, is_correct)

    trial_df.loc[:, OBJ_RT] = [t.total_seconds() if not (pd.isna(t)) else t for t in trial_df[OBJ_RT].tolist()]  # get diff in SECONDS
    trial_df.loc[:, OBJ_BUSSTOP] = [t.total_seconds() if not (pd.isna(t)) else t for t in trial_df[OBJ_BUSSTOP].tolist()]
    for col in [OBJ_TIME, OBJ_TIME_SELECTED]:
        trial_df.loc[:, col] = [t.time() if not (pd.isna(t)) else t for t in trial_df[col].tolist()]
    return trial_df


def load_trial_subjQ_info(sub_path, trial_df):
    """
    **Both** PAS and subject's valence judgement of the image (negative / not negative)
    :param sub_path:
    :param trial_df:
    :return:
    """
    # PAS
    subjQ_file = [f for f in os.listdir(sub_path) if SUBJ_Q in f][0]
    subjQ_data = pd.read_csv(os.path.join(sub_path, subjQ_file), sep="\t", header=None, names=[i for i in range(4)])
    # timings
    if len([f for f in os.listdir(sub_path) if SUBJ_TIMINGS in f]) > 0:
        subjQ_time_files = [f for f in os.listdir(sub_path) if SUBJ_TIMINGS in f][0]
        subjQ_time_data = pd.read_csv(os.path.join(sub_path, subjQ_time_files), sep="\t", header=None, names=[i for i in range(3)])
        start_time_rows = subjQ_time_data[subjQ_time_data[1] == SUBJ_Q_START]
        subj_start_times = [dt.datetime.strptime(t, '%Y-%m-%d %H:%M:%S.%f') for t in start_time_rows[2].tolist()]
        ans_time_rows = subjQ_time_data[subjQ_time_data[1] == SUBJ_Q_SELECTED]
        subj_selected_times = [dt.datetime.strptime(t, '%Y-%m-%d %H:%M:%S.%f') for t in ans_time_rows[2].tolist()]
        if start_time_rows.shape[0] < NUM_TRIALS:
            print(f"Subjective Q logs: missing trials = {NUM_TRIALS - start_time_rows.shape[0]}")
            missing_trials = NUM_TRIALS - start_time_rows.shape[0]
            subj_start_times.extend([np.nan for _ in range(missing_trials)])
            subj_selected_times.extend([np.nan for _ in range(missing_trials)])
    else:
        subj_start_times = np.nan
        subj_selected_times = np.nan

    trial_df[SUBJ_TIME], trial_df[SUBJ_TIME_SELECTED] = subj_start_times, subj_selected_times
    if pd.isnull(trial_df[SUBJ_TIME]).all():  # no time data
        trial_df.loc[:, SUBJ_TIME] = pd.to_datetime(trial_df[SUBJ_TIME])
        trial_df.loc[:, SUBJ_TIME_SELECTED] = pd.to_datetime(trial_df[SUBJ_TIME_SELECTED])
    trial_df.loc[:, SUBJ_RT] = trial_df[SUBJ_TIME_SELECTED] - trial_df[SUBJ_TIME]
    trial_df.loc[:, SUBJ_RT] = [t.total_seconds() if not (pd.isna(t)) else t for t in trial_df[SUBJ_RT].tolist()]  # get diff in SECONDS
    trial_df.loc[:, SUBJ_BUSSTOP] = trial_df[SUBJ_TIME] - trial_df[LAST_INTACT_TIME]  # time between the appearance of the last intact instance of the stimulus, and the subjective question
    trial_df.loc[:, SUBJ_BUSSTOP] = [t.total_seconds() if not (pd.isna(t)) else t for t in trial_df[SUBJ_BUSSTOP].tolist()]
    for col in [SUBJ_TIME, SUBJ_TIME_SELECTED]:
        trial_df.loc[:, col] = [t.time() if not (pd.isna(t)) else t for t in trial_df[col].tolist()]

    # just the responses
    subjQ_data = subjQ_data[subjQ_data[2] == SUBJ_SELECTED_LOC].reset_index(drop=True)
    subjQ_list = [int(x)+1 for x in subjQ_data[3].tolist()]  # the +1 is to convert the ratings coded in [0, 3] to PAS [1, 4]
    trial_df.loc[:, SUBJ_ANS] = pd.Series(subjQ_list)  # add as a column, fill with nans if missing trials

    # valence
    valQ_file = [f for f in os.listdir(sub_path) if VALENCE_Q in f][0]
    valQ_data = pd.read_csv(os.path.join(sub_path, valQ_file), sep="\t", header=None, names=[i for i in range(4)])
    if valQ_data.shape[0] != 0:  # ONLY if we even collected this data; otherwise, the valenceQ was skipped (2022-07-30 build) and there's nothing here
        valQ_data = valQ_data[valQ_data[2] == SELECTED_LOC].reset_index(drop=True)
        trial_df.loc[:, VAL_ANS] = pd.Series(valQ_data[3].astype(int))
        """
        IMPORTANT: IN THE ACTUAL VALENCE ANSWER, 0 AND 1 ARE THE *LOCATIONS* OF THE CHOSEN ANSWER.
        THE *LEFT* DISTRACTOR, 0, IS "YES - THE PICTURE WAS NEGATIVE". AND THE *RIGHT* DISTRACTOR, I.E, 1, IS 
        "NO, THE IMAGE WAS NOT NEGATIVE". MEANING, THAT TO MATCH VAL_ANS WITH TRIAL_STIM_VAL (WHERE 1=AVERSIVE & 0=NEUTRAL)
        WE NEED TO ***FLIP*** THE VAL_ANS COLUMN.
        """
        trial_df.replace({VAL_ANS: {1: 0, 0: 1}}, inplace=True)
        correct_conditions = [(pd.isna(trial_df[VAL_ANS]) & pd.isna(trial_df[TRIAL_STIM_VAL])),
                              trial_df[VAL_ANS] == trial_df[TRIAL_STIM_VAL],
                              trial_df[VAL_ANS] != trial_df[TRIAL_STIM_VAL]]
        correct_options = [np.nan, True, False]
        trial_df[VAL_ANS_CORRECT] = np.select(correct_conditions, correct_options, default=np.nan)
    else:
        trial_df.loc[:, VAL_ANS] = np.nan
        trial_df.loc[:, VAL_ANS_CORRECT] = np.nan
    return trial_df


def helper_calculate_durs(df):
    trial_df_list = list()
    for trial in df[3].unique():
        trial_data = df[df[3] == trial]
        new_df = pd.DataFrame()
        for ind, row in trial_data.iterrows():
            if row[0] == START_TIME:
                new_df.at[ind, 0] = trial_data.loc[ind, 0]
                new_df.at[ind, 1] = trial_data.loc[ind, 1]
                new_df.at[ind, 2] = trial_data.loc[ind, 2]
                new_df.at[ind, 3] = trial_data.loc[ind, 3]
            elif row[0] == END_TIME:
                new_df.at[ind, :] = trial_data.loc[ind, :]
            elif row[1] == GAZE_START:
                """
                We would like to get the gaze start and end stamps to get the total gaze duration. 
                The "start" and "end" time are coded in the bus stop log files as Unity "Time.time()".
                This is the time (in SECONDS) from the start of the application (i.e., experiment). 
                BUT -  Unity "Time.time()" IS **NOT** RELIABLE. When looking at the data, there are timestamps jumping
                in time. 
                THIS IS WHY instead of using the start/end message strings (i.e., their content, the Time.time() stamp)
                WE USE THE MESSAGE'S TIMESTAMP! Which is based on C#'s DateTime.Now property.  
                (i.e., instead of using column #2 with the message's content (time.time), we use the message's timestamp
                which is column #0). 
                """
                start = float(trial_data.loc[ind, 0])
                end = float(trial_data.loc[ind + 1, 0])
                new_df.at[ind, 0] = trial_data.loc[ind+1, 0]
                new_df.at[ind, 1] = GAZE_DURATION
                new_df.at[ind, 2] = (end - start) / 1000  # the difference between the stamps is in MILLISECONDS, / 1000 converts to seconds
                new_df.at[ind, 3] = trial
            elif row[1] == GAZE_END:
                continue
        trial_df_list.append(new_df)
    result_df = pd.concat(trial_df_list).reset_index(drop=True)
    return result_df


def load_trial_busstop_gaze_duration(sub_path, trial_df):
    # for each of the 10 busstops, load them and add trial information
    image_record_file = [f for f in os.listdir(sub_path) if IMAGE_RECORD in f and IMAGE in f]
    image_backup_files = [f for f in os.listdir(sub_path) if IMAGE_RECORD_BACKUP in f]
    missing_flag = 0
    i = 0

    for busstop_ind in range(BUSSTOP_FIRST, BUSSTOP_LAST+1):
        busstop_name = IMAGE + str(busstop_ind)
        try:  # read the bus stop's recording of gaze during the trials
            busstop_data = pd.read_csv(os.path.join(sub_path, [f for f in image_record_file if busstop_name in f][0]), sep="\t", header=None)
        except Exception:
            busstop_data = pd.DataFrame()  # empty dataframe in case file doesn't exist / completely blank
        # Bus stop data log was corrupted - is there a backup?
        if busstop_data.shape[0] <= 2:
            try:
                busstop_data_backup = pd.read_csv(os.path.join(sub_path, [f for f in image_backup_files if busstop_name in f][0]), sep="\t", header=None)
                if busstop_data_backup.shape[0] > 2:  # read the backup file
                    busstop_data_backup = busstop_data_backup[[3, 5]]
                    busstop_data_backup.rename(columns={3: BUSSTOP_GAZE_DUR + str(i), 5: BUSSTOP_GAZE_NUM + str(i)}, inplace=True)
                    busstop_data_backup[TRIAL_NUMBER] = list(range(0, busstop_data_backup.shape[0]))
                    trial_df = pd.merge(trial_df, busstop_data_backup, on=TRIAL_NUMBER)
                else:  # the file is empty
                    missing_flag = 1  # mark that the raw gaze data was lost
                    trial_df[BUSSTOP_GAZE_DUR + str(i)] = np.nan
                    trial_df[BUSSTOP_GAZE_NUM + str(i)] = np.nan

            except Exception:  # backup didn't work
                missing_flag = 1  # mark that the raw gaze data was lost
                trial_df[BUSSTOP_GAZE_DUR + str(i)] = np.nan
                trial_df[BUSSTOP_GAZE_NUM + str(i)] = np.nan

        else:  # Bus stop data was fine, no backup needed
            busstop_data = helper_mark_trial(busstop_data)
            busstop_start_and_end = busstop_data[busstop_data[1] != GAZE_DURATION]  # either start or end
            busstop_gaze_duration = helper_calculate_durs(busstop_start_and_end)
            busstop_gaze_duration = busstop_gaze_duration[busstop_gaze_duration[1] == GAZE_DURATION]
            gaze_duration_sum_per_trial = busstop_gaze_duration.groupby([3]).sum()
            gaze_periods_per_trial = busstop_gaze_duration.groupby([3]).count()
            trial_df[BUSSTOP_GAZE_DUR+str(i)] = gaze_duration_sum_per_trial[2]
            trial_df[BUSSTOP_GAZE_NUM + str(i)] = gaze_periods_per_trial[2]
        i += 1

    if trial_df.loc[:, [BUSSTOP_GAZE_DUR + str(i) for i in range(i)]].isnull().all().all():  # ET data recoded, but no eyes were in helmet
        missing_flag = 1

    return trial_df, missing_flag


def trial_busstop_gaze_summary(trial_df):
    """
    Add summary (avg, std, min, max) columns for gaze duration & number of gaze periods (in trial), based on
    the given dataframe, which contains columns per bus stop (gaze duration on the bus stop & # of periods).
    Each row = trial, and so to create a summary column per trial, the intact/scrambled/all bus stops of that trial
    are accumulated.
    :param trial_df: row=trial, column=data, at this point the df has 10*2 gaze columns (10 bus stops, gaze duration,
    # of periods).
    :return: trial_df with the gaze summary columns
    """
    all_busstop_dur_cols = [BUSSTOP_GAZE_DUR+str(i) for i in range(BUSSTOP_LAST-BUSSTOP_FIRST+1)]
    trial_df[BUSSTOP_GAZE_DUR_TOTAL] = trial_df[all_busstop_dur_cols].sum(axis=1)
    trial_df[BUSSTOP_GAZE_DUR_AVG_TOTAL] = trial_df[all_busstop_dur_cols].mean(axis=1)
    trial_df[BUSSTOP_GAZE_DUR_STD_TOTAL] = trial_df[all_busstop_dur_cols].std(axis=1)

    for ind, row in trial_df.iterrows():
        scrambled = row[TRIAL_SCRAMBLED_LOCS].split(";")
        intact = [str(i) for i in range(BUSSTOP_LAST-BUSSTOP_FIRST+1) if str(i) not in scrambled]
        intact_dur_cols = [b for b in all_busstop_dur_cols for i in intact if i in b]
        scrambled_dur_cols = [b for b in all_busstop_dur_cols if b not in intact_dur_cols]
        # stats
        trial_df.at[ind, BUSSTOP_GAZE_DUR_INTACT] = row[intact_dur_cols].sum()
        trial_df.at[ind, BUSSTOP_GAZE_DUR_AVG_INTACT] = row[intact_dur_cols].mean()
        trial_df.at[ind, BUSSTOP_GAZE_DUR_STD_INTACT] = row[intact_dur_cols].std()
        trial_df.at[ind, BUSSTOP_GAZE_DUR_MIN_INTACT] = row[intact_dur_cols].min()
        trial_df.at[ind, BUSSTOP_GAZE_DUR_MAX_INTACT] = row[intact_dur_cols].max()
        trial_df.at[ind, BUSSTOP_GAZE_DUR_SCRAMBLED] = row[scrambled_dur_cols].sum()
        trial_df.at[ind, BUSSTOP_GAZE_DUR_AVG_SCRAMBLED] = row[scrambled_dur_cols].mean()
        trial_df.at[ind, BUSSTOP_GAZE_DUR_STD_SCRAMBLED] = row[scrambled_dur_cols].std()
        trial_df.at[ind, BUSSTOP_GAZE_DUR_MIN_SCRAMBLED] = row[scrambled_dur_cols].min()
        trial_df.at[ind, BUSSTOP_GAZE_DUR_MAX_SCRAMBLED] = row[scrambled_dur_cols].max()
    return trial_df


def clean_replay(trial_df):
    """
    Clear out all the data from the bee-related columns in rows of replay-level trials. This is because during those
    levels, subjects were instructed not to care at all for the bee task (ignore) and focus only on busstops.
    """
    bee_cols = [CLUES_TAKEN, BEE_ANS, BEE_CORRECT, BEE_SELECT_LOC, TRIAL_MONEY]
    trial_df.at[REPLAY_LEVELS, bee_cols] = np.nan
    return trial_df


def get_trial_times(sub_path):
    """
    From the BusBeesAndPlayer.BusMotion output log, take the TRIAL_START_TIME timestamp as the beginning of the trial
    :param sub_path: path to subject data
    :return: df with trial number and trial start time
    """
    bus_motion_file = [f for f in os.listdir(sub_path) if FILE_BUS_MOTION in f][0]
    bus_data = pd.read_csv(os.path.join(sub_path, bus_motion_file), sep="\t", header=None)
    if bus_data.shape[0] < 2:  # bus motion data file exists but empty
        print("no bus motion data; cannot derive trial start and end times")
        return None
    # extract the real times each trial started in
    trial_start_data_time = bus_data.loc[bus_data[1] == TRIAL_START_TIME]
    trial_end_data_time = bus_data.loc[bus_data[1] == TRIAL_END_TIME]
    if not trial_start_data_time.empty:  # when we have the TRIAL_START_TIME marker in this file (build 22-04 onward)
        # trial start
        trial_start_data = trial_start_data_time[[2]]
        trial_start_data.rename(columns={2: TRIAL_START_TIME}, inplace=True)
        trial_start_data.loc[:, TRIAL_NUMBER] = list(range(0, trial_start_data.shape[0]))
        trial_start_data.reset_index(drop=True, inplace=True)
        trial_start_data.loc[:, TRIAL_START_TIME] = [dt.datetime.strptime(t, '%Y-%m-%d %H:%M:%S.%f') for t in trial_start_data[TRIAL_START_TIME].tolist()]
        trial_start_data.loc[:, DATE] = [d.date() for d in trial_start_data[TRIAL_START_TIME].tolist()]
        trial_start_data.loc[:, TRIAL_START] = [d.time() for d in trial_start_data[TRIAL_START_TIME].tolist()]
        # trial end: from build 22-07-30 onward
        trial_end_data = trial_end_data_time[[2]]
        trial_end_data.rename(columns={2: TRIAL_END_TIME}, inplace=True)
        trial_end_data.loc[:, TRIAL_NUMBER] = list(range(0, trial_end_data.shape[0]))
        trial_end_data.reset_index(drop=True, inplace=True)
        trial_end_data.loc[:, TRIAL_END_TIME] = [dt.datetime.strptime(t, '%Y-%m-%d %H:%M:%S.%f') for t in trial_end_data[TRIAL_END_TIME].tolist()]
        trial_end_data.loc[:, TRIAL_END] = [d.time() for d in trial_end_data[TRIAL_END_TIME].tolist()]

    else:
        start_datetime_str = bus_data[bus_data[0] == START_TIME][1].tolist()[0]
        start_datetime = dt.datetime.strptime(start_datetime_str, '%Y-%m-%d %H:%M:%S.%f')
        start_date = start_datetime.date()
        num_trials = bus_data[bus_data[0] == START_TIME].shape[0]-1
        trial_start_data = pd.DataFrame({TRIAL_NUMBER: list(range(0, num_trials)), TRIAL_START_TIME: [np.nan] * num_trials, DATE: [start_date] * num_trials, TRIAL_START: [np.nan] * num_trials})
        trial_end_data = None

    return trial_start_data, trial_end_data


def parse_busstop_gaze(trial_data, sub_path):
    """
    Create a separate dataframe where row = GAZE (in some trial, on some bus stop). In this dataframe,
    each gaze period (start, end) has its own row, detailing the beginning, end of gaze, the bus stop that
    was gazed, the trial number, and whether the bus stop was intact in that trial (false=scrambled)
    :param trial_data: the trial dataframe containing information about the scrambled bus stops in each trial
    :param sub_path: the path of the subject data
    :return: df where row=GAZE EVENT, and cols are: bus stop, trial number, start gaze time, end gaze time, is intact.
    THIS IS A PER-SUBJECT DF
    """
    # for each of the 10 busstops, load them and add trial information
    image_record_file = [f for f in os.listdir(sub_path) if IMAGE_RECORD in f and IMAGE in f]

    image_gaze_df = pd.DataFrame(columns=[BUSSTOP, TRIAL_NUMBER, GAZE_START, GAZE_END, IS_INTACT])
    i = 0
    for busstop_ind in range(BUSSTOP_FIRST, BUSSTOP_LAST + 1):
        busstop_name = IMAGE + str(busstop_ind)  # bus stops are named 2-11, this is useful for file parsing
        busstop_number = busstop_ind - BUSSTOP_FIRST  # but bus stops are numbered 0-9
        # here I don't do try-except as if we were to not have this data we would have stopped before arriving here
        busstop_data = pd.read_csv(os.path.join(sub_path, [f for f in image_record_file if busstop_name in f][0]),sep="\t", header=None)
        busstop_data = helper_mark_trial(busstop_data)
        busstop_start_and_end = busstop_data[busstop_data[1] != GAZE_DURATION]  # either start or end, gaze_duration is NOT RELIABLE
        time_start = dt.datetime.strptime(busstop_start_and_end.iloc[0, 1], '%Y-%m-%d %H:%M:%S.%f')  # this is the start time of the ENTIRE EXPERIMENT (i.e., Unity start time)

        trial_numbers_scrambled = set(trial_data[trial_data[TRIAL_SCRAMBLED_LOCS].str.contains(f"{busstop_number}")][TRIAL_NUMBER].tolist())

        for ind, row in busstop_start_and_end.iterrows():
            if row[1] == GAZE_START:
                """
                The "start" and "end" time are coded in the bus stop log files as Unity "Time.time()".
                This is the time (in SECONDS) from the start of the application (i.e., experiment). 
                BUT -  Unity "Time.time()" IS **NOT** RELIABLE. When looking at the data, there are timestamps jumping
                in time. 
                THIS IS WHY instead of using the start/end message strings (i.e., their content, the Time.time() stamp)
                WE USE THE MESSAGE'S TIMESTAMP! Which is based on C#'s DateTime.Now property. 
                I.e. we rely on the message's time, and not content. 
                Then, we turn these stamps into time, so we can have a timeline to compare with other events in the experiment. 
                """
                start_from_beginning = dt.timedelta(milliseconds=float(busstop_start_and_end.loc[ind, 0]))
                end_from_beginning = dt.timedelta(milliseconds=float(busstop_start_and_end.loc[ind + 1, 0]))
                image_gaze_df.loc[i] = [busstop_number, row[3], (time_start + start_from_beginning).time(), (time_start + end_from_beginning).time(), not row[3] in trial_numbers_scrambled]
                i += 1
            elif row[1] == GAZE_END:
                continue
    return image_gaze_df


def load_sub_trial_data(sub_path):
    sub_unity_path = os.path.join(sub_path, UNITY_OUTPUT_FOLDER)
    trial_data = load_trial_stim_info(sub_unity_path)
    trial_data = load_trial_bee_info(sub_unity_path, trial_data)
    if trial_data.shape[0] < NUM_TRIALS:  # there are less than NUM_TRIALS in the trial_data table
        missing_trials = NUM_TRIALS - trial_data.shape[0]
        print(f"missing trials found: {missing_trials} trials are missing")
        trial_data = trial_data.append([[] for _ in range(missing_trials)], ignore_index=True)
    trial_data = load_trial_objQ_info(sub_unity_path, trial_data)
    trial_data = load_trial_subjQ_info(sub_unity_path, trial_data)
    # Eye tracking data
    trial_data, missing_et_data = load_trial_busstop_gaze_duration(sub_unity_path, trial_data)
    if missing_et_data == 0:
        trial_data = trial_busstop_gaze_summary(trial_data)
        busstop_gaze_data = parse_busstop_gaze(trial_data, sub_unity_path)  # generate a row-per-gaze-event dataframe
    else:
        print(f"Bus stop GAZE duration data lost / corrupt / missing - no data")
        busstop_gaze_data = None
    # in the replay levels, THE BEE DOES NOT MATTER, so all bee-related info should be NaN as subjects did not follow anything here
    trial_data = clean_replay(trial_data)
    # get the times of trial start (bus starts moving, after ET validation) and end (bus stops, subject needs to select target bee)
    trial_start_times, trial_end_times = get_trial_times(sub_unity_path)  # get trial start times
    trial_data = pd.merge(trial_data, trial_start_times, on=TRIAL_NUMBER)
    if trial_end_times is not None:
        if trial_end_times.shape[0] != 0:
            trial_data = pd.merge(trial_data, trial_end_times, on=TRIAL_NUMBER)
            trial_durs = trial_data.loc[:, TRIAL_END_TIME] - trial_data.loc[:, TRIAL_START_TIME]
            trial_data.loc[:, TRIAL_DUR_SEC] = [t.total_seconds() for t in trial_durs.tolist()]  # GET DIFF IN SECONDS
        else:
            trial_data.loc[:, [TRIAL_END_TIME, TRIAL_END, TRIAL_DUR_SEC, TRIAL_DUR_SEC]] = np.nan
    else:
        trial_data.loc[:, [TRIAL_END_TIME, TRIAL_END, TRIAL_DUR_SEC, TRIAL_DUR_SEC]] = np.nan

    # mark trials per their condition based on trial number
    trial_data.loc[trial_data[TRIAL_NUMBER] >= empatica_parser.REPLAY_TRIAL, CONDITION] = REPLAY
    trial_data.loc[trial_data[TRIAL_NUMBER] < empatica_parser.REPLAY_TRIAL, CONDITION] = GAME
    # mark trials' by the TYPE of their SUBJECTIVE answer : if == 1, "SUBJ_ANS_IS1" col=True, else false
    # same for SUBJ_ANS_IS12 (if subjective rating is 1 or 2)
    trial_data.loc[:, SUBJ_ANS_IS1] = np.where(trial_data[SUBJ_ANS] == 1, 1, 0)
    trial_data.loc[:, SUBJ_ANS_IS12] = np.where((trial_data[SUBJ_ANS] == 1) | (trial_data[SUBJ_ANS] == 2), 1, 0)
    # reorder columns
    new_order = [DATE, TRIAL_NUMBER, TRIAL_START_TIME, TRIAL_START, TRIAL_END_TIME, TRIAL_END, TRIAL_DUR_SEC,
                 CONDITION, TRIAL_STIM_ID, TRIAL_STIM_NAME, TRIAL_STIM_VAL, TRIAL_SCRAMBLED_LOCS, LAST_INTACT_LOC,
                 LAST_INTACT_TIME, CLUES_TAKEN, BEE_ANS, TRIAL_MONEY, BEE_CORRECT, BEE_SELECT_LOC,
                 OBJ_TARGET_LOC, OBJ_TIME, OBJ_TIME_SELECTED, OBJ_ANS, OBJ_RT, OBJ_BUSSTOP,
                 SUBJ_TIME, SUBJ_TIME_SELECTED, SUBJ_ANS, SUBJ_RT, SUBJ_BUSSTOP, SUBJ_ANS_IS1, SUBJ_ANS_IS12,
                 VAL_ANS, VAL_ANS_CORRECT]

    rest_of_cols = [c for c in trial_data.columns if c not in new_order]  # all the ET-bus stop columns
    new_order.extend(rest_of_cols)
    trial_data = trial_data[new_order]
    if trial_data[VAL_ANS].isnull().all():  # no valence Q asked (build 2022-07-30)
        trial_data.drop([VAL_ANS, VAL_ANS_CORRECT], axis=1, inplace=True)
    return trial_data, busstop_gaze_data


def time_add_ms(df, col):
    time_list = list()
    original_times = df[col].tolist()
    for t in original_times:
        try:
            time_list.append(dt.datetime.strptime(t, '%H:%M:%S.%f').time())
        except ValueError:
            ms = ".000000"
            t = t + ms
            time_list.append(dt.datetime.strptime(t, '%H:%M:%S.%f').time())
    return time_list


def convert_str_cols_to_datetime(df):
    """
    When reading dataframe from csv, date time information becomes a string and needs to be converted back to datetime
    :param df:
    :return:
    """
    missing_cols = list()
    try:
        df.loc[:, TRIAL_START_TIME] = [dt.datetime.strptime(t, '%Y-%m-%d %H:%M:%S.%f') for t in df[TRIAL_START_TIME].tolist()]
        df.loc[:, DATE] = [d.date() for d in df[TRIAL_START_TIME].tolist()]
    except TypeError:
        missing_cols.append(TRIAL_START_TIME)

    try:
        df.loc[:, TRIAL_END_TIME] = [dt.datetime.strptime(t, '%Y-%m-%d %H:%M:%S.%f') for t in df[TRIAL_END_TIME].tolist()]
        df.loc[:, TRIAL_END] = [d.time() for d in df[TRIAL_END_TIME].tolist()]
    except TypeError:
        missing_cols.append(TRIAL_END_TIME)

    try:
        df.loc[:, TRIAL_START] = [d.time() for d in df[TRIAL_START_TIME].tolist()]
    except AttributeError:
        missing_cols.append(TRIAL_START)

    try:
        df.loc[:, OBJ_TIME_SELECTED] = [dt.datetime.strptime(t, '%H:%M:%S.%f').time() for t in df[OBJ_TIME_SELECTED].tolist()]
    except TypeError:
        missing_cols.append(OBJ_TIME_SELECTED)
    except ValueError:  # there's one with a value which is exactly on the second (00ms), do it manually
        time_list = time_add_ms(df, OBJ_TIME_SELECTED)
        df.loc[:, OBJ_TIME_SELECTED] = time_list

    try:
        df.loc[:, OBJ_TIME] = [dt.datetime.strptime(t, '%H:%M:%S.%f').time() for t in df[OBJ_TIME].tolist()]
    except TypeError:
        missing_cols.append(OBJ_TIME)
    except ValueError:  # there's one with a value which is exactly on the second (00ms), do it manually
        time_list = time_add_ms(df, OBJ_TIME)
        df.loc[:, OBJ_TIME] = time_list

    try:
        df.loc[:, SUBJ_TIME_SELECTED] = [dt.datetime.strptime(t, '%H:%M:%S.%f').time() for t in df[SUBJ_TIME_SELECTED].tolist()]
    except TypeError:
        missing_cols.append(SUBJ_TIME_SELECTED)

    try:
        df.loc[:, SUBJ_TIME] = [dt.datetime.strptime(t, '%H:%M:%S.%f').time() for t in df[SUBJ_TIME].tolist()]
    except TypeError:
        missing_cols.append(SUBJ_TIME)
    except ValueError:  # there's one with a value which is exactly on the second (00ms), do it manually
        time_list = time_add_ms(df, SUBJ_TIME)
        df.loc[:, SUBJ_TIME] = time_list

    if len(missing_cols) > 0:
        print(f"columns [{missing_cols}] are empty")
    return df


def extract_subject_data(sub_folder_path, sub_res_path):
    # Open up data files for saving processing outputs
    persub_output_path = os.path.join(sub_res_path, PER_SUBJECT)
    if not (os.path.isdir(persub_output_path)):
        os.mkdir(persub_output_path)

    subjects = [s for s in os.listdir(sub_folder_path)]
    subject_data = dict()
    for sub in subjects:
        print(f"-----------PARSING SUBJECT {sub}-----------")
        # open a subject output folder
        sub_output_path = os.path.join(persub_output_path, sub)
        if not (os.path.isdir(sub_output_path)):
            os.mkdir(sub_output_path)
        # parse subject beh + et data
        sub_path = os.path.join(sub_folder_path, sub)
        sub_trial_table_path = os.path.join(sub_output_path, "sub_trial_data.csv")
        busstop_trial_table_path = os.path.join(sub_output_path, "sub_busstop_gaze_data.csv")
        if os.path.exists(sub_trial_table_path):
            # load trial data
            sub_trial_data = pd.read_csv(sub_trial_table_path)
            sub_trial_data = convert_str_cols_to_datetime(sub_trial_data)  # when reading, str needs to be converted to datetime
            # load bus stop gaze data
            if os.path.exists(busstop_trial_table_path):
                sub_busstop_gaze_data = pd.read_csv(busstop_trial_table_path)
                sub_busstop_gaze_data[GAZE_START] = [dt.datetime.strptime(t, '%H:%M:%S.%f').time() for t in sub_busstop_gaze_data[GAZE_START].tolist()]
                sub_busstop_gaze_data[GAZE_END] = [dt.datetime.strptime(t, '%H:%M:%S.%f').time() for t in sub_busstop_gaze_data[GAZE_END].tolist()]
            else:
                sub_busstop_gaze_data = None
        else:
            sub_trial_data, sub_busstop_gaze_data = load_sub_trial_data(sub_path)  # subject behavior and ET data table
            sub_trial_data.to_csv(os.path.join(sub_output_path, "sub_trial_data.csv"), index=False)
            if sub_busstop_gaze_data is not None:
                sub_busstop_gaze_data.to_csv(os.path.join(sub_output_path, "sub_busstop_gaze_data.csv"), index=False)
            else:
                print(f"No gaze data for subject {sub}")
        # parse subject empatica data
        sub_trial_data, sub_peripheral_data = empatica_parser.load_sub_peripheral_data(sub_path, sub_trial_data, sub_busstop_gaze_data, sub_output_path, sub, sub_res_path)
        subject_data[sub] = {UNITY_OUTPUT_FOLDER: sub_trial_data, empatica_parser.EMPATICA_OUTPUT_FOLDER: sub_peripheral_data, ET_DATA_NAME: sub_busstop_gaze_data}

    return subject_data
