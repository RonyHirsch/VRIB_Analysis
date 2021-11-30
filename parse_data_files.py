import os
import pandas as pd
import numpy as np


TRIAL_NUMBER = "trialNumber"
TRIAL_STIM_ID = "stimIndex"
TRIAL_STIM_VAL = "stimValence"
TRIAL_SCRAMBLED_LOCS = "scrambledLocations"
CLUES_TAKEN = "cluesTaken"
BEE_ANS = "beeScore"
BEE_CORRECT = "beeCorrect"
BEE_SELECT_LOC = "beeSelectedLocation"
TRIAL_MONEY = "trialScore"
OBJ_ANS = "objectiveIsCorrect"
SUBJ_ANS = "subjectiveAwareness"
VAL_ANS = "valenceJudgement"
BUSSTOP_GAZE_DUR = "busstopGazeDuration"
BUSSTOP_GAZE_NUM = "busstopGazeNumOfPeriods"
# file and file content data
FILE_RANDOMIZE_TRIALS = "Block.randomizeTrials_"
CHOSEN_PIC_ORDER = "chosenPicsOrder"
CHOSEN_PIC_VAL = "chosenPicsValence"
START_TIME = "startTime"
END_TIME = "endTime"
# scoreCalc
FILE_SCORE_CALC = "ScoreCalc.ScoreCalc_"
SCORE_GAIN = "PointsWhenCorrect"
SCORE_LOSS = "PointsWhenWrong"
SCORE_CLUE = "PointsWhenClue"
SCORE_REPORT = "CurrPoints"
SCORE_START = "startMoney"
# target bee
FILE_TARGET_BEE = "TagertBeeSelection.controllerProjection."
BEE_CORRECT = "isCorrectInBeeSelection"
SELECTED_LOC = "selectedTargetInd"
# randomizeStim (randomization of scrambled/intact locations)
FILE_RANDOMIZE_STIM = "ExpStimRandomizer.randomizeStim_"
SCRAMBLED_LOCS = "ScrambledLocationsInTrial"
# objective question
OBJ_Q = "ObjTrialAwareness.objectiveQProjection"
OBJ_CORRECT_LOC = ".correctAnsLoc"
OBJ_SELECTED_LOC = ".selectedTargetInd"
OBJ_IS_CORRECT = ".isCorrectAnswer"
IN = "IN"
OUT = "OUT"
# subjective question
SUBJ_Q = "SubjTrialAwareness.subjectiveQProjection.selectedTargetInd"
SUBJ_SELECTED_LOC = "selectedTargetInd"
# velance question
VALENCE_Q = "ValTrialAwareness.valenceQProjection.selectedTargetInd"
# ET data: images' recordAtGaze
IMAGE_RECORD = "Tobii.XR.Examples.RecordAtGaze"
IMAGE = "Image"
BUSSTOP_FIRST = 2
BUSSTOP_LAST = 11
GAZE_DURATION = "GazeDuration"


def load_trial_stim_info(sub_path):
    # load stimulus index and valence
    rand_trials_name = [f for f in os.listdir(sub_path) if FILE_RANDOMIZE_TRIALS in f][0]
    rand_trials_data = pd.read_csv(os.path.join(sub_path, rand_trials_name), sep="\t")
    rand_trials_data = rand_trials_data.drop(columns=[rand_trials_data.columns[0]])
    trial_data_dict = {TRIAL_NUMBER: [i for i in range(len(rand_trials_data.loc[0, rand_trials_data.columns[1]].split(";")))],
                       rand_trials_data.loc[0, rand_trials_data.columns[0]]: rand_trials_data.loc[0, rand_trials_data.columns[1]].split(";"),
                       rand_trials_data.loc[1, rand_trials_data.columns[0]]: rand_trials_data.loc[1, rand_trials_data.columns[1]].split(";")}
    trial_data = pd.DataFrame.from_dict(trial_data_dict)
    trial_data.rename(columns={CHOSEN_PIC_ORDER: TRIAL_STIM_ID, CHOSEN_PIC_VAL: TRIAL_STIM_VAL}, inplace=True)

    # load stimulus scrambled locations
    rand_stim_name = [f for f in os.listdir(sub_path) if FILE_RANDOMIZE_STIM in f][0]
    rand_stim_data = pd.read_csv(os.path.join(sub_path, rand_stim_name), sep="\t", header=None)
    rand_stim_data = rand_stim_data[rand_stim_data[1] == SCRAMBLED_LOCS].reset_index(drop=True)
    trial_data[TRIAL_SCRAMBLED_LOCS] = rand_stim_data[[2]].copy()
    return trial_data


def helper_mark_trial(df):
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


def trial_to_score_data(score_df):
    # add a column denoting which trial number it is
    score_df = helper_mark_trial(score_df)

    # count clues and gain/loss
    curr_score = score_df[(score_df[3] == 0) & (score_df[1] == SCORE_START)]  # get the first trial's start money
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


def load_trial_bee_info(sub_path, trial_df):
    # score
    score_calc_name = [f for f in os.listdir(sub_path) if FILE_SCORE_CALC in f][0]
    score_calc_data = pd.read_csv(os.path.join(sub_path, score_calc_name), sep="\t", header=None)
    score_calc_data = trial_to_score_data(score_calc_data)
    trial_df = pd.merge(trial_df, score_calc_data, on=TRIAL_NUMBER)
    # information from target bee
    target_bee_files = [f for f in os.listdir(sub_path) if FILE_TARGET_BEE in f]
    bee_correct = pd.read_csv(os.path.join(sub_path, [f for f in target_bee_files if BEE_CORRECT in f][0]), sep="\t", header=None, names=[i for i in range(4)])
    bee_correct = bee_correct[bee_correct[2] == BEE_CORRECT].reset_index(drop=True)
    bee_selected_loc = pd.read_csv(os.path.join(sub_path, [f for f in target_bee_files if SELECTED_LOC in f][0]), sep="\t", header=None, names=[i for i in range(4)])
    bee_selected_loc = bee_selected_loc[bee_selected_loc[2] == SELECTED_LOC].reset_index(drop=True)
    trial_df[BEE_CORRECT] = bee_correct[3]
    trial_df[BEE_SELECT_LOC] = bee_selected_loc[3]
    return trial_df


def get_obj_correct(correct_loc, selected_loc, is_correct):
    # remain only with the actual response lines
    correct_loc = correct_loc[correct_loc[1] == IN].reset_index(drop=True)
    selected_loc = selected_loc[selected_loc[1] == OUT].reset_index(drop=True)
    is_correct = is_correct[is_correct[1] == OUT].reset_index(drop=True)
    comparison_df = pd.DataFrame.from_dict({"ans": correct_loc[3], "chose": selected_loc[3], "isCorrect?": is_correct[3]})
    comparison_df["isCorrectComparison"] = np.where((comparison_df['ans'] == comparison_df['chose']), True, False)
    comparison_df["FalseForMistake"] = np.where((comparison_df['isCorrect?'] == comparison_df['isCorrectComparison']), True, False)
    if not all(comparison_df["FalseForMistake"]):
        print("OBJ Q ERROR: INCONSISTENCY BETWEEN LOGS REGARDING SUBJECT'S OBJECTIVE Q ANSWER")
        return None
    else:
        return comparison_df["isCorrectComparison"]


def load_trial_objQ_info(sub_path, trial_df):
    objQ_files = [f for f in os.listdir(sub_path) if OBJ_Q in f]
    correct_ans_loc = pd.read_csv(os.path.join(sub_path, [f for f in objQ_files if OBJ_CORRECT_LOC in f][0]), sep="\t", header=None, names=[i for i in range(4)])
    selected_ans_loc = pd.read_csv(os.path.join(sub_path, [f for f in objQ_files if OBJ_SELECTED_LOC in f][0]), sep="\t", header=None, names=[i for i in range(4)])
    is_correct = pd.read_csv(os.path.join(sub_path, [f for f in objQ_files if OBJ_IS_CORRECT in f][0]), sep="\t", header=None, names=[i for i in range(4)])
    trial_df[OBJ_ANS] = get_obj_correct(correct_ans_loc, selected_ans_loc, is_correct)
    return trial_df


def load_trial_subjQ_info(sub_path, trial_df):
    # PAS
    subjQ_file = [f for f in os.listdir(sub_path) if SUBJ_Q in f][0]
    subjQ_data = pd.read_csv(os.path.join(sub_path, subjQ_file), sep="\t", header=None, names=[i for i in range(4)])
    # just the responses
    subjQ_data = subjQ_data[subjQ_data[2] == SUBJ_SELECTED_LOC].reset_index(drop=True)
    trial_df[SUBJ_ANS] = subjQ_data[3]
    # valence
    valQ_file = [f for f in os.listdir(sub_path) if VALENCE_Q in f][0]
    valQ_data = pd.read_csv(os.path.join(sub_path, valQ_file), sep="\t", header=None, names=[i for i in range(4)])
    valQ_data = valQ_data[valQ_data[2] == SELECTED_LOC].reset_index(drop=True)
    trial_df[VAL_ANS] = valQ_data[3]
    return trial_df


def load_trial_busstop_gaze_duration(sub_path, trial_df):
    # for each of the 10 busstops, load them and add trial information
    image_record_file = [f for f in os.listdir(sub_path) if IMAGE_RECORD in f and IMAGE in f]
    i = 0
    for busstop_ind in range(BUSSTOP_FIRST, BUSSTOP_LAST+1):
        busstop_name = IMAGE + str(busstop_ind)
        busstop_data = pd.read_csv(os.path.join(sub_path, [f for f in image_record_file if busstop_name in f][0]), sep="\t", header=None)
        busstop_data = helper_mark_trial(busstop_data)
        busstop_gaze_duration = busstop_data[busstop_data[1] == GAZE_DURATION]
        gaze_duration_sum_per_trial = busstop_gaze_duration.groupby([3]).sum()
        gaze_periods_per_trial = busstop_gaze_duration.groupby([3]).count()
        trial_df[BUSSTOP_GAZE_DUR+str(i)] = gaze_duration_sum_per_trial[2]
        trial_df[BUSSTOP_GAZE_NUM + str(i)] = gaze_periods_per_trial[2]
        i += 1
    return trial_df


if __name__ == "__main__":
    sub_path = r"C:\Users\ronyhirschhorn\Documents\TAU\VR\VRIB_DATA\data\415122"
    trial_data = load_trial_stim_info(sub_path)
    trial_data = load_trial_bee_info(sub_path, trial_data)
    trial_data = load_trial_objQ_info(sub_path, trial_data)
    trial_data = load_trial_subjQ_info(sub_path, trial_data)
    trial_data = load_trial_busstop_gaze_duration(sub_path, trial_data)
    c = 4
