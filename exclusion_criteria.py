from scipy.stats import binomtest
import parse_data_files


""" VRIB exclusion criteria

This module manages everything related to subject data exclusion, based on pre-registered requirements regarding
participant's behavior. The main method here is beh_exclusion, which is called by the analysis_manager module. 

@authors: RonyHirsch
"""


NUM_OF_TRIALS = 50
REPLAY_TRIAL = 40
OBJ_CHANCE_LEVEL = 0.25


def test_unattended_binom(sub, un_df):
    """
    Report the result of the binomial test on hit rate in the objective task in the PAS-1 rated trials in the unattended condition.
    """
    OBJ_CHANCE_LEVEL_PVAL = 0.05
    OBJ_CHANCE_LEVEL_CONFIDENCE = 1 - OBJ_CHANCE_LEVEL_PVAL
    pas_1 = un_df[un_df[parse_data_files.SUBJ_ANS] == 1]
    num_pas_1 = pas_1.shape[0]
    num_pas_1_obj_correct = pas_1[pas_1[parse_data_files.OBJ_ANS] == 1].shape[0]
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html#r37a6f66d8a8d-2
    binom_test = binomtest(k=num_pas_1_obj_correct, n=num_pas_1, p=OBJ_CHANCE_LEVEL, alternative='greater')
    binom_pval = binom_test.pvalue
    binom_ci = binom_test.proportion_ci(confidence_level=OBJ_CHANCE_LEVEL_CONFIDENCE)
    binom_pval_str = str(round(binom_pval, 3))
    binom_ci_str = f"[{str(round(binom_ci.low, 3))}, {str(round(binom_ci.high, 3))}]"
    print(f"SUBJECT {sub}: objective performance in PAS-1 is {num_pas_1_obj_correct} of {num_pas_1} trials ({round(100 * (num_pas_1_obj_correct / num_pas_1), 3)}%); p={binom_pval_str} >= {OBJ_CHANCE_LEVEL_PVAL}, CI={binom_ci_str}")
    return binom_pval < OBJ_CHANCE_LEVEL_PVAL


def include_subject(sub, beh_data):
    """
    This method marks subjects as included (1) or excluded (0) from analysis, based on the preregistered exclusion criteria.
    :param sub: subject code
    :param beh_data: subject data
    :return: 1 if subject can be included, 0 otherwise
    """
    # (1) DID THEY FAIL TO COMPLETE THE EXPERIMENT?
    if beh_data.shape[0] < NUM_OF_TRIALS:
        print(f"SUBJECT {sub} failed to complete the experiment; EXCLUDED")
        return 0

    # (2) DID THEY FAIL TO DETECT THE TARGET BEE IN 10/10 FIRST TRIALS?
    unattended_data = beh_data[beh_data[parse_data_files.TRIAL_NUMBER] < REPLAY_TRIAL]
    first_ten_trials = unattended_data.iloc[:10, :]
    bee_correct = first_ten_trials.loc[:, parse_data_files.BEE_CORRECT].tolist()
    if True not in bee_correct:
        print(f"SUBJECT {sub} chose the wrong bee in 10/10 first trials; EXCLUDED")
        return 0

    # (3) DID THEY FAIL TO SEE THE STIMULUS IN >= 50% OF THE ATTENDED TRIALS?
    attended_data = beh_data[beh_data[parse_data_files.TRIAL_NUMBER] >= REPLAY_TRIAL]
    subjective = attended_data.loc[:, parse_data_files.SUBJ_ANS].tolist()
    pas1 = subjective.count(1)
    if pas1 / len(subjective) >= 0.5:
        print(f"SUBJECT {sub} rated more than 50% of the trials in the attended condition as invisible; EXCLUDED")
        return 0
    # ALL IS WELL
    return 1


def beh_exclusion(sub_dict):
    """
    Preregistered exclusion criteria : https://osf.io/q6dxh date: September 11 2022
    :param sub_dict:
    :return: the same dictionary including only non-excluded subjects
    """
    raw_subs = list(sub_dict.keys())
    excluded_subs = list()
    result_dict = dict()
    for sub in raw_subs:
        beh_data = sub_dict[sub][parse_data_files.UNITY_OUTPUT_FOLDER]
        # CHECK THAT THE OBJECTIVE PERFORMANCE IN THE UNATTENDED PAS-1 TRIALS IS NOT ABOVE CHANCE LEVEL
        unattended_data = beh_data[beh_data[parse_data_files.TRIAL_NUMBER] < REPLAY_TRIAL]
        # report binomial test of IB phase for this subject
        test_unattended_binom(sub, unattended_data)
        # EXCLUDE SUBJECTS
        if include_subject(sub, beh_data) == 1:
            result_dict[sub] = sub_dict[sub]
        else:
            excluded_subs.append(sub)
    print(f"{len(excluded_subs)} subjects excluded due to behavior.")
    return result_dict

