import pandas as pd
from scipy.stats import binomtest
import os
import re
import math
import numpy as np
import itertools
import seaborn as sns
import plotter
import parse_data_files
import gaze_analysis
import beh_analysis
import peripheral_analysis

REPLAY_TRIAL = 40
OBJ_CHANCE_LEVEL = 0.25
OBJ_CHANCE_LEVEL_PVAL = 0.05
OBJ_CHANCE_LEVEL_CONFIDENCE = 1 - OBJ_CHANCE_LEVEL_PVAL


def test_unattended(sub, un_df):
    pas_1 = un_df[un_df[parse_data_files.SUBJ_ANS] == 1]
    num_pas_1 = pas_1.shape[0]
    num_pas_1_obj_correct = pas_1[pas_1[parse_data_files.OBJ_ANS] == 1].shape[0]
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html#r37a6f66d8a8d-2
    binom_test = binomtest(k=num_pas_1_obj_correct, n=num_pas_1, p=OBJ_CHANCE_LEVEL, alternative='greater')
    binom_pval = binom_test.pvalue
    binom_ci = binom_test.proportion_ci(confidence_level=OBJ_CHANCE_LEVEL_CONFIDENCE)
    if binom_pval < OBJ_CHANCE_LEVEL_PVAL:
        binom_pval_str = str(round(binom_pval, 3))
        binom_ci_str = f"[{str(round(binom_ci.low, 3))}, {str(round(binom_ci.high, 3))}]"
        print(f"SUBJECT {sub} EXCLUDED DUE TO BEHAVIOR: objective performance in PAS-1 is {num_pas_1_obj_correct} of {num_pas_1} trials ({round(100*(num_pas_1_obj_correct/num_pas_1), 3)}%); p={binom_pval_str} < {OBJ_CHANCE_LEVEL_PVAL}, CI={binom_ci_str}")
        return 1
    return 0


def beh_exclusion(sub_dict):
    raw_subs = list(sub_dict.keys())
    excluded_subs = list()
    result_dict = dict()
    for sub in raw_subs:
        beh_data = sub_dict[sub][parse_data_files.UNITY_OUTPUT_FOLDER]
        # CHECK THAT THE OBJECTIVE PERFORMANCE IN THE UNATTENDED PAS-1 TRIALS IS NOT ABOVE CHANCE LEVEL
        unattended_data = beh_data[beh_data[parse_data_files.TRIAL_NUMBER] < REPLAY_TRIAL]
        if test_unattended(sub, unattended_data) != 1:
            result_dict[sub] = sub_dict[sub]
        else:
            excluded_subs.append(sub)
    print(f"{len(excluded_subs)} subjects excluded due to behavior.")
    return result_dict