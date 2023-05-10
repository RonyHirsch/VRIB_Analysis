import sys
import pandas as pd
import os
import warnings
import parse_data_files
import gaze_analysis
import beh_analysis
import exclusion_criteria


""" VRIB analysis manager

This module manages everything related to the parsing and processing of the behavioral and eye-tracking output data 
of the VRIB experiment (see https://osf.io/6jyqx). 

@authors: RonyHirsch
"""


warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
SUB = "Subject"


def manage_analyses(data_path, save_path):
    # Open folder for saving processing outputs
    if not (os.path.isdir(save_path)):
        os.mkdir(save_path)

    # Load and parse all data
    sub_dict = parse_data_files.extract_subject_data(data_path, save_path)

    # Unify the the data and EXCLUDE subjects and/or trials
    sub_dict = exclusion_criteria.beh_exclusion(sub_dict)
    print(f" ---------------------- DATA ANALYSIS: N={len(sub_dict.keys())} ----------------------")

    # STEP 1: behavioral analysis
    all_subs_beh_df = pd.concat([sub_dict[sub][parse_data_files.UNITY_OUTPUT_FOLDER] for sub in sub_dict], keys=sub_dict.keys(), names=[SUB, None]).reset_index(level=SUB)
    all_subs_beh_df.to_csv(os.path.join(save_path, "raw_all_subs.csv"))
    beh_analysis.behavioral_analysis(all_subs_beh_df, save_path, exp="prereg")  # exp="pilot" for pilots 1 & 2, "prereg" for pre-registered one

    # STEP 2: gaze analysis
    gaze_analysis.et_analysis(all_subs_beh_df, save_path)

    return


if __name__ == "__main__":
    orig_stdout = sys.stdout
    f = open(os.path.join(r"\prereg\processed", "output_log.txt"), 'w')
    sys.stdout = f

    manage_analyses(data_path=r"\prereg\raw",
                    save_path=r"\prereg\processed")

    sys.stdout = orig_stdout
    f.close()



