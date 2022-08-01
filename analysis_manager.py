import pandas as pd
import os
import parse_data_files
import gaze_analysis
import beh_analysis
import peripheral_analysis
import exclusion_criteria

SUB = "Subject"
INTACT = "Intact"
SCRAMBLED = "Scrambled"


def manage_analyses(data_path, save_path):
    # Open folder for saving processing outputs
    if not (os.path.isdir(save_path)):
        os.mkdir(save_path)

    # Load and parse all data
    sub_dict = parse_data_files.extract_subject_data(data_path, save_path)

    # Unify the the data and EXCLUDE subjects and/or trials
    sub_dict = exclusion_criteria.beh_exclusion(sub_dict)
    print(f"DATA ANALYSIS: N={len(sub_dict.keys())}")

    # STEP 1: behavioral analysis
    all_subs_beh_df = pd.concat([sub_dict[sub][parse_data_files.UNITY_OUTPUT_FOLDER] for sub in sub_dict], keys=sub_dict.keys(), names=[SUB, None]).reset_index(level=SUB)
    all_subs_beh_df.to_csv(os.path.join(save_path, "raw_all_subs.csv"))
    beh_analysis.behavioral_analysis(all_subs_beh_df, save_path)
    return
    # STEP 2: gaze analysis
    gaze_analysis.et_analysis(all_subs_beh_df, save_path)


    # STEP 3: peripheral data analysis
    peripheral_analysis.peripheral_analysis(sub_dict, save_path)

    return


if __name__ == "__main__":
    manage_analyses(data_path=r"C:\Users\ronyhirschhorn\Downloads\pilot\raw", save_path=r"C:\Users\ronyhirschhorn\Downloads\pilot\result")
    #manage_analyses(data_path=r"D:\VRIB\round 1\raw", save_path=r"D:\VRIB\round 1\ANALYSIS_TEMP")  # ROUND 1
    #manage_analyses(data_path=r"D:\VRIB\round 2\raw", save_path=r"D:\VRIB\round 2\ANALYSIS_TEMP")  # ROUND 2


