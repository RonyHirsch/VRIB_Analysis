import pandas as pd
import os
import re
import math
import numpy as np
import itertools
import seaborn as sns
import plotter
import parse_data_files
SUB = "Subject"
INTACT = "Intact"
SCRAMBLED = "Scrambled"


def plot_avg_gaze(all_subs_df, save_path):
    subs = all_subs_df[SUB].unique().tolist()

    intact_df = all_subs_df[[SUB, parse_data_files.SUBJ_ANS, parse_data_files.BUSSTOP_GAZE_DUR_AVG_INTACT]]
    intact_df_avg_per_sub = intact_df.groupby([SUB, parse_data_files.SUBJ_ANS]).mean().reset_index()
    intact_df_avg_per_sub.rename(columns={parse_data_files.BUSSTOP_GAZE_DUR_AVG_INTACT: "Average_Gaze"}, inplace=True)
    intact_df_avg_per_sub["Image Type"] = True
    scrambled_df = all_subs_df[[SUB, parse_data_files.SUBJ_ANS, parse_data_files.BUSSTOP_GAZE_DUR_AVG_SCRAMBLED]]
    scrambled_df_avg_per_sub = scrambled_df.groupby([SUB, parse_data_files.SUBJ_ANS]).mean().reset_index()
    scrambled_df_avg_per_sub.rename(columns={parse_data_files.BUSSTOP_GAZE_DUR_AVG_SCRAMBLED: "Average_Gaze"}, inplace=True)
    scrambled_df_avg_per_sub["Image Type"] = False
    df = pd.concat([intact_df_avg_per_sub, scrambled_df_avg_per_sub])

    ratings = [1, 2, 3, 4]
    filler_df = pd.DataFrame(list(itertools.product(subs, ratings))).rename(columns={0: SUB, 1: parse_data_files.SUBJ_ANS})
    result_df = pd.merge(filler_df, df, on=[SUB, parse_data_files.SUBJ_ANS], how='left')
    result_df.fillna(0, inplace=True)

    y_name = "Average Gaze Duration (Seconds)"
    y_save_name = re.sub(r'(?<!^)(?=[A-Z])', '_', parse_data_files.BUSSTOP_GAZE_DUR_AVG_TOTAL).lower()
    x_name = re.sub('([A-Z])', r' \1', parse_data_files.SUBJ_ANS).title()
    x_save_name = re.sub(r'(?<!^)(?=[A-Z])', '_', parse_data_files.SUBJ_ANS).lower()
    x_axis = ["1", "2", "3", "4"]
    x_axis_vals = [1, 2, 3, 4]

    ymin = 0  # gaze duration cannot be negative
    y_max_val = max(all_subs_df[parse_data_files.BUSSTOP_GAZE_DUR_AVG_TOTAL])
    ymax = int(math.ceil(y_max_val / 10.0)) * 10  # round up to the nearest 10
    y_tick_skip = 1 if ymax - ymin < 11 else 5

    plotter.plot_raincloud(df=df, x_col_name=parse_data_files.SUBJ_ANS,
                           y_col_name="Average_Gaze",
                           plot_title=f"{y_name} Per {x_name}",
                           plot_x_name=parse_data_files.SUBJ_ANS,
                           plot_y_name="Average Gaze Duration",
                           save_path=save_path, save_name=f"{y_save_name}_over_{x_save_name}",
                           y_tick_interval=y_tick_skip, y_tick_min=ymin, y_tick_max=ymax,
                           x_axis_names=x_axis, y_tick_names=None,
                           group_col_name="Image Type",
                           group_name_mapping={"True": INTACT, "False": SCRAMBLED},
                           x_col_color_order=[["salmon", "salmon", "salmon", "salmon"], ["steelblue", "steelblue", "steelblue", "steelblue"]],
                           x_values=x_axis_vals)

    df.to_csv(os.path.join(save_path, "avg_gaze_duration_per_vis_intact_scrambled.csv"))
    return


def analyze_valence_gaze(all_subs_df, save_path):
    relevant_cols = ["Subject", "subjectiveAwareness", "trialNumber", "stimValence", "busstopGazeDurIntactAvg", "busstopGazeDurScrambledAvg"]
    df = all_subs_df[relevant_cols]
    pas_ratings = [1, 2, 3, 4, "234"]
    result_list = list()
    subs = all_subs_df[SUB].unique().tolist()
    filler_df = pd.DataFrame(subs).rename(columns={0: SUB})
    for rating in pas_ratings:
        if rating != "234":
            d = df[df["subjectiveAwareness"] == rating]  # PAS = rating trials
        else:  # this is for 2-3-4 collapsed
            d = df[df["subjectiveAwareness"] != 1]
        data = d.groupby([SUB, "stimValence"]).agg({"busstopGazeDurIntactAvg": 'mean', "busstopGazeDurScrambledAvg": 'mean'}).reset_index()
        # now not per valence but per rating
        data_across = d.groupby([SUB]).agg({"busstopGazeDurIntactAvg": 'mean', "busstopGazeDurScrambledAvg": 'mean'}).reset_index()
        data_across.rename(columns={"busstopGazeDurIntactAvg": f"busstopGazeDurIntactAvg_PAS{rating}_ALL",
                                    "busstopGazeDurScrambledAvg": f"busstopGazeDurScrambledAvg_PAS{rating}_ALL"}, inplace=True)
        result_list.append(data_across)
        for valence in [0, 1]:
            tmp = data[data["stimValence"] == valence]
            tmp.rename(columns={"busstopGazeDurIntactAvg": f"busstopGazeDurIntactAvg_PAS{rating}_VAL{valence}",
                                "busstopGazeDurScrambledAvg": f"busstopGazeDurScrambledAvg_PAS{rating}_VAL{valence}"}, inplace=True)
            tmp.drop(columns=["stimValence"], inplace=True)
            dat = pd.merge(filler_df, tmp, on=[SUB], how="left")
            result_list.append(dat)
    result_df = pd.concat(result_list, axis=1)
    result_df = result_df.loc[:, ~result_df.columns.duplicated()]
    result_df.to_csv(os.path.join(save_path, f"avg_gaze_per_pas.csv"))

    # plot 1 v 234 average gaze duration
    plot_res_list = list()
    for i in [1, 2, 3, 4]:
        for val in [0, 1]:
            small_df = result_df[["Subject", f"busstopGazeDurIntactAvg_PAS{i}_VAL{val}"]]
            small_df["subjectiveAwareness"] = i
            small_df["Stimulus"] = "Neutral" if val==0 else "Aversive"
            small_df.rename(columns={f"busstopGazeDurIntactAvg_PAS{i}_VAL{val}": "busstopGazeDurIntactAvg"}, inplace=True)
            plot_res_list.append(small_df)
    df_for_plot = pd.concat(plot_res_list)
    df_for_plot.fillna(0, inplace=True)
    plotter.plot_raincloud(df_for_plot, "subjectiveAwareness", "busstopGazeDurIntactAvg", "Average Gaze Duration (Intact) Per Visibility",
                           "Subjective Awareness Rating", "Average Gaze Duration (Seconds)",
                           save_path=save_path, save_name="PAS_avg_gaze_duration_intact",
                           x_col_color_order=[["steelblue", "steelblue", "steelblue", "steelblue"],
                                              ["palevioletred", "palevioletred", "palevioletred", "palevioletred"]],
                           y_tick_interval=1, y_tick_min=0, y_tick_max=None,
                           x_axis_names=["1", "2", "3", "4"], y_tick_names=None, group_col_name="Stimulus",
                           group_name_mapping=None, x_values=[1, 2, 3, 4])

    # in visibility-1, plot aversive v neutral
    plot_res_list = list()
    for i in [0, 1]:  # JUST FOR 1 AND 2-4 COLLAPSED
        small_df = result_df[["Subject", f"busstopGazeDurIntactAvg_PAS1_VAL{i}"]]
        small_df["Valence"] = i
        small_df.rename(columns={f"busstopGazeDurIntactAvg_PAS1_VAL{i}": "busstopGazeDurIntactAvg_PAS1"}, inplace=True)
        plot_res_list.append(small_df)
    df_for_plot = pd.concat(plot_res_list)
    df_for_plot.fillna(0, inplace=True)
    plotter.plot_raincloud(df_for_plot, "Valence", "busstopGazeDurIntactAvg_PAS1",
                           "Average Gaze Duration (Intact) Per Valence in Invisible Trials",
                           "Valence", "Average Gaze Duration (Seconds)",
                           save_path=save_path, save_name="PAS1_avg_gaze_duration_intact_per_valence",
                           x_col_color_order=[["steelblue", "deeppink"]],
                           y_tick_interval=1, y_tick_min=0, y_tick_max=7,
                           x_axis_names=["Neutral", "Aversive"], y_tick_names=None, group_col_name=None,
                           group_name_mapping=None, x_values=[0, 1])
    return


def et_analysis(all_subs_df, save_path):

    # A result folder
    et_output_path = os.path.join(save_path, "et")
    if not (os.path.isdir(et_output_path)):
        os.mkdir(et_output_path)

    # Descriptive: In trials where subjects report not seeing the stimulus, do they gaze at the stimuli?
    plot_avg_gaze(all_subs_df, et_output_path)

    # ANALYSIS : is there a difference in gaze patterns between aversive and neutral stimuli within each visibility rating?
    analyze_valence_gaze(all_subs_df, et_output_path)
    return