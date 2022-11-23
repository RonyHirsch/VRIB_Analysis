import pandas as pd
import os
import re
import math
import numpy as np
import itertools
import seaborn as sns
import plotter
import parse_data_files

pd.options.mode.chained_assignment = None  # default='warn' see: https://stackoverflow.com/questions/20625582/how-to-deal-with-settingwithcopywarning-in-pandas

""" VRIB gaze analysis manager

This module manages everything related to the processing of eye-tracking data towards analysis. Note that the statistical
analyses themselves (linear mixed models, t-tests etc) are not done here; the goal of this module is to output summary
data and plots, as well as aggregate data into a group dataframe that will be later analyzed with R (for linear models)
and JASP (for t-tests). 

@authors: RonyHirsch
"""

SUB = "Subject"
INTACT = "Intact"
SCRAMBLED = "Scrambled"

cond_map = {parse_data_files.UAT: "UAT", parse_data_files.AT: "AT"}


def plot_avg_gaze(all_subs_df, save_path, cond_name=" "):
    subs = all_subs_df[SUB].unique().tolist()

    intact_df = all_subs_df[[SUB, parse_data_files.SUBJ_ANS, parse_data_files.BUSSTOP_GAZE_DUR_AVG_INTACT]]
    intact_df_avg_per_sub = intact_df.groupby([SUB, parse_data_files.SUBJ_ANS]).mean().reset_index()
    intact_df_avg_per_sub.rename(columns={parse_data_files.BUSSTOP_GAZE_DUR_AVG_INTACT: "Average_Gaze"}, inplace=True)
    intact_df_avg_per_sub["Image Type"] = True
    scrambled_df = all_subs_df[[SUB, parse_data_files.SUBJ_ANS, parse_data_files.BUSSTOP_GAZE_DUR_AVG_SCRAMBLED]]
    scrambled_df_avg_per_sub = scrambled_df.groupby([SUB, parse_data_files.SUBJ_ANS]).mean().reset_index()
    scrambled_df_avg_per_sub.rename(columns={parse_data_files.BUSSTOP_GAZE_DUR_AVG_SCRAMBLED: "Average_Gaze"}, inplace=True)
    scrambled_df_avg_per_sub["Image Type"] = False

    df = pd.concat([intact_df_avg_per_sub, scrambled_df_avg_per_sub])  # one AFTER the other

    ratings = [1, 2, 3, 4]
    filler_df = pd.DataFrame(list(itertools.product(subs, ratings))).rename(columns={0: SUB, 1: parse_data_files.SUBJ_ANS})
    result_df = pd.merge(filler_df, df, on=[SUB, parse_data_files.SUBJ_ANS], how='left')
    #result_df.fillna(0, inplace=True)

    y_name = "Gaze Duration"
    y_save_name = re.sub(r'(?<!^)(?=[A-Z])', '_', parse_data_files.BUSSTOP_GAZE_DUR_AVG_TOTAL).lower()
    x_name = re.sub('([A-Z])', r' \1', parse_data_files.SUBJ_ANS).title()
    x_save_name = re.sub(r'(?<!^)(?=[A-Z])', '_', parse_data_files.SUBJ_ANS).lower()
    x_axis = ["1", "2", "3", "4"]
    x_axis_vals = [1, 2, 3, 4]

    ymin = 0  # gaze duration cannot be negative
    y_max_val = max(all_subs_df[parse_data_files.BUSSTOP_GAZE_DUR_AVG_TOTAL])
    ymax = int(math.ceil(y_max_val / 10.0)) * 10  # round up to the nearest 10  - DEPRECATED, TO MAKE THE PLOTS UNIFORM ACROSS CONDITIONS
    y_tick_skip = 1 if ymax - ymin < 11 else 5

    plotter.plot_raincloud(df=df, x_col_name=parse_data_files.SUBJ_ANS,
                           y_col_name="Average_Gaze",
                           plot_title=f"{y_name} Per {x_name} in {cond_map[cond_name]}",
                           plot_x_name="Subjective Awareness Rating",
                           plot_y_name="Average Gaze Duration (Seconds)",
                           save_path=save_path, save_name=f"{y_save_name}_over_{x_save_name}",
                           y_tick_interval=y_tick_skip, y_tick_min=ymin, y_tick_max=7,
                           x_axis_names=x_axis, x_values=x_axis_vals, y_tick_names=None,
                           group_col_name="Image Type",
                           group_name_mapping={"True": INTACT, "False": SCRAMBLED},
                           x_col_color_order=[["#2E4052", "#2E4052", "#2E4052", "#2E4052"], ["#22577A", "#22577A", "#22577A", "#22577A"]],
                          alpha_step=0, valpha=0.6)

    df.to_csv(os.path.join(save_path, "avg_gaze_duration_per_vis_intact_scrambled.csv"))
    return


def analyze_valence_gaze(all_subs_df, save_path, cond_name):
    relevant_cols = [SUB, parse_data_files.SUBJ_ANS, parse_data_files.TRIAL_NUMBER, parse_data_files.OBJ_ANS,
                     parse_data_files.SUBJ_BUSSTOP, parse_data_files.OBJ_BUSSTOP, parse_data_files.TRIAL_STIM_VAL,
                     parse_data_files.BUSSTOP_GAZE_DUR_AVG_INTACT, parse_data_files.BUSSTOP_GAZE_DUR_AVG_SCRAMBLED]
    df = all_subs_df[relevant_cols]
    # first, process df into a long format and save it
    long_df = pd.melt(df, id_vars=relevant_cols[:-2], value_vars=relevant_cols[-2:], var_name="presentation")
    long_df.loc[:, "presentation"] = long_df["presentation"].map({parse_data_files.BUSSTOP_GAZE_DUR_AVG_INTACT: "intact", parse_data_files.BUSSTOP_GAZE_DUR_AVG_SCRAMBLED: "scrambled"})
    long_df.rename({"value": "avgBusstopGazeDuration"}, axis=1, inplace=True)
    long_df.to_csv(os.path.join(save_path, f"avg_gaze_per_pas_long.csv"), index=False)

    pas_ratings = [1, 2, 3, 4, "234"]
    result_list = list()
    subs = all_subs_df[SUB].unique().tolist()
    filler_df = pd.DataFrame(subs).rename(columns={0: SUB})
    for rating in pas_ratings:
        if rating != "234":
            d = df[df[parse_data_files.SUBJ_ANS] == rating]  # PAS = rating trials
        else:  # this is for 2-3-4 collapsed
            d = df[df[parse_data_files.SUBJ_ANS] != 1]
        data = d.groupby([SUB, parse_data_files.TRIAL_STIM_VAL]).agg({parse_data_files.BUSSTOP_GAZE_DUR_AVG_INTACT: 'mean', parse_data_files.BUSSTOP_GAZE_DUR_AVG_SCRAMBLED: 'mean'}).reset_index()
        # now not per valence but per rating
        data_across = d.groupby([SUB]).agg({parse_data_files.BUSSTOP_GAZE_DUR_AVG_INTACT: 'mean', parse_data_files.BUSSTOP_GAZE_DUR_AVG_SCRAMBLED: 'mean'}).reset_index()
        data_across.rename(columns={parse_data_files.BUSSTOP_GAZE_DUR_AVG_INTACT: f"{parse_data_files.BUSSTOP_GAZE_DUR_AVG_INTACT}_PAS{rating}_ALL",
                                    parse_data_files.BUSSTOP_GAZE_DUR_AVG_SCRAMBLED: f"{parse_data_files.BUSSTOP_GAZE_DUR_AVG_SCRAMBLED}_PAS{rating}_ALL"}, inplace=True)
        result_list.append(data_across)
        for valence in [0, 1]:
            tmp = data[data[parse_data_files.TRIAL_STIM_VAL] == valence]
            tmp.rename(columns={parse_data_files.BUSSTOP_GAZE_DUR_AVG_INTACT: f"{parse_data_files.BUSSTOP_GAZE_DUR_AVG_INTACT}_PAS{rating}_VAL{valence}",
                                parse_data_files.BUSSTOP_GAZE_DUR_AVG_SCRAMBLED: f"{parse_data_files.BUSSTOP_GAZE_DUR_AVG_SCRAMBLED}_PAS{rating}_VAL{valence}"}, inplace=True)
            tmp.drop(columns=[parse_data_files.TRIAL_STIM_VAL], inplace=True)
            dat = pd.merge(filler_df, tmp, on=[SUB], how="left")
            result_list.append(dat)

    result_list_by_sub = [df.set_index(SUB) for df in result_list]  # set subject as index to concatenate by
    result_df = pd.concat(result_list_by_sub, axis=1)
    result_df.reset_index(inplace=True)
    result_df.to_csv(os.path.join(save_path, f"avg_gaze_per_pas.csv"), index=False)

    # plot 1 v 234 average gaze duration
    plot_res_list = list()
    for i in [1, 2, 3, 4]:
        for val in [0, 1]:
            small_df = result_df[[SUB, f"{parse_data_files.BUSSTOP_GAZE_DUR_AVG_INTACT}_PAS{i}_VAL{val}"]]
            small_df[parse_data_files.SUBJ_ANS] = i
            small_df["Stimulus"] = "Neutral" if val == 0 else "Aversive"
            small_df.rename(columns={f"{parse_data_files.BUSSTOP_GAZE_DUR_AVG_INTACT}_PAS{i}_VAL{val}": parse_data_files.BUSSTOP_GAZE_DUR_AVG_INTACT}, inplace=True)
            plot_res_list.append(small_df)
    df_for_plot = pd.concat(plot_res_list)  # VERTICALLY
    #df_for_plot.fillna(0, inplace=True)
    plotter.plot_raincloud(df_for_plot, parse_data_files.SUBJ_ANS, parse_data_files.BUSSTOP_GAZE_DUR_AVG_INTACT, f"Gaze Duration (Intact) Per Visibility in {cond_map[cond_name]}",
                           "Subjective Awareness Rating", "Average Gaze Duration (Seconds)",
                           save_path=save_path, save_name="PAS_avg_gaze_duration_intact",
                           x_col_color_order=[["#6DB1BF", "#6DB1BF", "#6DB1BF", "#6DB1BF"],
                                              ["#F39A9D", "#F39A9D", "#F39A9D", "#F39A9D"]],
                           y_tick_interval=1, y_tick_min=0, y_tick_max=7,
                           x_axis_names=["1", "2", "3", "4"], y_tick_names=None, group_col_name="Stimulus",
                           group_name_mapping=None, x_values=[1, 2, 3, 4], alpha_step=0.05, valpha=0.8)

    # in visibility-1, plot aversive v neutral
    """
    plot_res_list = list()
    for i in [0, 1]:  # JUST FOR 1 AND 2-4 COLLAPSED
        small_df = result_df[[SUB, f"{parse_data_files.BUSSTOP_GAZE_DUR_AVG_INTACT}_PAS1_VAL{i}"]]
        small_df["Valence"] = i
        small_df.rename(columns={f"{parse_data_files.BUSSTOP_GAZE_DUR_AVG_INTACT}_PAS1_VAL{i}": f"{parse_data_files.BUSSTOP_GAZE_DUR_AVG_INTACT}_PAS1"}, inplace=True)
        plot_res_list.append(small_df)
    df_for_plot = pd.concat(plot_res_list)  # VERTICALLY
    #df_for_plot.fillna(0, inplace=True)
    plotter.plot_raincloud(df_for_plot, "Valence", f"{parse_data_files.BUSSTOP_GAZE_DUR_AVG_INTACT}_PAS1",
                           "Average Gaze Duration (Intact) Per Valence in Invisible Trials",
                           "Valence", "Average Gaze Duration (Seconds)",
                           save_path=save_path, save_name="PAS1_avg_gaze_duration_intact_per_valence",
                           x_col_color_order=[["#6DB1BF", "#F39A9D"]],
                           y_tick_interval=1, y_tick_min=0, y_tick_max=10,
                           x_axis_names=["Neutral", "Aversive"], y_tick_names=None, group_col_name=None,
                           group_name_mapping=None, x_values=[0, 1], alpha_step=0.1, valpha=0.8)
    """
    return


def unify_for_comparison(cond_dict, save_path, save_name, new_col_name):
    result_list = list()
    for cond_name in cond_dict.keys():
        df = cond_dict[cond_name]
        df.loc[:, new_col_name] = cond_name
        result_list.append(df)
    result_df = pd.concat(result_list)  # VERTICALLY
    result_df.to_csv(os.path.join(save_path, save_name), index=False)
    return result_df


def et_analysis(all_subs_df, save_path):

    # A result folder
    et_output_path = os.path.join(save_path, "et")
    if not (os.path.isdir(et_output_path)):
        os.mkdir(et_output_path)

    conds = {parse_data_files.UAT: all_subs_df[all_subs_df[parse_data_files.TRIAL_NUMBER] < 40],
             parse_data_files.AT: all_subs_df[all_subs_df[parse_data_files.TRIAL_NUMBER] >= 40]}

    for condition in conds:  # separate ET analysis between attended from unattended trials
        cond_output_path = os.path.join(et_output_path, condition)
        if not (os.path.isdir(cond_output_path)):
            os.mkdir(cond_output_path)

        subs_df_cond = conds[condition]

        # Descriptive: For each visibility rating, do they gaze at the stimuli?
        plot_avg_gaze(subs_df_cond, cond_output_path, condition)

        # ANALYSIS : is there a difference in gaze patterns between aversive and neutral stimuli within each visibility rating?
        analyze_valence_gaze(subs_df_cond, cond_output_path, condition)

    # create comparison
    comp_output_path = os.path.join(et_output_path, "comparison")
    if not (os.path.isdir(comp_output_path)):
        os.mkdir(comp_output_path)

    avg_gaze = {parse_data_files.UAT: None, parse_data_files.AT: None}

    for condition in avg_gaze.keys():  # separate ET analysis between attended from unattended trials
        cond_output_path = os.path.join(et_output_path, condition)
        avg_gaze_in_cond = pd.read_csv(os.path.join(cond_output_path, f"avg_gaze_per_pas_long.csv"))
        avg_gaze[condition] = avg_gaze_in_cond

    unified_df = unify_for_comparison(avg_gaze, comp_output_path, f"avg_gaze_per_pas_long.csv", "condition")
    unified_df_intact = unified_df[unified_df["presentation"] == "intact"]
    unified_df_intact.to_csv(os.path.join(comp_output_path, f"avg_gaze_per_pas_long_intact.csv"), index=False)
    return
