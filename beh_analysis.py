import pandas as pd
import numpy as np
import os
import itertools
import parse_data_files
import plotter
import exclusion_criteria

BEH = "BEH"
ET = "ET"
MEAN = "Average"
STD = "STD"
MIN = "Min"
MAX = "Max"
CNT = "Count"
SUB = "Subject"

ATTN = "Attended"
UNATTEN = "Unattended"
PAS = "PAS"
NO_REPLAY = "no_replay"
NO_GAME = "no_game"
AVERSIVE_IND = [x for x in range(0, 20)]
AVERSIVE_VALENCE = 1
AVERSIVE = "Aversive"
NEUTRAL_IND = [x for x in range(20, 40)]
NEUTRAL_VALENCE = 0
NEUTRAL = "Neutral"
VALENCE_MAPPING = {str(AVERSIVE_VALENCE): AVERSIVE, str(NEUTRAL_VALENCE): NEUTRAL}


def PAS_calc_trial_prop_per_sub(df, subs, cond_name):
    ratings = [1, 2, 3, 4]
    filler_df = pd.DataFrame(list(itertools.product(subs, ratings))).rename(columns={0: SUB, 1: parse_data_files.SUBJ_ANS})
    # grouped_by_sub_PAS : index=[sub/PAS], column=sum of trials for this subject which had this PAS rating
    grouped_by_sub_PAS = df.groupby([SUB, parse_data_files.SUBJ_ANS]).agg({parse_data_files.TRIAL_NUMBER: 'count'})
    # df_PAS_pcntgs_per_sub : index=[sub/PAS], column=proportion (%) of trials for this subject which had this PAS rating
    df_PAS_pcntgs_per_sub = grouped_by_sub_PAS.groupby(level=0).apply(lambda x: 100 * x / float(x.sum()))
    df_PAS_pcntgs_per_sub.rename(columns={parse_data_files.TRIAL_NUMBER: f"{cond_name}{MEAN}_{PAS}"}, inplace=True)
    df_PAS_pcntgs_per_sub.reset_index(inplace=True)
    result_df = pd.merge(filler_df, df_PAS_pcntgs_per_sub, on=[SUB, parse_data_files.SUBJ_ANS], how='left')
    result_df.fillna(0, inplace=True)
    return result_df


def pas_comparison_cond(all_subs_df, save_path):
    """
    Extract data on PAS ratings: Create dataframes in which for each subject there is a breakdown of how many
    trials they had in each PAS score (1/2/3/4) in the attended and the unattended condition, and also within
    the unattended condition, between aversive and neutral stimuli.
    Output these tables, in addition to a plots of the frequency of each PAS rating in the different conditions.

    *** Quality Check *** :
    In the code there is an intentional redundancy: the ".._summary_df" are separate to the ".." original df, to make sure
    that they actually summarize the intended stats.
    In addition, "total_df" contains all the data from "PAS_task_comp_df" AND "PAS_task_val_comp_df"
    so that the numbers there SHOULD align.

    :param all_subs_df: dataframe containing all the trial data for all subjects
    :param save_path: path to save the data to
    :return: nothing. Saves everything to csvs and plots
    """
    unattended = all_subs_df[all_subs_df[parse_data_files.TRIAL_NUMBER] < exclusion_criteria.REPLAY_TRIAL]
    attended = all_subs_df[all_subs_df[parse_data_files.TRIAL_NUMBER] >= exclusion_criteria.REPLAY_TRIAL]
    conditions = {UNATTEN: unattended, ATTN: attended}
    subs = all_subs_df[SUB].unique().tolist()

    # **STEP 1**: Compare PAS ratings between attended and unattended conditions
    PAS_task_comp_list = list()
    PAS_task_comp_summary_list = list()
    for cond_name in conditions:
        cond = conditions[cond_name]
        cond_df = PAS_calc_trial_prop_per_sub(cond, subs, cond_name="")
        cond_df.loc[:, "condition"] = cond_name
        PAS_task_comp_list.append(cond_df)
        cond_PAS_list = list()
        # split by PAS to calculate cross-sub statistics for that rating
        for score in range(1, 5):
            score_df = cond_df[cond_df["subjectiveAwareness"] == score]
            score_summary = pd.DataFrame(score_df.describe())
            score_summary.loc[:, "PAS"] = score
            cond_PAS_list.append(score_summary)
        cond_PAS_df = pd.concat(cond_PAS_list)
        cond_PAS_df.loc[:, "condition"] = cond_name
        PAS_task_comp_summary_list.append(cond_PAS_df)

    # Save data (per subject, per rating)
    PAS_task_comp_df = pd.concat(PAS_task_comp_list)
    PAS_task_comp_df.to_csv(os.path.join(save_path, "PAS_comp_task.csv"), index=False)
    # Save data (across subjects, per rating)
    PAS_task_comp_summary_df = pd.concat(PAS_task_comp_summary_list)
    PAS_task_comp_summary_df.to_csv(os.path.join(save_path, "PAS_comp_task_proportion_stats.csv"))

    # Plot the distribution data: for each PAS rating, plot the proportion of trials rated as such, for UAT and AT
    plotter.plot_raincloud(df=PAS_task_comp_df, x_col_name="subjectiveAwareness", y_col_name="Average_PAS",
                           plot_title="PAS Rating Distribution",
                           plot_x_name="Subjective Awareness Rating", plot_y_name="Fraction of Trials (%)",
                           save_path=save_path, save_name="PAS_comp_task",
                           x_col_color_order=[["teal", "teal", "teal", "teal"],
                                              ["goldenrod", "goldenrod", "goldenrod", "goldenrod"]],
                           y_tick_interval=25, y_tick_min=0, y_tick_max=100,
                           x_axis_names=["1", "2", "3", "4"], y_tick_names=None, group_col_name="condition",
                           group_name_mapping=None, x_values=[1, 2, 3, 4])

    # **STEP 2**: Compare PAS ratings between aversive and neutral trials, output data as wide for JASP analysis
    PAS_task_val_comp_list = list()
    PAS_task_val_comp_summary_list = list()
    total_df = pd.DataFrame(list(subs)).rename(columns={0: SUB})
    total_df.set_index(SUB, inplace=True)
    for cond_name in conditions:
        cond = conditions[cond_name]
        cond_df = PAS_calc_trial_prop_per_sub(cond, subs, cond_name="")
        for score in range(1, 5):
            score_df = cond_df[cond_df["subjectiveAwareness"] == score]
            score_df = score_df[[SUB, "Average_PAS"]]
            score_df.rename(columns={"Average_PAS": f"{cond_name}_{score}"}, inplace=True)
            score_df.set_index(SUB, inplace=True)
            total_df = pd.concat([total_df, score_df], axis=1)
        for valence in [0, 1]:
            valence_name = "Aversive" if valence == 1 else "Neutral"
            val = conditions[cond_name][conditions[cond_name][parse_data_files.TRIAL_STIM_VAL] == valence]
            val_df = PAS_calc_trial_prop_per_sub(val, subs, cond_name="")
            val_df_forlist = val_df.copy()
            val_df_forlist.loc[:, "condition"] = cond_name
            val_df_forlist.loc[:, "valence"] = valence_name
            PAS_task_val_comp_list.append(val_df_forlist)
            cond_val_PAS_list = list()
            for score in range(1, 5):
                score_val_df = val_df[val_df["subjectiveAwareness"] == score]
                score_summary = pd.DataFrame(score_val_df.describe())
                score_summary.loc[:, "PAS"] = score
                cond_val_PAS_list.append(score_summary)
                score_val_df = score_val_df[[SUB, "Average_PAS"]]
                score_val_df.rename(columns={"Average_PAS": f"{cond_name}_{score}_{valence_name}"}, inplace=True)
                score_val_df.set_index(SUB, inplace=True)
                total_df = pd.concat([total_df, score_val_df], axis=1)
            cond_val_PAS_df = pd.concat(cond_val_PAS_list)
            cond_val_PAS_df.loc[:, "condition"] = cond_name
            cond_val_PAS_df.loc[:, "valence"] = valence_name
            PAS_task_val_comp_summary_list.append(cond_val_PAS_df)

    # Save data (per subject, per rating)
    PAS_task_val_comp_df = pd.concat(PAS_task_val_comp_list)
    PAS_task_val_comp_df.to_csv(os.path.join(save_path, "PAS_comp_task_valence.csv"))
    # Save data (across subjects, per rating)
    PAS_task_val_comp_summary_df = pd.concat(PAS_task_val_comp_summary_list)
    PAS_task_val_comp_summary_df.to_csv(os.path.join(save_path, "PAS_comp_task_valence_proportion_stats.csv"))

    # Plot the distribution data in the UAT: for each PAS rating, plot the proportion of trials rated as such, for Aversive and Neutral
    uat_only = PAS_task_val_comp_df[PAS_task_val_comp_df["condition"] == UNATTEN]
    plotter.plot_raincloud(df=uat_only,
                           x_col_name="subjectiveAwareness", y_col_name="Average_PAS",
                           plot_title="PAS Rating Distribution in Unattended",
                           plot_x_name="Subjective Awareness Rating", plot_y_name="Fraction of Trials (%)",
                           save_path=save_path, save_name="PAS_comp_task_valence_UAT",
                           x_col_color_order=[
                               ["#F39A9D", "#F39A9D", "#F39A9D", "#F39A9D"],
                               ["#6DB1BF", "#6DB1BF", "#6DB1BF", "#6DB1BF"]],
                           y_tick_interval=25, y_tick_min=0, y_tick_max=100,
                           x_axis_names=["1", "2", "3", "4"], y_tick_names=None, group_col_name="valence",
                           group_name_mapping=None, x_values=[1, 2, 3, 4], alpha_step=0.1, valpha=0.8, is_right=True)

    # Save the dataframe that includes one column per each PAS x Valence, per subject. This is the JASP structure
    total_df.to_csv(os.path.join(save_path, "PAS_comp_all_JASP_struct.csv"))

    return total_df, PAS_task_comp_df, PAS_task_val_comp_df


def calculate_pcnt_correct(df, filler_df, col, rating):
    grouped_by_sub = df.groupby([SUB, col]).agg({parse_data_files.TRIAL_NUMBER: 'count'})
    pcntgs_per_sub = grouped_by_sub.groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).reset_index()
    pcntgs_per_sub = pcntgs_per_sub[pcntgs_per_sub[col]]  # True and False complete each other anyway
    pcntgs_per_sub.drop(col, axis=1, inplace=True)
    pcntgs_per_sub = pd.merge(filler_df, pcntgs_per_sub, on=[SUB], how="left")
    pcntgs_per_sub.rename(columns={parse_data_files.TRIAL_NUMBER: f"pcntCorrectInObj_{rating}"}, inplace=True)
    return pcntgs_per_sub


def pas_obj_perf(all_subs_df, save_path):
    pas_ratings = [1, 2, 3, 4, "234"]
    subs = all_subs_df[SUB].unique().tolist()
    filler_df = pd.DataFrame(subs).rename(columns={0: SUB})
    result_list = list()
    for rating in pas_ratings:
        if rating != "234":
            try:
                df = all_subs_df[all_subs_df[parse_data_files.SUBJ_ANS] == rating]  # PAS = rating trials
            except KeyError:  # No such ratings
                df = None
        else:  # this is for 2-3-4 collapsed
            try:
                df = all_subs_df[all_subs_df[parse_data_files.SUBJ_ANS] != 1]
            except KeyError:
                df = pd.DataFrame()
        if not df.empty:
            pcntgs_per_sub = calculate_pcnt_correct(df, filler_df, parse_data_files.OBJ_ANS, rating)
        else:
            pcntgs_per_sub = pd.concat([filler_df, pd.DataFrame({f"pcntCorrectInObj_{rating}": [np.nan] * len(subs)})], axis=1)
        result_list.append(pcntgs_per_sub)

    result_list_by_sub = [df.set_index(SUB) for df in result_list]  # set subject as index to concatenate by
    result_df = pd.concat(result_list_by_sub, axis=1)
    result_df = result_df.loc[:, ~result_df.columns.duplicated()]
    result_df.reset_index(inplace=True)
    result_df.to_csv(os.path.join(save_path, f"objective_correct_per_pas.csv"), index=False)

    plot_res_list = list()
    for i in [1, 2, 3, 4, "234"]:
        small_df = result_df[["Subject", f"pcntCorrectInObj_{i}"]]
        small_df["subjectiveAwareness"] = i
        small_df.rename(columns={f"pcntCorrectInObj_{i}": "pcntCorrectInObj"}, inplace=True)
        small_df = small_df.sort_values(by=SUB).reset_index()
        plot_res_list.append(small_df)
    df_for_plot = pd.concat(plot_res_list)
    df_for_plot.to_csv(os.path.join(save_path, "PAS_pcnt_correct_in_objective_long.csv"), index=False)
    plotter.plot_raincloud(df_for_plot, "subjectiveAwareness", "pcntCorrectInObj", "Overall Performance in Objective Task Per Visibility",
                           "Subjective Awareness Rating", "% Correct in Objective Task",
                           save_path=save_path, save_name="PAS_pcnt_correct_in_objective",
                           x_col_color_order=[["#559CAD", "#559CAD", "#559CAD", "#559CAD"]],
                           y_tick_interval=25, y_tick_min=0, y_tick_max=100,
                           x_axis_names=["1", "2", "3", "4"], y_tick_names=None, group_col_name=None,
                           group_name_mapping=None, x_values=[1, 2, 3, 4], add_horizontal_line=25)

    return df_for_plot


def pas_val_perf(all_subs_df, save_path):
    pas_ratings = [1, 2, 3, 4, "234"]
    subs = all_subs_df[SUB].unique().tolist()
    result_list = list()
    for rating in pas_ratings:
        if rating != "234":
            df = all_subs_df[all_subs_df[parse_data_files.SUBJ_ANS] == rating]  # PAS = rating trials
        else:  # this is for 2-3-4 collapsed
            df = all_subs_df[all_subs_df[parse_data_files.SUBJ_ANS] != 1]

        sub_correct_percent = df.groupby([SUB]).mean()  # % correct in this subjective rating
        sub_trial_count = df.groupby([SUB]).count()  # number of total trials in this subjective rating
        temp_df = pd.DataFrame()
        temp_df[SUB] = subs
        temp_df.set_index(SUB, inplace=True) # set subject as index to concatenate by
        temp_df[f"trialNumber_{rating}"] = sub_trial_count["trialNumber"]
        temp_df[f"ValenceCorrectProportion_{rating}"] = sub_correct_percent[parse_data_files.VAL_ANS_CORRECT] * 100  # turn into %
        temp_df[f"ValenceCorrectCount_{rating}"] = temp_df[f"ValenceCorrectProportion_{rating}"] * temp_df[f"trialNumber_{rating}"]
        temp_df[f"ValenceWrongCount_{rating}"] = temp_df[f"trialNumber_{rating}"] - temp_df[f"ValenceCorrectCount_{rating}"]
        temp_df[f"ValenceWrongProportion_{rating}"] = 100 - temp_df[f"ValenceCorrectProportion_{rating}"]
        result_list.append(temp_df)

    result_df = pd.concat(result_list, axis=1)
    result_df = result_df.loc[:, ~result_df.columns.duplicated()]
    result_df.reset_index(inplace=True)
    result_df.to_csv(os.path.join(save_path, f"valence_correct_per_pas.csv"))

    # now plot
    plot_res_list = list()
    for i in [1, 2, 3, 4, "234"]:
        small_df = result_df[["Subject", f"ValenceCorrectProportion_{i}"]]
        small_df["subjectiveAwareness"] = i
        small_df.rename(columns={f"ValenceCorrectProportion_{i}": "ValenceCorrectProportion"}, inplace=True)
        plot_res_list.append(small_df)
    df_for_plot = pd.concat(plot_res_list)
    df_for_plot.fillna(0, inplace=True)
    df_for_plot.to_csv(os.path.join(save_path, "PAS_pcnt_correct_in_valence_long.csv"), index=False)
    plotter.plot_raincloud(df_for_plot, "subjectiveAwareness", "ValenceCorrectProportion", "Overall Performance in Valence Task Per Visibility",
                           "Subjective Awareness Rating", "% Correct in Valence Task",
                           save_path=save_path, save_name="PAS_pcnt_correct_in_valence",
                           x_col_color_order=[["#559CAD", "#559CAD", "#559CAD", "#559CAD"]],
                           y_tick_interval=25, y_tick_min=0, y_tick_max=100,
                           x_axis_names=["Unseen (PAS 1)", "Seen (PAS 2-4)"], y_tick_names=None, group_col_name=None,
                           group_name_mapping=None, x_values=[1, 2], add_horizontal_line=50)
    return df_for_plot


def val_obj_perf(all_subs_df, save_path):
    pas_ratings = [1, 2, 3, 4, "234"]
    subs = all_subs_df[SUB].unique().tolist()
    filler_df = pd.DataFrame(subs).rename(columns={0: SUB})
    result_list = list()
    for rating in pas_ratings:
        if rating != "234":
            df = all_subs_df[all_subs_df[parse_data_files.SUBJ_ANS] == rating]  # PAS = rating trials
        else:  # this is for 2-3-4 collapsed
            df = all_subs_df[all_subs_df[parse_data_files.SUBJ_ANS] != 1]
        for val in [0, 1]:
            df_val = df[df["stimValence"] == val]  # just the valence ones
            pcntgs_per_sub = calculate_pcnt_correct(df_val, filler_df, parse_data_files.OBJ_ANS, rating)
            pcntgs_per_sub_cols = pcntgs_per_sub.columns.tolist()
            pcntgs_per_sub.rename(columns={pcntgs_per_sub_cols[-1]: f"{pcntgs_per_sub_cols[-1]}_Valence{val}"}, inplace=True)
            pcntgs_per_sub = pcntgs_per_sub.sort_values(by=SUB).reset_index()
            result_list.append(pcntgs_per_sub)

    result_list_by_sub = [df.set_index(SUB) for df in result_list]  # set subject as index to concatenate by
    result_df = pd.concat(result_list_by_sub, axis=1)
    result_df = result_df.loc[:, ~result_df.columns.duplicated()]
    result_df.reset_index(inplace=True)
    result_df.drop(columns=["index"], inplace=True)
    result_df.to_csv(os.path.join(save_path, f"obj_correct_per_valence_pas.csv"))
    return result_df


def behavioral_analysis(all_subs_df, save_path):

    # A result folder
    beh_output_path = os.path.join(save_path, parse_data_files.UNITY_OUTPUT_FOLDER)
    if not (os.path.isdir(beh_output_path)):
        os.mkdir(beh_output_path)

    # STEP 1: VISIBILITY (PAS): subjects' PAS rating (1/2/3/4) frequency in the UAT v AT conditions, and between aversive and neutral stimuli
    pas_comparison_cond(all_subs_df, beh_output_path)

    # ANALYSIS 3: WERE SUBJECTS AT CHANCE IN THE OBJECTIVE TASK WHEN VISIBILITY WAS 1?
    pas_obj_perf(all_subs_df, beh_output_path)

    conds = {parse_data_files.UAT: all_subs_df[all_subs_df[parse_data_files.TRIAL_NUMBER] < 40],
             parse_data_files.AT: all_subs_df[all_subs_df[parse_data_files.TRIAL_NUMBER] >= 40]}
    obj_across = dict()
    for condition in conds:  # separate ET analysis between attended from unattended trials
        cond_output_path = os.path.join(beh_output_path, condition)
        if not (os.path.isdir(cond_output_path)):
            os.mkdir(cond_output_path)
        subs_df_cond = conds[condition]
        df_obj = pas_obj_perf(subs_df_cond, cond_output_path)
        df_obj.loc[:, "condition"] = condition
        obj_across[condition] = df_obj

    obj_across_df = pd.concat([obj_across[k] for k in obj_across.keys()])
    obj_across_df_new = obj_across_df[(obj_across_df["subjectiveAwareness"] == 1) | (obj_across_df["subjectiveAwareness"] == 234)]
    obj_across_df_new.loc[:, "subjectiveAwareness"] = obj_across_df_new["subjectiveAwareness"].map({1: 1, 234: 2})
    plotter.plot_raincloud(obj_across_df_new, "subjectiveAwareness", "pcntCorrectInObj",
                         "Overall Performance in Objective Task Per Visibility",
                         "Subjective Awareness Rating", "% Correct in Objective Task",
                         save_path=beh_output_path, save_name="PAS_pcnt_correct_in_objective_CONDITIONS",
                         group_col_name="condition", group_name_mapping={parse_data_files.UAT: "Unattended", parse_data_files.AT: "Attended"},
                         x_col_color_order=[["teal", "teal", "teal", "teal"], ["goldenrod", "goldenrod", "goldenrod", "goldenrod"]],
                         y_tick_interval=25, y_tick_min=0, y_tick_max=100,
                         x_axis_names=["Unseen (PAS 1)", "Seen (PAS 2-4)"], y_tick_names=None, x_values=[1, 2], add_horizontal_line=25)


    # ANALYSIS 4: WERE SUBJECTS AT CHANCE IN THE VALENCE TASK WHEN VISIBILITY WAS 1  - DEPRACATED
    """
    pas_val_perf(all_subs_df, beh_output_path)

    val_across = dict()
    for condition in conds:  # separate ET analysis between attended from unattended trials
        cond_output_path = os.path.join(beh_output_path, condition)
        if not (os.path.isdir(cond_output_path)):
            os.mkdir(cond_output_path)
        subs_df_cond = conds[condition]
        df_val = pas_val_perf(subs_df_cond, cond_output_path)
        df_val.loc[:, "condition"] = condition
        val_across[condition] = df_val

    val_across_df = pd.concat([val_across[k] for k in val_across.keys()])
    val_across_df_new = val_across_df[(val_across_df["subjectiveAwareness"] == 1) | (val_across_df["subjectiveAwareness"] == 234)]
    val_across_df_new.loc[:, "subjectiveAwareness"] = val_across_df_new["subjectiveAwareness"].map({1: 1, 234: 2})
    plotter.plot_raincloud(val_across_df_new, "subjectiveAwareness", "ValenceCorrectProportion", "Overall Performance in Valence Task Per Visibility",
                           "Subjective Awareness Rating", "% Correct in Valence Task",
                           save_path=beh_output_path, save_name="PAS_pcnt_correct_in_valence_CONDITIONS",
                           group_col_name="condition", group_name_mapping={parse_data_files.UAT: "Unattended", parse_data_files.AT: "Attended"},
                           x_col_color_order=[["teal", "teal", "teal", "teal"], ["goldenrod", "goldenrod", "goldenrod", "goldenrod"]],
                           y_tick_interval=25, y_tick_min=0, y_tick_max=100,
                           x_axis_names=["Unseen (PAS 1)", "Seen (PAS 2-4)"], y_tick_names=None,  x_values=[1, 2], add_horizontal_line=50)

    """
    # ANALYSIS 5: WERE SUBJECTS BETTER IN OBJECTIVE TASK FOR AVERSIVE / NEUTRAL STIMULI?
    val_obj_perf(all_subs_df, beh_output_path)

    # ANALYSIS 6: PLOT THE RELATIONSHIPS BETWEEN TIME TO OBJECTIVE TASK, AND OBJECTIVE TASK PERFORMANCE
    plotter.plot_raincloud(df=all_subs_df, x_col_name="objectiveIsCorrect", y_col_name="objectiveTimeFromLastIntactSec",
                           plot_title="SOA and Performance in Objective Task",
                           plot_x_name="Is Correct in 4AFC", plot_y_name="Time Between Last Intact and 4AFC (seconds)",
                           save_path=beh_output_path, save_name="4AFC_SOA",
                           x_col_color_order=[["#337C8E", "#972B60"]],
                           y_tick_interval=25, y_tick_min=0, y_tick_max=100,
                           x_axis_names=["0", "1"], y_tick_names=None, x_values=[False, True])

    return


def beh_pilot_plots(data_path, save_path):
    data = pd.read_csv(data_path)

    plotter.plot_raincloud(data, "Valence_numeric", "Average Valence Rating",
                           "Stimulus Valence Rating",
                           "Stimulus Valence", "Average Valence Rating",
                           save_path=save_path, save_name="avg_valence_ratings",
                           x_col_color_order=[["#6DB1BF", "#F39A9D"]],
                           y_tick_interval=1, y_tick_min=1, y_tick_max=5,
                           x_axis_names=["Neutral", "Aversive"],
                           y_tick_names=["1: Unhappy", "2", "3", "4", "5: Happy"], group_col_name=None,
                           group_name_mapping=None, x_values=[0, 1], valpha=0.8, alpha_step=0)

    plotter.plot_raincloud(data, "Valence_numeric", "Average Arousal Rating",
                           "Stimulus Arousal Rating",
                           "Stimulus Valence", "Average Arousal Rating",
                           save_path=save_path, save_name="avg_arousal_ratings",
                           x_col_color_order=[["#6DB1BF", "#F39A9D"]],
                           y_tick_interval=1, y_tick_min=1, y_tick_max=5,
                           x_axis_names=["Neutral", "Aversive"],
                           y_tick_names=["1: Calm", "2", "3", "4", "5: Excited"], group_col_name=None,
                           group_name_mapping=None, x_values=[0, 1], valpha=0.8, alpha_step=0)
    return
