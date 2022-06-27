import pandas as pd
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

ATTN = "ATTENDED"
UNATTEN = "UNATTENDED"
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
    df_PAS_pcntgs_per_sub.rename(columns={parse_data_files.TRIAL_NUMBER: f"{cond_name}_{MEAN}_{PAS}"}, inplace=True)
    df_PAS_pcntgs_per_sub.reset_index(inplace=True)
    result_df = pd.merge(filler_df, df_PAS_pcntgs_per_sub, on=[SUB, parse_data_files.SUBJ_ANS], how='left')
    result_df.fillna(0, inplace=True)
    return result_df


def pas_comparison_cond(all_subs_df, save_path):
    unattended = all_subs_df[all_subs_df[parse_data_files.TRIAL_NUMBER] < exclusion_criteria.REPLAY_TRIAL]
    attended = all_subs_df[all_subs_df[parse_data_files.TRIAL_NUMBER] >= exclusion_criteria.REPLAY_TRIAL]
    conditions = {UNATTEN: unattended, ATTN: attended}
    subs = all_subs_df[SUB].unique().tolist()
    # count how many PAS 1/2/3/4 are per each subject:
    df_PAS_stats_list = list()
    for cond_name in conditions:
        cond = conditions[cond_name]
        # WITHIN A COND, ACROSS ALL TRIALS
        result_df = PAS_calc_trial_prop_per_sub(cond, subs, cond_name=cond_name + "_ALL")
        # PER VALENCE
        aversive = cond[cond[parse_data_files.TRIAL_STIM_VAL] == 1]
        neutral = cond[cond[parse_data_files.TRIAL_STIM_VAL] == 0]
        valences = {AVERSIVE: aversive, NEUTRAL: neutral}
        val_list = list()
        for val in valences:
            data = valences[val]
            result_df_val = PAS_calc_trial_prop_per_sub(data, subs, cond_name=cond_name + f"_{val}")
            val_list.append(result_df_val)
        df_PAS_stats_list.append(result_df)
        df_PAS_stats_list.extend(val_list)
    df_PAS_stats_list_by_sub = [df.set_index(SUB) for df in df_PAS_stats_list]  # set subject as index to concatenate by
    df_PAS_stats_unified = pd.concat(df_PAS_stats_list_by_sub, axis=1)
    df_PAS_stats_unified = df_PAS_stats_unified.loc[:, ~df_PAS_stats_unified.columns.duplicated()]
    df_PAS_stats_unified.reset_index(inplace=True)
    df_PAS_stats_unified.to_csv(os.path.join(save_path, f"{PAS}_rate_per_sub.csv"))

    # trial PAS ratings per condition: attended, unattended
    for_plot_UA = df_PAS_stats_unified[["Subject", "subjectiveAwareness", "UNATTENDED_ALL_Average_PAS"]]
    for_plot_UA["condition"] = "Unattended"
    for_plot_UA.rename(columns={"UNATTENDED_ALL_Average_PAS": "Average_PAS"}, inplace=True)
    for_plot_AT = df_PAS_stats_unified[["Subject", "subjectiveAwareness", "ATTENDED_ALL_Average_PAS"]]
    for_plot_AT["condition"] = "Attended"
    for_plot_AT.rename(columns={"ATTENDED_ALL_Average_PAS": "Average_PAS"}, inplace=True)
    df_PAS_for_plot = pd.concat([for_plot_UA, for_plot_AT])
    plotter.plot_raincloud(df_PAS_for_plot, "subjectiveAwareness", "Average_PAS", "Overall Trial PAS Ratings",
                           "Subjective Awareness Rating", "Average % of Trials",
                           save_path=save_path, save_name="PAS_rate_across_subs",
                           x_col_color_order=[["teal", "teal", "teal", "teal"],
                                              ["goldenrod", "goldenrod", "goldenrod", "goldenrod"]],
                           y_tick_interval=25, y_tick_min=0, y_tick_max=100,
                           x_axis_names=["1", "2", "3", "4"], y_tick_names=None, group_col_name="condition",
                           group_name_mapping=None, x_values=[1, 2, 3, 4])


    # in UNATTENDED, trial PAS ratings between aversive and neutral
    for_plot_UA_Aversive = df_PAS_stats_unified[["Subject", "subjectiveAwareness", "UNATTENDED_Aversive_Average_PAS"]]
    for_plot_UA_Aversive["Stimulus"] = "Aversive"
    for_plot_UA_Aversive.rename(columns={"UNATTENDED_Aversive_Average_PAS": "Average_PAS"}, inplace=True)
    for_plot_UA_Neutral = df_PAS_stats_unified[["Subject", "subjectiveAwareness", "UNATTENDED_Neutral_Average_PAS"]]
    for_plot_UA_Neutral["Stimulus"] = "Neutral"
    for_plot_UA_Neutral.rename(columns={"UNATTENDED_Neutral_Average_PAS": "Average_PAS"}, inplace=True)
    df_UA_PAS_for_plot = pd.concat([for_plot_UA_Aversive, for_plot_UA_Neutral])
    plotter.plot_raincloud(df_UA_PAS_for_plot, "subjectiveAwareness", "Average_PAS", "Unattended Condition PAS Ratings",
                           "Subjective Awareness Rating", "Average % of Trials",
                           save_path=save_path, save_name="PAS_rate_across_subs_UAT",
                           x_col_color_order=[
                               ["#88292F", "#88292F", "#88292F", "#88292F"],
                               ["#22395B", "#22395B", "#22395B", "#22395B"]],
                           y_tick_interval=25, y_tick_min=0, y_tick_max=100,
                           x_axis_names=["1", "2", "3", "4"], y_tick_names=None, group_col_name="Stimulus",
                           group_name_mapping=None, x_values=[1, 2, 3, 4], alpha_step=0.25, valpha=0.65)


    df_PAS_stats_unified_across_subs_avg = df_PAS_stats_unified.groupby(["subjectiveAwareness"]).mean()
    df_PAS_stats_unified_across_subs_std = df_PAS_stats_unified.groupby(["subjectiveAwareness"]).std()
    df_PAS_stats_unified_across_subs_avg.to_csv(os.path.join(save_path, f"{PAS}_rate_across_subs_mean.csv"))
    df_PAS_stats_unified_across_subs_std.to_csv(os.path.join(save_path, f"{PAS}_rate_across_subs_std.csv"))
    return df_PAS_stats_unified


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
            df = all_subs_df[all_subs_df[parse_data_files.SUBJ_ANS] == rating]  # PAS = rating trials
        else:  # this is for 2-3-4 collapsed
            df = all_subs_df[all_subs_df[parse_data_files.SUBJ_ANS] != 1]
        pcntgs_per_sub = calculate_pcnt_correct(df, filler_df, parse_data_files.OBJ_ANS, rating)
        result_list.append(pcntgs_per_sub)

    result_list_by_sub = [df.set_index(SUB) for df in result_list]  # set subject as index to concatenate by
    result_df = pd.concat(result_list_by_sub, axis=1)
    result_df = result_df.loc[:, ~result_df.columns.duplicated()]
    result_df.reset_index(inplace=True)
    result_df.to_csv(os.path.join(save_path, f"objective_correct_per_pas.csv"))

    plot_res_list = list()
    for i in [1, 234]:  # JUST FOR 1 AND 2-4 COLLAPSED
        small_df = result_df[["Subject", f"pcntCorrectInObj_{i}"]]
        small_df["subjectiveAwareness"] = i if i==1 else 2
        small_df.rename(columns={f"pcntCorrectInObj_{i}": "pcntCorrectInObj"}, inplace=True)
        small_df = small_df.sort_values(by=SUB).reset_index()
        plot_res_list.append(small_df)
    df_for_plot = pd.concat(plot_res_list)
    plotter.plot_raincloud(df_for_plot, "subjectiveAwareness", "pcntCorrectInObj", "Overall Performance in Objective Task Per Visibility",
                           "Subjective Awareness Rating", "% Correct in Objective Task",
                           save_path=save_path, save_name="PAS_pcnt_correct_in_objective",
                           x_col_color_order=[["#559CAD", "#559CAD", "#559CAD", "#559CAD"]],
                           y_tick_interval=25, y_tick_min=0, y_tick_max=100,
                           x_axis_names=["1", "2-4"], y_tick_names=None, group_col_name=None,
                           group_name_mapping=None, x_values=[1, 2], add_horizontal_line=25)

    return result_df


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
    for i in [1, 234]:  # JUST FOR 1 AND 2-4 COLLAPSED
        small_df = result_df[["Subject", f"ValenceCorrectProportion_{i}"]]
        small_df["subjectiveAwareness"] = i if i == 1 else 2
        small_df.rename(columns={f"ValenceCorrectProportion_{i}": "ValenceCorrectProportion"}, inplace=True)
        plot_res_list.append(small_df)
    df_for_plot = pd.concat(plot_res_list)
    df_for_plot.fillna(0, inplace=True)
    plotter.plot_raincloud(df_for_plot, "subjectiveAwareness", "ValenceCorrectProportion", "Overall Performance in Valence Task Per Visibility",
                           "Subjective Awareness Rating", "% Correct in Valence Task",
                           save_path=save_path, save_name="PAS_pcnt_correct_in_valence",
                           x_col_color_order=[["#559CAD", "#559CAD", "#559CAD", "#559CAD"]],
                           y_tick_interval=25, y_tick_min=0, y_tick_max=100,
                           x_axis_names=["1", "2-4"], y_tick_names=None, group_col_name=None,
                           group_name_mapping=None, x_values=[1, 2], add_horizontal_line=50)
    return result_df


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

    # ANALYSIS 1 & 2:
    # DID THE MANIPULATION WORK? PAS COMPARISON BETWEEN AT AND UAT
    # DID THE VISIBILITY CHANGE BETWEEN AVERSIVE AND NEUTRAL STIMULI?
    pas_comparison_cond(all_subs_df, beh_output_path)

    # ANALYSIS 3: WERE SUBJECTS AT CHANCE IN THE OBJECTIVE TASK WHEN VISIBILITY WAS 1?
    pas_obj_perf(all_subs_df, beh_output_path)

    # ANALYSIS 4: WERE SUBJECTS AT CHANCE IN THE VALENCE TASK WHEN VISIBILITY WAS 1
    pas_val_perf(all_subs_df, beh_output_path)

    # ANALYSIS 5: WERE SUBJECTS BETTER IN OBJECTIVE TASK FOR AVERSIVE / NEUTRAL STIMULI?
    val_obj_perf(all_subs_df, beh_output_path)

    return


def beh_pilot_plots(data_path, save_path):
    data = pd.read_csv(data_path)

    plotter.plot_raincloud(data, "Valence_numeric", "Average Valence Rating",
                           "Stimulus Valence Rating",
                           "Stimulus Valence", "Average Valence Rating",
                           save_path=save_path, save_name="avg_valence_ratings",
                           x_col_color_order=[["#22395B", "#88292F"]],
                           y_tick_interval=1, y_tick_min=1, y_tick_max=5,
                           x_axis_names=["Neutral", "Aversive"],
                           y_tick_names=["1: Unhappy", "2", "3", "4", "5: Happy"], group_col_name=None,
                           group_name_mapping=None, x_values=[0, 1])

    plotter.plot_raincloud(data, "Valence_numeric", "Average Arousal Rating",
                           "Stimulus Arousal Rating",
                           "Stimulus Valence", "Average Arousal Rating",
                           save_path=save_path, save_name="avg_arousal_ratings",
                           x_col_color_order=[["#22395B", "#88292F"]],
                           y_tick_interval=1, y_tick_min=1, y_tick_max=5,
                           x_axis_names=["Neutral", "Aversive"],
                           y_tick_names=["1: Calm", "2", "3", "4", "5: Excited"], group_col_name=None,
                           group_name_mapping=None, x_values=[0, 1])
    return
