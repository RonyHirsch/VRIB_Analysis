import pandas as pd
import numpy as np
import os
import re
import math
import seaborn as sns
from pathlib import Path
import parse_data_files
import plotter
import stats_module

BEH = "BEH"
ET = "ET"
MEAN = "Average"
STD = "STD"
MIN = "Min"
MAX = "Max"
CNT = "Count"
SUB = "Subject"
REPLAY_TRIAL = 40
NO_REPLAY = "no_replay"
NO_GAME = "no_game"
AVERSIVE_IND = [x for x in range(0, 20)]
AVERSIVE_VALENCE = 1
AVERSIVE = "Aversive"
NEUTRAL_IND = [x for x in range(20, 40)]
NEUTRAL_VALENCE = 0
NEUTRAL = "Neutral"
VALENCE_MAPPING = {str(AVERSIVE_VALENCE): AVERSIVE, str(NEUTRAL_VALENCE): NEUTRAL}


def param_over_time(param, time, df, save_path, plot_color="tab:blue", y_tick_skip=1, ymin=None, ymax=None,
                    test="pearson", y_tick_labels=None):
    # Plot PARAM x TIME
    x_name = re.sub('([A-Z])', r' \1', time).title()
    y_name = re.sub('([A-Z])', r' \1', param).title()
    y_save_name = re.sub(r'(?<!^)(?=[A-Z])', '_', param).lower()
    plotter.line_plot(df=df, x_col_name=time, y_col_names=[param], plot_title=f"{y_name} Over Time",
                      plot_x_name=x_name, plot_y_name=y_name, save_path=save_path,  y_tick_interval=y_tick_skip,
                      save_name=f"{y_save_name}_over_time", colors={param: plot_color}, y_tick_min=ymin,
                      y_tick_max=ymax, show_legend=False, y_tick_labels=y_tick_labels)

    # Calculate correlation between them
    stats_module.correlation(data=df, col1=param, col2=time, corr_method=test, save_path=save_path)

    # Calculate correlation separately for TASK trials (1-40) and REPLAY trials (41-50)
    if param == parse_data_files.SUBJ_ANS:
        no_replay = df[df[parse_data_files.TRIAL_NUMBER] < REPLAY_TRIAL]
        save_name = re.sub(r'(?<!^)(?=[A-Z])', '_', param).lower() + "_" + \
                    re.sub(r'(?<!^)(?=[A-Z])', '_', time).lower() + "_" + NO_REPLAY

        no_game = df[df[parse_data_files.TRIAL_NUMBER] >= REPLAY_TRIAL]
        save_name = re.sub(r'(?<!^)(?=[A-Z])', '_', param).lower() + "_" + \
                    re.sub(r'(?<!^)(?=[A-Z])', '_', time).lower() + "_" + NO_GAME
        stats_module.correlation(data=no_game, col1=param, col2=time, corr_method=test, save_path=save_path,
                                 save_name=save_name)
    return


def beh_over_time(data: pd.DataFrame, save_path):
    """
    Manager function for analyses of behavioral metrics over time (with the progress of trials).
    The metrics are the ones in cols_over_time:
    - parse_data_files.SUBJ_ANS = the PAS (subjective awareness rating) score given to the stimulus
    - parse_data_files.OBJ_ANS = whether the subject was correct in the objective task of picking the right stimulus
    - parse_data_files.TRIAL_MONEY = the accumulated money (total) up until (and including) that trial
    :param sub_dfs:
    :param save_path:
    :return:
    """
    col_time = parse_data_files.TRIAL_NUMBER  # the colum indicating the trial number
    cols_over_time = {parse_data_files.SUBJ_ANS: ["spearman", 1, ["1", "2", "3", "4"]],
                      parse_data_files.TRIAL_MONEY: ["pearson", 5, None]}
    for col in cols_over_time:
        param_over_time(param=col, time=col_time, df=data, save_path=save_path, test=cols_over_time[col][0],
                        y_tick_skip=cols_over_time[col][1], y_tick_labels=cols_over_time[col][2])
    return


def param_across_something(param, xcol, df, save_path, y_tick_skip=10, ymin=None, ymax=None, y_tick_names=None,
                     x_ticks_names=["1", "2", "3", "4"], corr_method="pearson", plot_by_group=None,
                     group_names_dict=None, group_shades=None, save_name_prefix=""):
    # Plot PARAM x SOMETHING
    x_name = re.sub('([A-Z])', r' \1', xcol).title()
    y_name = re.sub('([A-Z])', r' \1', param).title()
    y_save_name = re.sub(r'(?<!^)(?=[A-Z])', '_', param).lower()
    x_save_name = re.sub(r'(?<!^)(?=[A-Z])', '_', xcol).lower()
    if plot_by_group is None:
        group_shades = [sns.color_palette("Blues", len(df[xcol].unique()))] if group_shades is None else group_shades
        plotter.plot_raincloud(df=df, x_col_name=xcol, y_col_name=param, plot_title=f"{y_name} Per {x_name} Rating",
                               plot_x_name=x_name, plot_y_name=y_name, y_tick_names=y_tick_names,  save_path=save_path,
                               save_name=f"{save_name_prefix}{y_save_name}_over_{x_save_name}",
                               x_col_color_order=group_shades,
                               y_tick_interval=y_tick_skip,
                               x_axis_names=x_ticks_names, y_tick_min=ymin, y_tick_max=ymax)

    else:
        data = df[df[plot_by_group].notna()]  # we are interested in groups of values, if there is a "nan" group it is noninteresting
        if group_shades is None:
            shades = ["afmhot_r", "ocean_r", "copper_r", "BuPu", "summer_r"]
            colors = [sns.color_palette(shade, len(data[xcol].unique())+2)[1:-1] for shade in shades[:len(data[plot_by_group].unique())]]
        else:
            shades = group_shades
            colors = [sns.color_palette(shade, len(data[xcol].unique()) + 2)[1:-1] for shade in shades]
        group_name = re.sub(r'(?<!^)(?=[A-Z])', '_', plot_by_group).lower()
        plotter.plot_raincloud(df=data, x_col_name=xcol, y_col_name=param,
                               group_col_name=plot_by_group, group_name_mapping=group_names_dict,
                               plot_title=f"{y_name} Per {x_name}",
                               plot_x_name=x_name, plot_y_name=y_name, save_path=save_path,
                               save_name=f"{save_name_prefix}{y_save_name}_over_{x_save_name}_by_{group_name}",
                               x_col_color_order=colors,
                               y_tick_interval=y_tick_skip,
                               x_axis_names=x_ticks_names, y_tick_min=ymin, y_tick_max=ymax)

    # Calculate correlation between them: PAS is rank, so Spearman
    if corr_method is not None:
        corr = stats_module.correlation(data=df, col1=param, col2=xcol, corr_method=corr_method, save_path=save_path)
    return


def beh_over_pas(data: pd.DataFrame, save_path):
    col_pas = parse_data_files.SUBJ_ANS

    # How many trials per rating, separately for unattended and attended conditions, across subjects
    unattended = data[data[parse_data_files.TRIAL_NUMBER] < REPLAY_TRIAL]
    attended = data[data[parse_data_files.TRIAL_NUMBER] >= REPLAY_TRIAL]

    unattended_dict = {0: list(), 1: list(), 2: list(), 3: list()}
    attended_dict = {0: list(), 1: list(), 2: list(), 3: list()}
    sub_list = data[SUB].unique()
    for sub in sub_list:
        sub_unattended = unattended[unattended[SUB] == sub]
        sub_attended = attended[attended[SUB] == sub]
        unattended_pas = sub_unattended[col_pas].value_counts().rename_axis('PAS_rating').reset_index(name='count')
        attended_pas = sub_attended[col_pas].value_counts().rename_axis('PAS_rating').reset_index(name='count')
        for pas in [0, 1, 2, 3]:
            # unattended
            if pas in unattended_pas['PAS_rating'].unique():
                unattended_dict[pas].append(unattended_pas[unattended_pas['PAS_rating'] == pas]['count'].iloc[0])
            else:
                unattended_dict[pas].append(0)
            # attended
            if pas in attended_pas['PAS_rating'].unique():
                attended_dict[pas].append(attended_pas[attended_pas['PAS_rating'] == pas]['count'].iloc[0])
            else:
                attended_dict[pas].append(0)

    unattended_df = pd.DataFrame.from_dict(unattended_dict)
    unattended_df[SUB] = sub_list
    attended_df = pd.DataFrame.from_dict(attended_dict)
    attended_df[SUB] = sub_list

    unattended_df.to_csv(os.path.join(save_path, "pas_per_sub_unattended.csv"))
    attended_df.to_csv(os.path.join(save_path, "pas_per_sub_attended.csv"))

    # For each PAS rating, what is the % correct in the OBJECTIVE task
    unattended_obj_per_pas = unattended.groupby([SUB, parse_data_files.SUBJ_ANS]).mean()[parse_data_files.OBJ_ANS].reset_index()
    counts = unattended.groupby([SUB, parse_data_files.SUBJ_ANS]).count()[parse_data_files.OBJ_ANS].reset_index()
    result = pd.merge(unattended_obj_per_pas, counts, on=[SUB, parse_data_files.SUBJ_ANS])
    result.to_csv(os.path.join(save_path, "pas_obj_correct_per_sub_unattended.csv"))

    attended_obj_per_pas = attended.groupby([SUB, parse_data_files.SUBJ_ANS]).mean()[parse_data_files.OBJ_ANS].reset_index()
    counts = attended.groupby([SUB, parse_data_files.SUBJ_ANS]).count()[parse_data_files.OBJ_ANS].reset_index()
    result = pd.merge(attended_obj_per_pas, counts, on=[SUB, parse_data_files.SUBJ_ANS])
    result.to_csv(os.path.join(save_path, "pas_obj_correct_per_sub_attended.csv"))

    # For each PAS rating, what is the % correct in the OBJECTIVE task per valence
    unattended_valence_0 = unattended[unattended[parse_data_files.TRIAL_STIM_VAL] == 0]
    unattended_obj_per_pas_valence_0 = unattended_valence_0.groupby([SUB, parse_data_files.SUBJ_ANS]).mean()[parse_data_files.OBJ_ANS].reset_index()
    counts = unattended_valence_0.groupby([SUB, parse_data_files.SUBJ_ANS]).count()[parse_data_files.OBJ_ANS].reset_index()
    result = pd.merge(unattended_obj_per_pas_valence_0, counts, on=[SUB, parse_data_files.SUBJ_ANS])
    result.to_csv(os.path.join(save_path, "pas_obj_correct_per_sub_unattended_valence0.csv"))

    unattended_valence_1 = unattended[unattended[parse_data_files.TRIAL_STIM_VAL] == 1]
    unattended_obj_per_pas_valence_1 = unattended_valence_1.groupby([SUB, parse_data_files.SUBJ_ANS]).mean()[parse_data_files.OBJ_ANS].reset_index()
    counts = unattended_valence_1.groupby([SUB, parse_data_files.SUBJ_ANS]).count()[parse_data_files.OBJ_ANS].reset_index()
    result = pd.merge(unattended_obj_per_pas_valence_1, counts, on=[SUB, parse_data_files.SUBJ_ANS])
    result.to_csv(os.path.join(save_path, "pas_obj_correct_per_sub_unattended_valence1.csv"))


    # For each PAS rating, what is the % correct in the VALENCE task
    unattended_VALENCE_per_pas = unattended.groupby([SUB, parse_data_files.SUBJ_ANS]).mean()[parse_data_files.VAL_ANS_CORRECT].reset_index()
    counts = unattended.groupby([SUB, parse_data_files.SUBJ_ANS]).count()[parse_data_files.VAL_ANS_CORRECT].reset_index()
    result = pd.merge(unattended_VALENCE_per_pas, counts, on=[SUB, parse_data_files.SUBJ_ANS])
    result.to_csv(os.path.join(save_path, "pas_valence_correct_per_sub_unattended.csv"))

    attended_VALENCE_per_pas = attended.groupby([SUB, parse_data_files.SUBJ_ANS]).mean()[parse_data_files.VAL_ANS_CORRECT].reset_index()
    counts = attended.groupby([SUB, parse_data_files.SUBJ_ANS]).count()[parse_data_files.VAL_ANS_CORRECT].reset_index()
    result = pd.merge(attended_VALENCE_per_pas, counts, on=[SUB, parse_data_files.SUBJ_ANS])
    result.to_csv(os.path.join(save_path, "pas_valence_correct_per_sub_attended.csv"))

    # For each PAS rating, what is the % correct in the VALENCE task PER VALENCE
    unattended_valence_0 = unattended[unattended[parse_data_files.TRIAL_STIM_VAL] == 0]
    unattended_VALENCE_per_pas_0 = unattended_valence_0.groupby([SUB, parse_data_files.SUBJ_ANS]).mean()[parse_data_files.VAL_ANS_CORRECT].reset_index()
    counts = unattended_valence_0.groupby([SUB, parse_data_files.SUBJ_ANS]).count()[parse_data_files.VAL_ANS_CORRECT].reset_index()
    result = pd.merge(unattended_VALENCE_per_pas_0, counts, on=[SUB, parse_data_files.SUBJ_ANS])
    result.to_csv(os.path.join(save_path, "pas_valence_correct_per_sub_unattended_VALENCE0.csv"))

    unattended_valence_1 = unattended[unattended[parse_data_files.TRIAL_STIM_VAL] == 1]
    unattended_VALENCE_per_pas_1 = unattended_valence_1.groupby([SUB, parse_data_files.SUBJ_ANS]).mean()[parse_data_files.VAL_ANS_CORRECT].reset_index()
    counts = unattended_valence_1.groupby([SUB, parse_data_files.SUBJ_ANS]).count()[parse_data_files.VAL_ANS_CORRECT].reset_index()
    result = pd.merge(unattended_VALENCE_per_pas_1, counts, on=[SUB, parse_data_files.SUBJ_ANS])
    result.to_csv(os.path.join(save_path, "pas_valence_correct_per_sub_unattended_VALENCE1.csv"))

    attended_valence_0 = attended[attended[parse_data_files.TRIAL_STIM_VAL] == 0]
    attended_VALENCE_per_pas_0 = attended_valence_0.groupby([SUB, parse_data_files.SUBJ_ANS]).mean()[parse_data_files.VAL_ANS_CORRECT].reset_index()
    attended_VALENCE_per_pas_0.to_csv(os.path.join(save_path, "pas_valence_correct_per_sub_attended_VALENCE0.csv"))

    attended_valence_1 = attended[attended[parse_data_files.TRIAL_STIM_VAL] == 1]
    attended_VALENCE_per_pas_1 = attended_valence_1.groupby([SUB, parse_data_files.SUBJ_ANS]).mean()[
        parse_data_files.VAL_ANS_CORRECT].reset_index()
    attended_VALENCE_per_pas_1.to_csv(os.path.join(save_path, "pas_valence_correct_per_sub_attended_VALENCE1.csv"))


    # For each VALENCE, what is the Valence rating count
    unattended_pas_per_valence = unattended.groupby([SUB, parse_data_files.TRIAL_STIM_VAL, parse_data_files.SUBJ_ANS]).count().reset_index().iloc[:, :4]
    unattended_pas_per_valence.to_csv(os.path.join(save_path, "valence_per_pas_unattended.csv"))


    #cols_over_pas = {parse_data_files.CLUES_TAKEN: [parse_data_files.BEE_CORRECT, ["afmhot_r", "summer_r"]],
    #                 parse_data_files.VAL_ANS_CORRECT: [parse_data_files.OBJ_ANS, ["copper_r", "ocean_r"]],
    #                 parse_data_files.OBJ_ANS: [None, None]}
    #for col in cols_over_pas:
    #    param_across_something(param=col, xcol=col_pas, df=data, save_path=save_path, y_tick_skip=1,
    #                           corr_method="spearman", plot_by_group=cols_over_pas[col][0],
    #                           group_shades=cols_over_pas[col][1])
    return


def param_across_obj(param, obj, df, save_path, y_tick_skip=1, ymin=None, ymax=None,
                     x_tick_names=["False", "True"], y_tick_names=["1", "2", "3", "4"]):
    # Plot PARAM x OBJECTIVE performance (True/False)
    x_name = re.sub('([A-Z])', r' \1', obj).title()
    y_name = re.sub('([A-Z])', r' \1', param).title()
    y_save_name = re.sub(r'(?<!^)(?=[A-Z])', '_', param).lower()
    plotter.plot_raincloud(df=df, x_col_name=obj, y_col_name=param, plot_title=f"{y_name} Per {x_name}",
                           plot_x_name=x_name, plot_y_name=y_name, save_path=save_path,
                           save_name=f"{y_save_name}_over_obj",
                           x_col_color_order=[sns.color_palette("Blues", len(df[obj].unique()))],
                           y_tick_interval=y_tick_skip, x_axis_names=x_tick_names,
                           y_tick_min=ymin, y_tick_max=ymax, y_tick_names=y_tick_names)

    # STAT: ?????????????????
    return


def beh_over_obj(data: pd.DataFrame, save_path):
    col_obj = parse_data_files.OBJ_ANS
    cols_over_obj = {parse_data_files.SUBJ_ANS: [1, ["1", "2", "3", "4"], None, None, None],
                     parse_data_files.OBJ_RT: [1000, None, parse_data_files.TRIAL_STIM_VAL, VALENCE_MAPPING, ["copper_r", "ocean_r"]]}
    for col in cols_over_obj:
        param_across_obj(param=col, obj=col_obj, df=data, save_path=save_path,
                         y_tick_skip=cols_over_obj[col][0], y_tick_names=cols_over_obj[col][1],
                         group_col_name=cols_over_obj[col][2], group_name_mapping=cols_over_obj[col][3], group_shades=cols_over_obj[col][4])
    return


def gaze_per_pas(sub_df, save_path, cols_over_pas, data_name, data_colors=None):
    col_pas = parse_data_files.SUBJ_ANS
    ymin = 0  # gaze duration cannot be negative
    y_max_val = max([max(sub_df[cols_over_pas[c]]) for c in cols_over_pas])
    ymax = int(math.ceil(y_max_val / 10.0)) * 10  # round up to the nearest 10
    y_tick_skip = 1 if ymax-ymin < 11 else 5

    x_name = re.sub('([A-Z])', r' \1', col_pas).title()
    x_save_name = re.sub(r'(?<!^)(?=[A-Z])', '_', col_pas).lower()
    x_axis = ["1", "2", "3", "4"]
    x_axis_vals = [0, 1, 2, 3]

    prefix_name = re.sub('([A-Z])', r' \1', data_name).title()

    for colname in cols_over_pas:
        col = cols_over_pas[colname]
        y_name = "Gaze Duration (Seconds)"
        y_save_name = re.sub(r'(?<!^)(?=[A-Z])', '_', col).lower()
        # overall
        plotter.plot_raincloud(df=sub_df, x_col_name=col_pas, y_col_name=col, plot_title=f"{prefix_name} {y_name} Per {x_name} : {colname}",
                               plot_x_name=col_pas, plot_y_name=col,
                               save_path=save_path, save_name=f"{data_name}{y_save_name}_over_{x_save_name}",
                               x_col_color_order=data_colors, y_tick_interval=y_tick_skip, y_tick_min=ymin, y_tick_max=ymax,
                               x_axis_names=x_axis, y_tick_names=None, group_col_name=None, group_name_mapping=None,
                               x_values=x_axis_vals)

        plotter.plot_raincloud(df=sub_df, x_col_name=col_pas, y_col_name=col,
                               plot_title=f"{prefix_name} {y_name} Per {x_name} : {colname}",
                               plot_x_name=col_pas, plot_y_name=col,
                               save_path=save_path, save_name=f"{data_name}{y_save_name}_over_{x_save_name}_VALENCE",
                               y_tick_interval=y_tick_skip, y_tick_min=ymin,
                               y_tick_max=ymax,
                               x_axis_names=x_axis, y_tick_names=None, group_col_name=parse_data_files.TRIAL_STIM_VAL,
                               group_name_mapping=VALENCE_MAPPING,
                               x_col_color_order=[sns.color_palette("copper_r", 4), sns.color_palette("ocean_r", 4)],
                               x_values=x_axis_vals)


        """
        param_across_something(param=col, xcol=col_pas, df=sub_df, save_path=save_path, y_tick_skip=y_tick_skip,
                         corr_method=None, ymin=ymin, ymax=ymax, save_name_prefix=data_name, group_shades=data_colors)

        # split to groups - AVERSIVE V NEUTRAL
        param_across_something(param=col, xcol=col_pas, df=sub_df, save_path=save_path, y_tick_skip=y_tick_skip,
                         corr_method=None, ymin=ymin, ymax=ymax,
                         plot_by_group=parse_data_files.TRIAL_STIM_VAL, group_names_dict=VALENCE_MAPPING,
                         group_shades=["copper_r", "ocean_r"], save_name_prefix=data_name)

        # split to groups - OBJECTIVE CORRECT V INCORRECT
        param_across_something(param=col, xcol=col_pas, df=sub_df, save_path=save_path, y_tick_skip=y_tick_skip,
                         corr_method=None, ymin=ymin, ymax=ymax,
                         plot_by_group=parse_data_files.OBJ_ANS, group_names_dict=None,
                         group_shades=["afmhot_r", "summer_r"], save_name_prefix=data_name)
        """

    return


def identify_valence_outliers(real_valence, val_judge_df):
    rating_match_dict = dict()
    for row in real_valence.itertuples():
        stim_id = row.stimPicName
        stim_val = row.stimValence
        stim_rating_df = val_judge_df[val_judge_df[parse_data_files.TRIAL_STIM_NAME] == stim_id]  # a df with <=2 rows (one for each valence rating)
        total_ratings = stim_rating_df["count"].sum()
        try:
            stim_correct_val_rating = (stim_rating_df[stim_rating_df[parse_data_files.VAL_ANS] == stim_val][["count"]] / total_ratings).iloc[0,0]
        except Exception:
            print(f"All ratings of image {stim_id} (valence: {stim_val}) were the opposite rating: {stim_rating_df}")
            stim_correct_val_rating = 0
        rating_match_dict[stim_id] = stim_correct_val_rating

    rating_match_df = pd.DataFrame.from_dict(rating_match_dict.items())
    rating_match_df.rename(columns={0: parse_data_files.TRIAL_STIM_NAME, 1: "pcnt valence rating matching real valence"},
                           inplace=True)
    rating_match_df = pd.merge(rating_match_df, real_valence, on=parse_data_files.TRIAL_STIM_NAME).sort_values(by=[parse_data_files.TRIAL_STIM_VAL, parse_data_files.TRIAL_STIM_NAME])
    outliers = list()
    for image in rating_match_dict:
        if rating_match_dict[image] < 0.5:
            outliers.append(image)
    return rating_match_df, outliers


def beh_over_stim(sub_df, save_path):
    # Analysis (descriptive): for every stimulus, how many 1/2/3/4 PAS ratings did it have
    stim_pas = pd.DataFrame({"count": sub_df.groupby([parse_data_files.TRIAL_STIM_NAME, parse_data_files.SUBJ_ANS]).size()}).reset_index()
    # Analysis (descriptive): for every stimulus, how many valence 1/0 ratings did it have
    stim_val_jdg = pd.DataFrame({"count": sub_df.groupby([parse_data_files.TRIAL_STIM_NAME, parse_data_files.VAL_ANS]).size()}).reset_index()
    stim_real_val = sub_df[[parse_data_files.TRIAL_STIM_NAME, parse_data_files.TRIAL_STIM_VAL]].drop_duplicates()
    stim_pas.to_csv(os.path.join(save_path, "stim_pas_ratings.csv"))
    stim_val_jdg.to_csv(os.path.join(save_path, "stim_valence_rating.csv"))
    stim_real_val.to_csv(os.path.join(save_path, "stim_valence_real.csv"))

    # Identify outliers - BASED ON ATTENDED CONDITION ONLY
    # Analysis (descriptive): for every stimulus, how many valence 1/0 ratings did it have
    attended = sub_df[sub_df[parse_data_files.TRIAL_NUMBER] >= REPLAY_TRIAL]
    attended_stim_pas = pd.DataFrame({"count": attended.groupby([parse_data_files.TRIAL_STIM_NAME, parse_data_files.SUBJ_ANS]).size()}).reset_index()
    attended_stim_val_jdg = pd.DataFrame({"count": attended.groupby([parse_data_files.TRIAL_STIM_NAME, parse_data_files.VAL_ANS]).size()}).reset_index()
    attended_stim_val_sumtrials = pd.DataFrame({"count": attended.groupby([parse_data_files.TRIAL_STIM_NAME]).size()}).reset_index()
    attended_stim_real_val = attended[[parse_data_files.TRIAL_STIM_NAME, parse_data_files.TRIAL_STIM_VAL]].drop_duplicates()
    attended_stim_pas.to_csv(os.path.join(save_path, "OUTLIER_stim_pas_ratings_attended.csv"))
    attended_stim_val_jdg.to_csv(os.path.join(save_path, "OUTLIER_stim_valence_rating_attended.csv"))
    attended_stim_real_val.to_csv(os.path.join(save_path, "OUTLIER_stim_valence_realattended.csv"))
    attended_stim_val_sumtrials.to_csv(os.path.join(save_path, "OUTLIER_stim_valence_trial_count.csv"))

    rating_match_df, outlier_list = identify_valence_outliers(attended_stim_real_val, attended_stim_val_jdg)
    rating_match_df.to_csv(os.path.join(save_path, "OUTLIER_stim_valence_rating_match_real.csv"))
    print(f"{len(outlier_list)} outlier images found: {outlier_list}. These pictures will be removed from further analyses")

    # PLOT
    plotter.plot_bar(rating_match_df, x_col_name=parse_data_files.TRIAL_STIM_NAME,
             y_col_name="pcnt valence rating matching real valence",
             plot_title="Agreement between Actual and Rated Valence",
             plot_x_name="Picture", plot_y_name="% of agreement between rating and valence",
             save_path=save_path, save_name="valence_agreement", hue_col_name=parse_data_files.TRIAL_STIM_VAL,
             x_col_axis_order=rating_match_df[parse_data_files.TRIAL_STIM_NAME],
             conf_interval=None, palette=None, y_tick_interval=0.1, y_min_max=[0, 1],
                     y_tick_names=[f"{y}" for y in range(0, 101, 10)],
                     x_axis_names=None)

    # Drop outliers
    sub_df_no_outliers = sub_df[~(sub_df[parse_data_files.TRIAL_STIM_NAME].isin(outlier_list))]

    # Is stimulus VISIBILITY explained by stimulus VALENCE? Do people see more the aversive stimuli?
    # Mann-Whitney U test
    # IV = visibility (PAS), DV = valence (not the rating but the group it beloned to)
    # Take all the stimulus data as-is (not average per subject)

    # How many trials per rating, separately for unattended and attended conditions, across subjects
    unattended = sub_df_no_outliers[sub_df_no_outliers[parse_data_files.TRIAL_NUMBER] < REPLAY_TRIAL]
    attended = sub_df_no_outliers[sub_df_no_outliers[parse_data_files.TRIAL_NUMBER] >= REPLAY_TRIAL]

    unattended_1 = unattended[unattended[parse_data_files.TRIAL_STIM_VAL] == 1][parse_data_files.SUBJ_ANS].tolist()
    unattended_0 = unattended[unattended[parse_data_files.TRIAL_STIM_VAL] == 0][parse_data_files.SUBJ_ANS].tolist()
    max_len = max(len(unattended_0), len(unattended_1))
    unattended_0 += [np.nan] * (max_len - len(unattended_0))
    unattended_1 += [np.nan] * (max_len - len(unattended_1))
    unattended_ttest_df = pd.DataFrame()
    unattended_ttest_df["Aversive_PAS"] = unattended_1
    unattended_ttest_df["Neutral_PAS"] = unattended_0
    unattended_ttest_df.to_csv(os.path.join(save_path, "unattended_pas_1v0.csv"))
    stats_module.mann_whitney_test(data=unattended_ttest_df, col1="Aversive_PAS", col2="Neutral_PAS",
                                    save_path=save_path, save_name="unattended_pas_1v0")


    attended_1 = attended[attended[parse_data_files.TRIAL_STIM_VAL] == 1][parse_data_files.SUBJ_ANS].tolist()
    attended_0 = attended[attended[parse_data_files.TRIAL_STIM_VAL] == 0][parse_data_files.SUBJ_ANS].tolist()
    max_len = max(len(attended_0), len(attended_1))
    attended_0 += [np.nan] * (max_len - len(attended_0))
    attended_1 += [np.nan] * (max_len - len(attended_1))
    attended_ttest_df = pd.DataFrame()
    attended_ttest_df["Aversive_PAS"] = attended_1
    attended_ttest_df["Neutral_PAS"] = attended_0
    attended_ttest_df.to_csv(os.path.join(save_path, "attended_pas_1v0.csv"))
    stats_module.mann_whitney_test(data=attended_ttest_df, col1="Aversive_PAS", col2="Neutral_PAS",
                                    save_path=save_path, save_name="attended_pas_1v0")

    return sub_df_no_outliers, outlier_list


def behavioral_analysis(sub_df, save_path):
    # ANALYSIS 1: BEHAVIOR OVER TRIALS (TIME)
    beh_over_time(sub_df, save_path)

    # ANALYSIS 2: BEHAVIOR PER STIMULUS ID
    sub_df_no_outliers, outlier_list = beh_over_stim(sub_df, save_path)

    # ANALYSIS 3: BEHAVIOR PER VISIBILITY (PAS RATING)
    beh_over_pas(sub_df_no_outliers, save_path)

    # DEPRECATED ANALYSIS 4: BEHAVIOR PER OBJECTIVE PERFORMANCE ON STIMULUS
    #beh_over_obj(sub_df_no_outliers, save_path)

    return


def et_analysis(sub_df, save_path):
    # ANALYSIS 1: GAZE DURATION ACROSS SUBJECTIVE AWARENESS PER OBJECTIVE PERFORMANCE
    # PLOTS
    pas_ratings = 4
    # average gaze duration per intact/scrambled busstop
    unattended = sub_df[sub_df[parse_data_files.TRIAL_NUMBER] < REPLAY_TRIAL]
    attended = sub_df[sub_df[parse_data_files.TRIAL_NUMBER] >= REPLAY_TRIAL]
    data = {"ALL_": [sub_df, [sns.color_palette("Blues", pas_ratings)]],
            "unattended_": [unattended, [sns.color_palette("Purples", pas_ratings)]],
            "attended_": [attended, [sns.color_palette("viridis", pas_ratings)]]}
    for dname in data:
        gaze_per_pas(data[dname][0], save_path, cols_over_pas={"Intact": parse_data_files.BUSSTOP_GAZE_DUR_AVG_INTACT,
                                                            "Scrambled": parse_data_files.BUSSTOP_GAZE_DUR_AVG_SCRAMBLED},
                     data_name=dname, data_colors=data[dname][1])
        data[dname][0].to_csv(os.path.join(save_path, f"{dname}_avg_gaze_over_pas.csv"))


    # DEPRECATED sum of gaze duration (total) per intact/scrambled busstop
    #gaze_per_pas(sub_df, save_path, cols_over_pas=[parse_data_files.BUSSTOP_GAZE_DUR_INTACT,
    #                                               parse_data_files.BUSSTOP_GAZE_DUR_SCRAMBLED])

    return


def manage_analyses(data_path, save_path):
    # STEP 1: get all the subject dataframes
    sub_dfs = parse_data_files.extract_subject_data(data_path)
    all_subs_df = pd.concat([sub_dfs[sub] for sub in sub_dfs], keys=sub_dfs.keys(), names=[SUB, None]).reset_index(level=SUB)
    all_subs_df.to_csv(os.path.join(save_path, "raw_all_subs.csv"))

    # STEP 2: ANALYZE BEHAVIOR
    beh_path = os.path.join(save_path, BEH)
    Path(beh_path).mkdir(parents=True, exist_ok=True)
    behavioral_analysis(all_subs_df, beh_path)

    # STEP 3: ANALYZE ET
    et_path = os.path.join(save_path, ET)
    Path(et_path).mkdir(parents=True, exist_ok=True)
    et_analysis(all_subs_df, et_path)


if __name__ == "__main__":
    manage_analyses(data_path=r"C:\Users\ronyhirschhorn\Documents\TAU\VR\VRIB_DATA\data",
                             save_path=r"C:\Users\ronyhirschhorn\Documents\TAU\VR\VRIB_DATA\res")