import pandas as pd
import numpy as np
import os
import itertools
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import parse_data_files
import plotter
import exclusion_criteria

pd.options.mode.chained_assignment = None  # default='warn' see: https://stackoverflow.com/questions/20625582/how-to-deal-with-settingwithcopywarning-in-pandas

""" VRIB beh analysis manager

This module manages everything related to the processing of behavioral data towards analysis. Note that the statistical
analyses themselves (linear mixed models, t-tests etc) are not done here; the goal of this module is to output summary
data and plots, as well as aggregate data into a group dataframe that will be later analyzed with R (for linear models)
and JASP (for t-tests). 

@authors: RonyHirsch
"""


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

AXIS_SIZE = 19
TICK_SIZE = 17
LABEL_PAD = 8


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

    pas_xs = {1: 1, 2: 2, 3: 3, 4: 4}
    palette = {"Unattended": "#C33C54", "Attended": "#254E70"}
    plt.gcf()
    plt.figure()
    sns.reset_orig()

    for pas in pas_xs:
        df_pas = PAS_task_comp_df[PAS_task_comp_df['subjectiveAwareness'] == pas]
        for cond in list(palette.keys()):
            df_cond = df_pas[df_pas['condition'] == cond]
            if not df_cond.empty:  # if we even have data in this condition
                x_loc = pas_xs[pas]
                # so that conditions won't overlap
                if cond == "Unattended":
                    x_loc -= 0.05
                else:
                    x_loc += 0.05
                y_vals = df_cond['Average_PAS']
                # plot violin
                violin = plt.violinplot(y_vals, positions=[x_loc], widths=0.75, showmeans=True, showextrema=False, showmedians=False)
                # make it a half-violin plot (only to the LEFT of center)
                for b in violin['bodies']:
                    # get the center
                    m = np.mean(b.get_paths()[0].vertices[:, 0])
                    if cond == "Unattended":
                        # modify the paths to not go further right than the center
                        b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], -np.inf, m)
                    else:
                        # modify the paths to not go further left than the center
                        b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], m, np.inf)
                    b.set_color(palette[cond])

                # change the color of the mean lines (showmeans=True)
                violin['cmeans'].set_color("black")
                violin['cmeans'].set_linewidth(2)
                # control the length like before
                m = np.mean(violin['cmeans'].get_paths()[0].vertices[:, 0])
                if cond == "Unattended":
                    violin['cmeans'].get_paths()[0].vertices[:, 0] = np.clip(
                        violin['cmeans'].get_paths()[0].vertices[:, 0], -np.inf, m)
                else:
                    violin['cmeans'].get_paths()[0].vertices[:, 0] = np.clip(
                        violin['cmeans'].get_paths()[0].vertices[:, 0], m, np.inf)

                # then scatter
                if cond == "Unattended":
                    scat_x = (np.ones(len(y_vals)) * (x_loc - 0.2)) + (np.random.rand(len(y_vals)) * 0.2)
                else:
                    scat_x = (np.ones(len(y_vals)) * (x_loc + 0.025)) + (np.random.rand(len(y_vals)) * 0.25)
                plt.scatter(x=scat_x, y=y_vals, marker="o", s=50, color=palette[cond], alpha=0.6, edgecolor=palette[cond])

    # cosmetics
    plt.xticks([x for x in range(1, 5, 1)], fontsize=TICK_SIZE + 5)
    plt.yticks(np.arange(0, 101, step=25), fontsize=TICK_SIZE + 5)

    plt.title("PAS Rating Distribution", pad=LABEL_PAD + 5)
    plt.ylabel("Proportion of Trials (%)", fontsize=AXIS_SIZE + 5, labelpad=LABEL_PAD)
    plt.xlabel("PAS Rating", fontsize=AXIS_SIZE + 5, labelpad=LABEL_PAD)

    # The following two lines generate custom fake lines that will be used as legend entries:
    markers = [plt.Line2D([0, 0], [0, 0], color=palette[label], marker='o', linestyle='') for label in palette]
    new_labels = [label for label in palette]
    legend = plt.legend(markers, new_labels, title="Condition", markerscale=1, fontsize=TICK_SIZE + 2)
    plt.setp(legend.get_title(), fontsize=TICK_SIZE + 2)

    # Save plot
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(15, 12)
    plt.savefig(os.path.join(save_path, f"PAS_comp_task.svg"), format="svg", dpi=1000, bbox_inches='tight', pad_inches=0.01)
    del figure
    plt.close()

    # Save data (across subjects, per rating)
    PAS_task_comp_summary_df = pd.concat(PAS_task_comp_summary_list)
    PAS_task_comp_summary_df.to_csv(os.path.join(save_path, "PAS_comp_task_proportion_stats.csv"))

    return


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
    return


def time_analysis(df, save_path, y_col, y_col_name, save_name):
    df_uat = df[df[parse_data_files.CONDITION] == "Unattended"]  # clues matter only during the game-phase
    # avg per sub
    clues_per_sub = df_uat.groupby([SUB]).mean(numeric_only=True).reset_index()  # average within-subject
    clues_per_sub_avg = clues_per_sub[[SUB, y_col]]
    clues_per_sub_avg.to_csv(os.path.join(save_path, f"{save_name}_per_sub.csv"), index=False)
    # avg in time, ACROSS subjects
    clues_per_trial = df_uat.groupby([parse_data_files.TRIAL_NUMBER]).mean(numeric_only=True).reset_index()
    clues_per_trial_avg = clues_per_trial[[parse_data_files.TRIAL_NUMBER, y_col]]
    clues_per_trial_avg.rename({y_col: f"{y_col_name} Avg"}, axis=1, inplace=True)
    clues_per_trial_std = df_uat.groupby([parse_data_files.TRIAL_NUMBER]).std().reset_index()
    clues_per_trial_std = clues_per_trial_std[[parse_data_files.TRIAL_NUMBER, y_col]]
    clues_per_trial_std.rename({y_col: f"{y_col_name} SD"}, axis=1, inplace=True)
    clues_per_trial = pd.merge(clues_per_trial_avg, clues_per_trial_std, on=parse_data_files.TRIAL_NUMBER, how='outer')
    clues_per_trial.to_csv(os.path.join(save_path, f"{save_name}_per_trial.csv"), index=False)
    return clues_per_trial


def clues_analysis(df, save_path):
    clues_per_trial = time_analysis(df=df, save_path=os.path.join(save_path, parse_data_files.UAT),
                                    y_col=parse_data_files.CLUES_TAKEN, y_col_name="Clue", save_name="clues_avg")

    # PLOT
    uat_trials = df[df[parse_data_files.TRIAL_NUMBER] < 40]
    uat_trials = uat_trials.loc[:, [SUB, parse_data_files.TRIAL_NUMBER, parse_data_files.CLUES_TAKEN]]
    uat_trials.to_csv(os.path.join(save_path, parse_data_files.UAT, "clues_per_trial_sub.csv"), index=False)
    plt.clf()
    plt.figure()
    sns.reset_orig()
    subs = sorted(list(uat_trials[SUB].unique()))
    # individual lines' colors; change color to color="#CBD2D0" to have them all the same
    colormap = plt.cm.gist_rainbow  # choose colormap: http://matplotlib.org/1.2.1/examples/pylab_examples/show_colormaps.html
    colors = [colormap(i) for i in np.linspace(0, 1, len(subs))]
    for i in range(len(subs)):  # plot individual data lines
        sub = subs[i]
        sub_trials = uat_trials[uat_trials[SUB] == sub]
        sns.lineplot(x=parse_data_files.TRIAL_NUMBER, y=parse_data_files.CLUES_TAKEN, data=sub_trials, color=colors[i], label="", linewidth=1, alpha=0.8)
    # now plot average line
    sns.lineplot(x=parse_data_files.TRIAL_NUMBER, y="Clue Avg", data=clues_per_trial, color="#4C5454", label="", linewidth=4)
    # cosmetics
    plt.xticks(fontsize=plotter.F_AXES_NAME)
    plt.yticks(np.arange(min(uat_trials[parse_data_files.CLUES_TAKEN]), max(uat_trials[parse_data_files.CLUES_TAKEN])+1, 1.0), fontsize=plotter.F_AXES_NAME)
    plt.title("Clues Per Trial", fontsize=plotter.F_TITLE, pad=plotter.LABELPAD)
    plt.xlabel("Trial Number", fontsize=plotter.F_AXES_TITLE, labelpad=plotter.LABELPAD)
    plt.ylabel("Clues Taken", fontsize=plotter.F_AXES_TITLE, labelpad=plotter.LABELPAD)
    plt.legend().remove()
    # save
    figure = plt.gcf()  # get current figure
    plt.savefig(os.path.join(save_path, parse_data_files.UAT, f"{plotter.LINE}_clues_avg_per_trial.png"), dpi=plotter.DPI, bbox_inches='tight')
    del figure
    plt.clf()
    plt.cla()
    plt.close()

    return


def trial_score_analysis(df, save_path):
    perf_per_trial = time_analysis(df=df, save_path=os.path.join(save_path, parse_data_files.UAT),
                                   y_col=parse_data_files.TRIAL_MONEY, y_col_name="Score", save_name="score_avg")

    # PLOT
    uat_trials = df[df[parse_data_files.TRIAL_NUMBER] < 40]
    plt.clf()
    plt.figure()
    sns.reset_orig()
    subs = sorted(list(uat_trials[SUB].unique()))
    # individual lines' colors; change color to color="#CBD2D0" to have them all the same
    colormap = plt.cm.gist_rainbow  # choose colormap: http://matplotlib.org/1.2.1/examples/pylab_examples/show_colormaps.html
    colors = [colormap(i) for i in np.linspace(0, 1, len(subs))]
    for i in range(len(subs)):  # plot individual data lines
        sub = subs[i]
        sub_trials = uat_trials[uat_trials[SUB] == sub]
        sns.lineplot(x=parse_data_files.TRIAL_NUMBER, y=parse_data_files.TRIAL_MONEY, data=sub_trials, color=colors[i], label="", linewidth=1, alpha=0.8)
    # now plot average line
    sns.lineplot(x=parse_data_files.TRIAL_NUMBER, y="Score Avg", data=perf_per_trial, color="#4C5454", label="", linewidth=4)
    # cosmetics
    plt.xticks(fontsize=plotter.F_AXES_NAME)
    plt.title("Score Per Trial", fontsize=plotter.F_TITLE, pad=plotter.LABELPAD)
    plt.xlabel("Trial Number", fontsize=plotter.F_AXES_TITLE, labelpad=plotter.LABELPAD)
    plt.ylabel("Total Score", fontsize=plotter.F_AXES_TITLE, labelpad=plotter.LABELPAD)
    plt.legend().remove()
    # save
    figure = plt.gcf()  # get current figure
    plt.savefig(os.path.join(save_path, parse_data_files.UAT, f"{plotter.LINE}_score_avg_per_trial.png"), dpi=plotter.DPI, bbox_inches='tight')
    del figure
    plt.clf()
    plt.cla()
    plt.close()

    """ DEPRECATED
    # fit a linear regression model to the data
    X = perf_per_trial[parse_data_files.TRIAL_NUMBER]
    y = perf_per_trial["Score Avg"]
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()  # ordinary least squares linear regression model
    # regression summary
    result = model.summary().as_csv()
    print("-------------------------- SCORE MODEL --------------------------")
    print(result)
    with open(os.path.join(save_path, parse_data_files.UAT, "score_per_trial_model.csv"), 'w') as f:
        f.write(result)
    # plot
    plotter.regression_plot(X=perf_per_trial[parse_data_files.TRIAL_NUMBER], y=perf_per_trial["Score Avg"],
                            y_pred=model.predict(), x_axis="Trial Number", y_axis="Score in Trial",
                            title="Regression Plot: Score per Trial", scatter_color="#521945",
                            save_path=os.path.join(save_path, parse_data_files.UAT), save_name="score_per_trial_model")
    """
    return


def val_obj_rt(all_subs_df, beh_output_path):
    df_agg = all_subs_df.groupby([SUB, parse_data_files.SUBJ_ANS, parse_data_files.TRIAL_STIM_VAL]).mean().reset_index()
    df_agg = df_agg.loc[:, [SUB, parse_data_files.SUBJ_ANS, parse_data_files.TRIAL_STIM_VAL, parse_data_files.OBJ_RT]]

    for pas in [1, 4]:
        df = df_agg[df_agg[parse_data_files.SUBJ_ANS] == pas]

        df_aversive = df[df[parse_data_files.TRIAL_STIM_VAL] == 1]
        df_aversive.drop(columns=[parse_data_files.TRIAL_STIM_VAL, parse_data_files.SUBJ_ANS], inplace=True)
        df_aversive.rename({parse_data_files.OBJ_RT: f"{parse_data_files.OBJ_RT}_pas{pas}_aversive"}, axis=1, inplace=True)

        df_neutral = df[df[parse_data_files.TRIAL_STIM_VAL] != 1]
        df_neutral.drop(columns=[parse_data_files.TRIAL_STIM_VAL, parse_data_files.SUBJ_ANS], inplace=True)
        df_neutral.rename({parse_data_files.OBJ_RT: f"{parse_data_files.OBJ_RT}_pas{pas}_neutral"}, axis=1, inplace=True)
        df_result = pd.merge(df_aversive, df_neutral, on=SUB)
        df_result.to_csv(os.path.join(beh_output_path, f"agg_per_sub_vis_val_pas{pas}.csv"), index=False)
    return


def intact_pas_soa(data, save_path):
    """
    Plot a raincloud where each dot is equivalent of a single trial, and the plot is of PAS rating x time between
    the last intact stimulus and PAS.
    :param data:
    :param save_path:
    :return:
    """
    pas_xs = {1: 1, 2: 2, 3: 3, 4: 4}
    palette = {"Unattended": "#C33C54", "Attended": "#254E70"}
    plt.gcf()
    plt.figure()
    sns.reset_orig()

    for pas in pas_xs:
        df_pas = data[data['subjectiveAwareness'] == pas]
        for cond in list(palette.keys()):
            df_cond = df_pas[df_pas['condition'] == cond]
            if not df_cond.empty:  # if we even have data in this condition
                x_loc = pas_xs[pas]
                # so that conditions won't overlap
                if cond == "Attended":
                    x_loc -= 0.05
                else:
                    x_loc += 0.05
                y_vals = df_cond['subjectiveTimeFromLastIntactSec']
                # plot violin
                violin = plt.violinplot(y_vals, positions=[x_loc], widths=0.75, showmeans=True, showextrema=False, showmedians=False)
                # make it a half-violin plot (only to the LEFT of center)
                for b in violin['bodies']:
                    # get the center
                    m = np.mean(b.get_paths()[0].vertices[:, 0])
                    if cond == "Attended":
                        # modify the paths to not go further right than the center
                        b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], -np.inf, m)
                    else:
                        # modify the paths to not go further left than the center
                        b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], m, np.inf)
                    b.set_color(palette[cond])

                # change the color of the mean lines (showmeans=True)
                violin['cmeans'].set_color("black")
                violin['cmeans'].set_linewidth(2)
                # control the length like before
                m = np.mean(violin['cmeans'].get_paths()[0].vertices[:, 0])
                if cond == "Attended":
                    violin['cmeans'].get_paths()[0].vertices[:, 0] = np.clip(
                        violin['cmeans'].get_paths()[0].vertices[:, 0], -np.inf, m)
                else:
                    violin['cmeans'].get_paths()[0].vertices[:, 0] = np.clip(
                        violin['cmeans'].get_paths()[0].vertices[:, 0], m, np.inf)

                # then scatter
                if cond == "Attended":
                    scat_x = (np.ones(len(y_vals)) * (x_loc - 0.15)) + (np.random.rand(len(y_vals)) * 0.13)
                else:
                    scat_x = (np.ones(len(y_vals)) * (x_loc + 0.025)) + (np.random.rand(len(y_vals)) * 0.13)
                plt.scatter(x=scat_x, y=y_vals, marker="o", s=50, color=palette[cond], alpha=0.6,
                            edgecolor=palette[cond])

    # cosmetics
    plt.xticks([x for x in range(1, 5, 1)], fontsize=TICK_SIZE + 5)
    # plt.yticks([y for y in range(ymin, ymax + 1, skip)], fontsize=TICK_SIZE)
    plt.ylim(data["subjectiveTimeFromLastIntactSec"].min() - 10, data["subjectiveTimeFromLastIntactSec"].max() + 10)
    # plt.locator_params(axis='y', nbins=8)
    plt.yticks(fontsize=TICK_SIZE + 5)

    plt.title("o", pad=LABEL_PAD + 5)
    plt.ylabel("Time from last intact stimulus (seconds)", fontsize=AXIS_SIZE + 5, labelpad=LABEL_PAD)
    plt.xlabel("PAS Rating", fontsize=AXIS_SIZE + 5, labelpad=LABEL_PAD)

    # The following two lines generate custom fake lines that will be used as legend entries:
    markers = [plt.Line2D([0, 0], [0, 0], color=palette[label], marker='o', linestyle='') for label in palette]
    new_labels = [label for label in palette]
    legend = plt.legend(markers, new_labels, title="Condition", markerscale=1, fontsize=TICK_SIZE + 2)
    plt.setp(legend.get_title(), fontsize=TICK_SIZE + 2)

    # save plot
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(15, 12)
    plt.savefig(os.path.join(save_path, f"pas_soa_from_intact.svg"), format="svg", dpi=1000, bbox_inches='tight', pad_inches=0.01)
    del figure
    plt.close()
    return


def sub_bee_perf(df, save_path):
    df_uat = df[df["condition"] == parse_data_files.GAME]
    df_uat[parse_data_files.BEE_CORRECT] = df_uat[parse_data_files.BEE_CORRECT].map({True: 1, False: 0})
    df_uat_per_sub = df_uat.groupby([SUB]).mean().reset_index()
    df_uat_per_sub = df_uat_per_sub[[SUB, parse_data_files.BEE_CORRECT]]
    df_uat_per_sub.to_csv(os.path.join(save_path, parse_data_files.UAT, "bee_accuracy_per_sub.csv"), index=False)
    return


def behavioral_analysis(all_subs_df, save_path, exp="prereg"):

    # A result folder
    beh_output_path = os.path.join(save_path, parse_data_files.UNITY_OUTPUT_FOLDER)
    if not (os.path.isdir(beh_output_path)):
        os.mkdir(beh_output_path)

    if not (os.path.isdir(os.path.join(beh_output_path, parse_data_files.AT))):
        os.mkdir(os.path.join(beh_output_path, parse_data_files.AT))

    if not (os.path.isdir(os.path.join(beh_output_path, parse_data_files.UAT))):
        os.mkdir(os.path.join(beh_output_path, parse_data_files.UAT))


    """
    BEHAVIORAL ANALYSIS: pre-registered results
    """

    # VISIBILITY (PAS): participants' PAS rating (1/2/3/4) frequency in the UAT v AT conditions, and between aversive and neutral stimuli
    pas_comparison_cond(all_subs_df, beh_output_path)

    # OBJECTIVE (4AFC): participants' performance in the objective task (chance=25%) per PAS score
    pas_obj_perf(all_subs_df, beh_output_path)

    if exp == "prereg":
        # SOA: visibility per SOA - plot!
        intact_pas_soa(all_subs_df, save_path)
    # else - we don't have this information as in the pilot experiments it was not collected


    """
    BEHAVIORAL ANALYSIS: additional analyses - difference between aversive and neutral stimuli
    """

    # ANALYSIS: RT differences in obj between aversive and neutral
    val_obj_rt(all_subs_df, beh_output_path)


    """
    GAME ANALYSIS: analyses that were not pre-registered, showcasing participants' interaction with the VRIB platform
    """

    #PLOT THE RELATIONSHIPS BETWEEN TIME TO OBJECTIVE TASK, AND OBJECTIVE TASK PERFORMANCE
    plotter.plot_raincloud(df=all_subs_df, x_col_name="objectiveIsCorrect", y_col_name="objectiveTimeFromLastIntactSec",
                           plot_title="SOA and Performance in Objective Task",
                           plot_x_name="Is Correct in 4AFC", plot_y_name="Time Between Last Intact and 4AFC (seconds)",
                           save_path=beh_output_path, save_name="4AFC_SOA",
                           x_col_color_order=[["#337C8E", "#972B60"]],
                           y_tick_interval=25, y_tick_min=0, y_tick_max=100,
                           x_axis_names=["0", "1"], y_tick_names=None, x_values=[False, True])

    # AVERAGE CLUES PER TRIAL
    clues_analysis(df=all_subs_df, save_path=beh_output_path)

    # SCORE PER TRIAL
    trial_score_analysis(df=all_subs_df, save_path=beh_output_path)

    # BEE TASK PERFORMANCE PER SUB
    sub_bee_perf(df=all_subs_df, save_path=beh_output_path)

    return
