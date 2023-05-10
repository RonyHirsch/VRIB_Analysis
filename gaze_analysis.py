import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
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

AXIS_SIZE = 19
TICK_SIZE = 17
LABEL_PAD = 8

cond_map = {parse_data_files.UAT: "UAT", parse_data_files.AT: "AT"}


def unify_for_comparison(cond_dict, save_path, save_name, new_col_name):
    result_list = list()
    for cond_name in cond_dict.keys():
        df = cond_dict[cond_name]
        df.loc[:, new_col_name] = cond_name
        result_list.append(df)
    result_df = pd.concat(result_list)  # VERTICALLY
    result_df.to_csv(os.path.join(save_path, save_name), index=False)
    return result_df


def analyze_valence_gaze(all_subs_df, et_output_path):
    import matplotlib.pyplot as plt
    pas_xs = {1: 1, 2: 2, 3: 3, 4: 4}
    palette = {1: "#F86624", 0: "#9f9f9f"}  # aversive = 1, neutral = 0
    valence = {1: "Aversive", 0: "Neutral"}
    conds = ["Unattended", "Attended"]
    plt.gcf()
    plt.figure()
    sns.reset_orig()

    for cond in conds:
        subs_df_cond = all_subs_df[all_subs_df[parse_data_files.CONDITION] == cond]
        for pas in pas_xs:
            df_pas = subs_df_cond[subs_df_cond[parse_data_files.SUBJ_ANS] == pas]
            for val in list(palette.keys()):
                df_val = df_pas[df_pas[parse_data_files.TRIAL_STIM_VAL] == val]
                if not df_val.empty:  # if we even have data in this condition
                    df_avgd = df_val.groupby([SUB]).mean().reset_index()
                    # here we DO replace nan with 0, to reflect that nans mean that subjects looked 0 secs on the stimulus instance!
                    df_avgd[parse_data_files.BUSSTOP_GAZE_DUR_AVG_INTACT] = df_avgd[parse_data_files.BUSSTOP_GAZE_DUR_AVG_INTACT].fillna(0)
                    x_loc = pas_xs[pas]
                    # so that conditions won't overlap
                    if val == 1:  # aversive
                        x_loc -= 0.05
                    else:
                        x_loc += 0.05
                    y_vals = df_avgd[parse_data_files.BUSSTOP_GAZE_DUR_AVG_INTACT]
                    # plot violin
                    violin = plt.violinplot(y_vals, positions=[x_loc], widths=0.75, showmeans=True, showextrema=False, showmedians=False)
                    # make it a half-violin plot (only to the LEFT of center)
                    for b in violin['bodies']:
                        # get the center
                        m = np.mean(b.get_paths()[0].vertices[:, 0])
                        if val == 1:  # aversive
                            # modify the paths to not go further right than the center
                            b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], -np.inf, m)
                        else:
                            # modify the paths to not go further left than the center
                            b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], m, np.inf)
                        b.set_color(palette[val])

                    # change the color of the mean lines (showmeans=True)
                    violin['cmeans'].set_color("black")
                    violin['cmeans'].set_linewidth(2)
                    # control the length like before
                    m = np.mean(violin['cmeans'].get_paths()[0].vertices[:, 0])
                    if val == 1:  # aversive
                        violin['cmeans'].get_paths()[0].vertices[:, 0] = np.clip(violin['cmeans'].get_paths()[0].vertices[:, 0], -np.inf, m)
                    else:
                        violin['cmeans'].get_paths()[0].vertices[:, 0] = np.clip(violin['cmeans'].get_paths()[0].vertices[:, 0], m, np.inf)

                    # then scatter
                    if val == 1:  # aversive
                        scat_x = (np.ones(len(y_vals)) * (x_loc - 0.15)) + (np.random.rand(len(y_vals)) * 0.13)
                    else:
                        scat_x = (np.ones(len(y_vals)) * (x_loc + 0.025)) + (np.random.rand(len(y_vals)) * 0.13)
                    plt.scatter(x=scat_x, y=y_vals, marker="o", s=50, color=palette[val], alpha=0.6, edgecolor=palette[val])
        # cosmetics
        plt.xticks([x for x in range(1, 5, 1)], fontsize=22)
        plt.yticks([y for y in range(0, 7, 1)], fontsize=22)

        plt.title(f"{cond}", fontsize=27, pad=13)
        plt.ylabel("Average Gaze Duration (seconds)", fontsize=24, labelpad=8)
        plt.xlabel("PAS Rating", fontsize=24, labelpad=8)

        # The following two lines generate custom fake lines that will be used as legend entries:
        markers = [plt.Line2D([0, 0], [0, 0], color=palette[label], marker='o', linestyle='') for label in palette]
        new_labels = [valence[label] for label in palette]
        legend = plt.legend(markers, new_labels, title="Valence", markerscale=1, fontsize=19)
        plt.setp(legend.get_title(), fontsize=19)

        # save plot
        figure = plt.gcf()  # get current figure
        figure.set_size_inches(15, 12)
        plt.savefig(os.path.join(et_output_path, f"avg_gaze_duration_valence_INTACT_{cond}.svg"), format="svg", dpi=1000, bbox_inches='tight',pad_inches=0.01)
        del figure
        plt.close()
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


def trial_gaze_analysis(df, save_path):
    gaze_per_trial = time_analysis(df=df, save_path=os.path.join(save_path, "unattended"),
                                   y_col=parse_data_files.BUSSTOP_GAZE_DUR_AVG_INTACT, y_col_name="Gaze",
                                   save_name="gaze_avg")

    # PLOT
    uat_trials = df[df[parse_data_files.TRIAL_NUMBER] < 40]
    uat_trials = uat_trials.loc[:, [SUB, parse_data_files.TRIAL_NUMBER, parse_data_files.BUSSTOP_GAZE_DUR_AVG_INTACT]]

    plt.clf()
    plt.figure()
    sns.reset_orig()
    subs = sorted(list(uat_trials[SUB].unique()))
    # individual lines' colors; change color to color="#CBD2D0" to have them all the same
    colormap = plt.cm.BrBG  # choose colormap: http://matplotlib.org/1.2.1/examples/pylab_examples/show_colormaps.html
    colors = [colormap(i) for i in np.linspace(0, 1, len(subs))]
    for i in range(len(subs)):  # plot individual data lines
        sub = subs[i]
        sub_trials = uat_trials[uat_trials[SUB] == sub]
        sns.lineplot(x=parse_data_files.TRIAL_NUMBER, y=parse_data_files.BUSSTOP_GAZE_DUR_AVG_INTACT, data=sub_trials, color=colors[i], label="", linewidth=1, alpha=0.8)
    # now plot average line
    sns.lineplot(x=parse_data_files.TRIAL_NUMBER, y="Gaze Avg", data=gaze_per_trial, color="#4C5454", label="", linewidth=4)
    # cosmetics
    plt.xticks(fontsize=plotter.F_AXES_NAME)
    plt.title("Gaze Duration Per Trial", fontsize=plotter.F_TITLE, pad=plotter.LABELPAD)
    plt.xlabel("Trial Number", fontsize=plotter.F_AXES_TITLE, labelpad=plotter.LABELPAD)
    plt.ylabel("Gaze Duration (Seconds)", fontsize=plotter.F_AXES_TITLE, labelpad=plotter.LABELPAD)
    plt.legend().remove()
    # save
    figure = plt.gcf()  # get current figure
    plt.savefig(os.path.join(save_path, parse_data_files.UAT, f"{plotter.LINE}_gaze_avg_per_trial.png"), dpi=plotter.DPI, bbox_inches='tight')
    del figure
    plt.clf()
    plt.cla()
    plt.close()
    return


def gaze_dur_pas(att_avg_gaze, uat_avg_gaze, save_path):
    pas_xs = {1: 1, 2: 2, 3: 3, 4: 4}
    palette = {"Unattended": {"intact": "#e5625e", "scrambled": "#f7b2bd"}, "Attended": {"intact": "#034078", "scrambled": "#1282a2"}}
    conds = {"Unattended": uat_avg_gaze, "Attended": att_avg_gaze}
    plt.gcf()
    plt.figure()
    sns.reset_orig()
    for cond in list(conds.keys()):
        df_cond = conds[cond]
        colors_cond = palette[cond]
        for pas in pas_xs:
            df_pas = df_cond[df_cond['subjectiveAwareness'] == pas]
            if not df_pas.empty:
                for presentation in list(colors_cond.keys()):
                    df_pres = df_pas[df_pas['presentation'] == presentation]
                    if not df_pres.empty:  # if we even have data in this condition
                        df_pres = df_pres[df_pres["avgBusstopGazeDuration"].notna()]
                        color = colors_cond[presentation]
                        x_loc = pas_xs[pas]
                        # so that conditions won't overlap
                        if presentation == "intact":
                            x_loc -= 0.05
                        else:
                            x_loc += 0.05
                        y_vals = df_pres["avgBusstopGazeDuration"]
                        # plot violin
                        violin = plt.violinplot(y_vals, positions=[x_loc], widths=0.75, showmeans=True, showextrema=False, showmedians=False)
                        # make it a half-violin plot (only to the LEFT of center)
                        for b in violin['bodies']:
                            # get the center
                            m = np.mean(b.get_paths()[0].vertices[:, 0])
                            if presentation == "intact":
                                # modify the paths to not go further right than the center
                                b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], -np.inf, m)
                            else:
                                # modify the paths to not go further left than the center
                                b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], m, np.inf)
                            b.set_color(color)

                        # change the color of the mean lines (showmeans=True)
                        violin['cmeans'].set_color("black")
                        violin['cmeans'].set_linewidth(2)
                        # control the length like before
                        m = np.mean(violin['cmeans'].get_paths()[0].vertices[:, 0])
                        if presentation == "intact":
                            violin['cmeans'].get_paths()[0].vertices[:, 0] = np.clip(violin['cmeans'].get_paths()[0].vertices[:, 0], -np.inf, m)
                        else:
                            violin['cmeans'].get_paths()[0].vertices[:, 0] = np.clip(violin['cmeans'].get_paths()[0].vertices[:, 0], m, np.inf)

                        # then scatter
                        if presentation == "intact":
                            scat_x = (np.ones(len(y_vals)) * (x_loc - 0.15)) + (np.random.rand(len(y_vals)) * 0.13)
                        else:
                            scat_x = (np.ones(len(y_vals)) * (x_loc + 0.025)) + (np.random.rand(len(y_vals)) * 0.13)
                        plt.scatter(x=scat_x, y=y_vals, marker="o", s=50, color=color, alpha=0.6, edgecolor=color)

        # cosmetics
        plt.xticks([x for x in range(1, 5, 1)], fontsize=TICK_SIZE + 5)
        plt.yticks([y for y in range(0, 10, 1)], fontsize=TICK_SIZE + 5)

        plt.title(f"{cond}", fontsize=AXIS_SIZE + 8, pad=LABEL_PAD + 5)
        plt.ylabel("Average Gaze Duration (seconds)", fontsize=AXIS_SIZE + 5, labelpad=LABEL_PAD)
        plt.xlabel("PAS Rating", fontsize=AXIS_SIZE + 5, labelpad=LABEL_PAD)

        # The following two lines generate custom fake lines that will be used as legend entries:
        markers = [plt.Line2D([0, 0], [0, 0], color=colors_cond[presentation], marker='o', linestyle='') for presentation in list(colors_cond.keys())]
        new_labels = [label.title() for label in list(colors_cond.keys())]
        legend = plt.legend(markers, new_labels, title="Condition", markerscale=1, fontsize=TICK_SIZE + 2)
        plt.setp(legend.get_title(), fontsize=TICK_SIZE + 2)

        # save plot
        figure = plt.gcf()  # get current figure
        figure.set_size_inches(15, 12)
        plt.savefig(os.path.join(save_path, f"busstop_gaze_dur_total_avg_{cond}.svg"), format="svg", dpi=1000, bbox_inches='tight', pad_inches=0.01)
        del figure
        plt.close()
    return


def gaze_per_cond(subs_df_cond, cond_output_path):
    relevant_cols = [SUB, parse_data_files.SUBJ_ANS, parse_data_files.TRIAL_NUMBER, parse_data_files.OBJ_ANS,
                     parse_data_files.SUBJ_BUSSTOP, parse_data_files.OBJ_BUSSTOP, parse_data_files.TRIAL_STIM_VAL,
                     parse_data_files.BUSSTOP_GAZE_DUR_AVG_INTACT, parse_data_files.BUSSTOP_GAZE_DUR_AVG_SCRAMBLED]
    df = subs_df_cond[relevant_cols]
    # process df into a long format and save it
    long_df = pd.melt(df, id_vars=relevant_cols[:-2], value_vars=relevant_cols[-2:], var_name="presentation")
    long_df.loc[:, "presentation"] = long_df["presentation"].map(
        {parse_data_files.BUSSTOP_GAZE_DUR_AVG_INTACT: "intact",
         parse_data_files.BUSSTOP_GAZE_DUR_AVG_SCRAMBLED: "scrambled"})
    long_df.rename({"value": "avgBusstopGazeDuration"}, axis=1, inplace=True)
    long_df.to_csv(os.path.join(cond_output_path, f"avg_gaze_per_pas_long.csv"), index=False)

    # for t-tests
    df_per_sub = long_df.groupby([SUB, "presentation"]).mean().reset_index().loc[:, [SUB, "presentation", "avgBusstopGazeDuration"]]
    df_per_sub_intact = df_per_sub[df_per_sub["presentation"] == "intact"].loc[:, [SUB, "avgBusstopGazeDuration"]]
    df_per_sub_intact.rename({"avgBusstopGazeDuration": "avgGaze_intact"}, axis=1, inplace=True)
    df_per_sub_scrmbld = df_per_sub[df_per_sub["presentation"] != "intact"].loc[:, [SUB, "avgBusstopGazeDuration"]]
    df_per_sub_scrmbld.rename({"avgBusstopGazeDuration": "avgGaze_scrambled"}, axis=1, inplace=True)
    df_per_sub_unified = pd.merge(df_per_sub_intact, df_per_sub_scrmbld, on=SUB)
    df_per_sub_unified.to_csv(os.path.join(cond_output_path, f"avg_gaze_per_sub_presentation.csv"), index=False)
    return


def et_analysis(all_subs_df, save_path):

    # A result folder
    et_output_path = os.path.join(save_path, "et")
    if not (os.path.isdir(et_output_path)):
        os.mkdir(et_output_path)

    if not (os.path.isdir(os.path.join(et_output_path, parse_data_files.AT))):
        os.mkdir(os.path.join(et_output_path, parse_data_files.AT))

    if not (os.path.isdir(os.path.join(et_output_path, parse_data_files.UAT))):
        os.mkdir(os.path.join(et_output_path, parse_data_files.UAT))

    # GAZE (INTACT) PER TRIAL
    trial_gaze_analysis(df=all_subs_df, save_path=et_output_path)

    # create per condition
    conds = {parse_data_files.UAT: all_subs_df[all_subs_df[parse_data_files.TRIAL_NUMBER] < 40],
             parse_data_files.AT: all_subs_df[all_subs_df[parse_data_files.TRIAL_NUMBER] >= 40]}
    for condition in conds:  # separate ET analysis between attended from unattended trials
        cond_output_path = os.path.join(et_output_path, condition)
        subs_df_cond = conds[condition]
        gaze_per_cond(subs_df_cond, cond_output_path)

    # GAZE DURATION (INTACT) PER TRIAL SEPARATELY FOR AVERSIVE AND NEUTRAL TRIALS!
    analyze_valence_gaze(all_subs_df, et_output_path)
    val_cond_df = all_subs_df.groupby([SUB, parse_data_files.CONDITION, parse_data_files.SUBJ_ANS,
                                       parse_data_files.TRIAL_STIM_VAL]).mean().reset_index().loc[:,
                  [SUB, parse_data_files.CONDITION, parse_data_files.SUBJ_ANS, parse_data_files.TRIAL_STIM_VAL,
                   parse_data_files.BUSSTOP_GAZE_DUR_AVG_INTACT]]
    for condition in ["Attended", "Unattended"]:
        val_df = val_cond_df[val_cond_df[parse_data_files.CONDITION] == condition]
        pas = 1 if condition == "Unattended" else 4
        df = val_df[val_df[parse_data_files.SUBJ_ANS] == pas]
        df_aversive = df[df[parse_data_files.TRIAL_STIM_VAL] == 1]
        df_aversive.drop(columns=[parse_data_files.TRIAL_STIM_VAL, parse_data_files.SUBJ_ANS, parse_data_files.CONDITION], inplace=True)
        df_aversive.rename({parse_data_files.BUSSTOP_GAZE_DUR_AVG_INTACT: f"gazeIntact_pas{pas}_aversive"}, axis=1, inplace=True)
        df_neutral = df[df[parse_data_files.TRIAL_STIM_VAL] != 1]
        df_neutral.drop(columns=[parse_data_files.TRIAL_STIM_VAL, parse_data_files.SUBJ_ANS, parse_data_files.CONDITION], inplace=True)
        df_neutral.rename({parse_data_files.BUSSTOP_GAZE_DUR_AVG_INTACT: f"gazeIntact_pas{pas}_neutral"}, axis=1, inplace=True)
        df_result = pd.merge(df_aversive, df_neutral, on=SUB)
        df_result.to_csv(os.path.join(et_output_path, condition, "avg_gaze_duration_valence.csv"), index=False)

    # create comparison
    comp_output_path = os.path.join(et_output_path, "comparison")
    if not (os.path.isdir(comp_output_path)):
        os.mkdir(comp_output_path)

    avg_gaze = {parse_data_files.UAT: None, parse_data_files.AT: None}

    for condition in avg_gaze.keys():  # separate ET analysis between attended from unattended trials
        cond_output_path = os.path.join(et_output_path, condition)
        avg_gaze_in_cond = pd.read_csv(os.path.join(cond_output_path, f"avg_gaze_per_pas_long.csv"))
        avg_gaze[condition] = avg_gaze_in_cond

    # GAZE DURATION PER PAS SCORE SEPARATELY FOR THE ATTENDED AND UNATTENDED CONDITIONS (PER PRESENTATION TYPE)
    gaze_dur_pas(att_avg_gaze=avg_gaze[parse_data_files.AT], uat_avg_gaze=avg_gaze[parse_data_files.UAT], save_path=et_output_path)

    unified_df = unify_for_comparison(avg_gaze, comp_output_path, f"avg_gaze_per_pas_long.csv", "condition")
    unified_df_intact = unified_df[unified_df["presentation"] == "intact"]
    unified_df_intact.to_csv(os.path.join(comp_output_path, f"avg_gaze_per_pas_long_intact.csv"), index=False)

    # for t-tests per trial
    avg_gaze_per_trial_pas1_intact = all_subs_df[all_subs_df[parse_data_files.SUBJ_ANS] == 1]
    avg_gaze_per_trial_pas1_intact = avg_gaze_per_trial_pas1_intact.loc[:, [SUB, parse_data_files.SUBJ_ANS, parse_data_files.BUSSTOP_GAZE_DUR_AVG_INTACT]]
    avg_gaze_per_trial_pas1_intact.to_csv(os.path.join(et_output_path, "avg_gaze_per_trial_pas1_intact.csv"), index=False)

    return
