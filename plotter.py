import os
import gc
import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np

pd.options.mode.chained_assignment = None  # default='warn' see: https://stackoverflow.com/questions/20625582/how-to-deal-with-settingwithcopywarning-in-pandas

# plot parameters
F_AXES_NAME = 11
F_AXES_TITLE = 12
F_TITLE = 15
LABELPAD = 5
DPI = 1000

# plot save name
BAR = 'bar'
RAIN = 'raincloud'
LINE = 'line'

JITTER_WIDTH = 0.12
VIOLIN_OFFSET = 0.35
SCATTER_MARKER_SIZE = 8
SCATTER_MARKER_ALPHA = 0.5
VIOLIN_ALPHA = 0.5
ALPHA_STEP = 0.2
LINE_WIDTH = 0.5
VIOLIN_EDGE_COLOR = "black"

F_HEADER = 16
F_AXES_TITLE = 14
F_HORIZ_LINES = 11
XLABELPAD = 20
YLABELPAD = 20
BUFFER = 0.01
W = 10
H = 7.5
DPI = 1000


def plot_PAS_comp_task(path_to_pas):
    return


def plot_bar(df, x_col_name, y_col_name, plot_title, plot_x_name, plot_y_name, save_path, save_name, hue_col_name=None,
             x_col_axis_order=None, conf_interval=None, palette=None, y_tick_interval=5, x_axis_names=None,
             y_min_max=None, y_tick_names=None):
    gc.collect()
    plt.clf()
    plt.figure()
    sns.reset_orig()
    sns.barplot(data=df, x=x_col_name, order=x_col_axis_order, y=y_col_name, hue=hue_col_name, ci=conf_interval,
                palette=palette, errcolor="silver", errwidth=LINE_WIDTH)

    plt.yticks(np.arange(round(df[y_col_name].min()), df[y_col_name].max() + (y_tick_interval / 2), y_tick_interval),
               fontsize=F_AXES_NAME)
    if x_axis_names is not None:
        plt.xticks(ticks=df[x_col_name].unique(), labels=x_axis_names, fontsize=F_AXES_NAME)
    else:
        plt.xticks(fontsize=F_AXES_NAME)

    if y_min_max is not None:
        if y_tick_names is not None:
            plt.yticks(ticks=np.arange(y_min_max[0], y_min_max[1] + (y_tick_interval / 2), y_tick_interval), labels=y_tick_names, fontsize=F_AXES_NAME)
        else:
            plt.yticks(ticks=np.arange(y_min_max[0], y_min_max[1] + (y_tick_interval / 2), y_tick_interval))

    plt.title(plot_title, fontsize=F_TITLE, pad=LABELPAD)
    plt.xlabel(plot_x_name, fontsize=F_AXES_TITLE, labelpad=LABELPAD)
    plt.ylabel(plot_y_name, fontsize=F_AXES_TITLE, labelpad=LABELPAD)

    figure = plt.gcf()  # get current figure
    figure.set_size_inches(20.5, 10.5)
    plt.savefig(os.path.join(save_path, f"{BAR}_{save_name}.png"), dpi=DPI, bbox_inches='tight')

    del figure
    plt.clf()
    plt.cla()
    plt.close()
    gc.collect()
    return


def make_it_rain(df, x_col_name, y_col_name, x_col_color_order, violin_alpha=VIOLIN_ALPHA,
                 marker_alpha=SCATTER_MARKER_ALPHA, x_values=None, right=None):

    #sns.set_theme(style="whitegrid")
    sns.set_style(style="ticks")

    violins_list = list()
    positions_list = list()
    df.loc[:, x_col_name] = pd.to_numeric(df[x_col_name])
    if x_values is None:
        x_values = np.sort(df[x_col_name].unique())
    for x_value in x_values:
        positions_list.append(x_value)  # the future "position" of this data on the X axis
        violins_list.append(df[df[x_col_name] == x_value])  # the data that will be plotted in this location

    legend_flag = 0
    vio = None  # vio is "returned" to the plotting function JUST FOR LEGEND COLOR PURPOSES

    for c in range(len(violins_list)):  # plot violin-by-violin
        if violins_list[c].empty:
            continue
        else:
            data = violins_list[c][y_col_name]
            #data = [y if not (np.isnan(y)) else 0 for y in data]  # ZEROS out the nans
            data = [y for y in data if not (np.isnan(y))]  # get rid of nans
            if len(data) == 0:  # in case ALL data is nan, so we won't crash, let's just plot something on the zero axis
                data = [y if not (np.isnan(y)) else 0 for y in violins_list[c][y_col_name]]  # ZEROS out the nans
            if not(all(isinstance(x, (int, float)) for x in data)):
                data = np.where(data, 1, 0)  # replace True with 1, False with 0

            # violin_plot: just a single violin!
            violin = plt.violinplot(data, positions=[positions_list[c]], showmeans=True, showextrema=False, showmedians=False)

            # change the color of the mean lines (showmeans=True)
            violin['cmeans'].set_color("black")
            violin['cmeans'].set_linewidth(1)
            # control the length like before
            m = np.mean(violin['cmeans'].get_paths()[0].vertices[:, 0])
            if right is None or right is False:
                violin['cmeans'].get_paths()[0].vertices[:, 0] = np.clip(violin['cmeans'].get_paths()[0].vertices[:, 0], -np.inf, m)
            else:
                violin['cmeans'].get_paths()[0].vertices[:, 0] = np.clip(violin['cmeans'].get_paths()[0].vertices[:, 0], m, np.inf)

            # get the color of the first violin to be the legend color
            if legend_flag == 0:
                vio = violin['bodies'][0]
                legend_flag = 1

            # make it a half-violin plot (only to the LEFT of center)
            b = violin['bodies'][0]  # single violin = single body
            # set alpha
            b.set_alpha(violin_alpha)

            # make it a half violin
            m = np.mean(b.get_paths()[0].vertices[:, 0])  # get the center
            if right is None or right is False:
                # modify the paths to not go further right than the center
                b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], -np.inf, m)
            else:
                # modify the paths to not go further left than the center
                b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], m, np.inf)
            if x_col_color_order is not None:
                b.set_color(x_col_color_order[c])
            # add violin edges
            b.set_edgecolor(x_col_color_order[c])  # VIOLIN_EDGE_COLOR

            b.set_linewidth(1)
            # then scatter
            if right is None or right is False:
                scat_x = (np.ones(len(data)) * (positions_list[c] - 0.09)) + (np.random.rand(len(data)) * 0.09)
            else:
                scat_x = (np.ones(len(data)) * (positions_list[c] + 0.09)) + (np.random.rand(len(data)) * 0.09)
            plt.scatter(x=scat_x, y=data, marker="o", color=x_col_color_order[c], alpha=marker_alpha, s=SCATTER_MARKER_SIZE, edgecolor=x_col_color_order[c])

            # complete with a boxplot
            #plt.boxplot(data, positions=[positions_list[c] + VIOLIN_OFFSET], notch=False,
            #            medianprops=dict(color='black', linewidth=LINE_WIDTH),
            #            showfliers=False)

    return vio


def plot_raincloud(df, x_col_name, y_col_name, plot_title, plot_x_name, plot_y_name, save_path, save_name,
                   x_col_color_order=None, y_tick_interval=5, y_tick_min=None, y_tick_max=None,
                   x_axis_names=None, y_tick_names=None, group_col_name=None, group_name_mapping=None, x_values=None,
                   add_horizontal_line=None, alpha_step=ALPHA_STEP, valpha=VIOLIN_ALPHA, group_name=None, is_right=False):
    gc.collect()
    plt.clf()
    plt.figure()
    sns.reset_orig()

    if group_col_name is not None:
        labels = list()
        malpha = SCATTER_MARKER_ALPHA
        group_name = re.sub('([A-Z])', r' \1', group_col_name).title() if group_name is None else group_name
        i = 0
        for val in df[group_col_name].unique():
            if is_right == False:
                is_right_flag = 0
            else:
                is_right_flag = 1
            if group_name_mapping is not None:
                if not isinstance(val, str):
                    try:
                        label = group_name_mapping[val]
                    except KeyError:
                        label = group_name_mapping[str(val)]
                else:
                    label = group_name_mapping[val]
            else:
                label = str(val)
            data = df[df[group_col_name] == val]
            vio = make_it_rain(data, x_col_name, y_col_name, x_col_color_order[i], violin_alpha=valpha,
                               marker_alpha=malpha, x_values=x_values, right=is_right)
            if is_right_flag == 0:
                is_right = True
            else:
                is_right = False
            #valpha -= alpha_step  # LEFT AND RIGHT VIOLINS REPLACE THE NEED FOR ALPHAS
            #malpha -= alpha_step
            i += 1
            vcolor = vio.get_facecolor().flatten()
            labels.append((mpatches.Patch(color=vcolor), label))
        if len(df[x_col_name].unique().tolist()) > 10:  # a wide plot
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.8))
        plt.legend(*zip(*labels), title=group_name)

    else:
        labels = [""]
        if x_col_color_order is None:
            make_it_rain(df, x_col_name, y_col_name, x_col_color_order, violin_alpha=VIOLIN_ALPHA,
                         marker_alpha=SCATTER_MARKER_ALPHA, x_values=x_values)
        else:
            make_it_rain(df, x_col_name, y_col_name, x_col_color_order[0], violin_alpha=VIOLIN_ALPHA,
                         marker_alpha=SCATTER_MARKER_ALPHA, x_values=x_values)

    plt.gcf()
    # Axes and titles
    y_list = pd.to_numeric(df[y_col_name]).tolist()
    y_min = round(min(y_list)) if y_tick_min is None else y_tick_min
    y_max = round(max(y_list)) if y_tick_max is None else y_tick_max
    if y_tick_names is not None:
        plt.yticks(ticks=np.arange(y_min, y_max + (y_tick_interval / 2), y_tick_interval), labels=y_tick_names,
                   fontsize=F_AXES_NAME)
    else:
        plt.yticks(np.arange(y_min, y_max + (y_tick_interval / 2), y_tick_interval), fontsize=F_AXES_NAME)
    if x_axis_names is not None:
        plt.xticks(ticks=np.sort(x_values), labels=x_axis_names, fontsize=F_AXES_NAME)
    else:
        plt.xticks(ticks=np.sort(df[x_col_name].unique()), fontsize=F_AXES_NAME)
    plt.title(plot_title, fontsize=F_TITLE, pad=LABELPAD)
    plt.xlabel(plot_x_name, fontsize=F_AXES_TITLE, labelpad=LABELPAD)
    plt.ylabel(plot_y_name, fontsize=F_AXES_TITLE, labelpad=LABELPAD-1.7)

    if add_horizontal_line is not None:
        plt.axhline(y=add_horizontal_line, color='slategrey', linestyle='--', linewidth=1)

    figure = plt.gcf()  # get current figure
    if len(df[x_col_name].unique().tolist()) > 10:  # a wide plot
        figure.set_size_inches(20, 10)
    figure.savefig(os.path.join(save_path, f"{save_name}.png"), dpi=DPI, bbox_inches='tight')
    figure.savefig(os.path.join(save_path, f"{save_name}.svg"), format="svg", dpi=DPI, bbox_inches='tight')

    del figure
    plt.clf()
    plt.cla()
    plt.close()
    gc.collect()
    return


def regression_plot(X,y, y_pred, x_axis, y_axis, title, save_path, save_name, scatter_color="#361f27", line_color="#912f56"):
    gc.collect()
    plt.clf()
    plt.figure()
    sns.reset_orig()

    #plt.scatter(X, y, color='black')
    sns.scatterplot(x=X, y=y, c="#361f27", s=7)
    plt.plot(X, y_pred, color="#912f56")
    plt.xlabel(x_axis, fontsize=F_AXES_TITLE, labelpad=LABELPAD)
    plt.ylabel(y_axis, fontsize=F_AXES_TITLE, labelpad=LABELPAD)
    plt.title(title, fontsize=F_TITLE, pad=LABELPAD)

    figure = plt.gcf()  # get current figure
    plt.savefig(os.path.join(save_path, f"{LINE}_{save_name}.png"), dpi=DPI, bbox_inches='tight')
    plt.legend().remove()

    del figure
    plt.clf()
    plt.cla()
    plt.close()
    gc.collect()
    return


def simple_avg_line(df, x_col_name, avg_col_name, std_col_name, line_color, plot_title, plot_x_name, plot_y_name,
                    save_path, save_name, y_tick_interval=5, y_tick_labels=None, y_tick_min=None, y_tick_max=None):
    gc.collect()
    plt.clf()
    plt.figure()
    sns.reset_orig()

    sns.lineplot(x=x_col_name, y=avg_col_name, data=df, color=line_color, label="")
    plt.fill_between(df.index.values.tolist(),
                     [y - sd for y, sd in zip(df[avg_col_name], df[std_col_name])],
                     [y + sd for y, sd in zip(df[avg_col_name], df[std_col_name])],
                     alpha=.2, color=line_color)

    plt.xticks(fontsize=F_AXES_NAME)
    plt.title(plot_title, fontsize=F_TITLE, pad=LABELPAD)
    plt.xlabel(plot_x_name, fontsize=F_AXES_TITLE, labelpad=LABELPAD)
    plt.ylabel(plot_y_name, fontsize=F_AXES_TITLE, labelpad=LABELPAD)

    plt.legend().remove()
    figure = plt.gcf()  # get current figure
    plt.savefig(os.path.join(save_path, f"{LINE}_{save_name}.png"), dpi=DPI, bbox_inches='tight')

    del figure
    plt.clf()
    plt.cla()
    plt.close()
    gc.collect()

    return


def line_plot(df, x_col_name, y_col_names, plot_title, plot_x_name, plot_y_name, save_path, save_name,
              colors, y_tick_interval=5, y_tick_labels=None, y_tick_min=None, y_tick_max=None, show_legend=False):
    gc.collect()
    plt.clf()
    plt.figure()
    sns.reset_orig()

    # aggregate data: get avg and std
    avg = df.groupby([x_col_name]).mean()
    std = df.groupby([x_col_name]).std()
    for i in range(len(y_col_names)):
        data_name = re.sub('([A-Z])', r' \1', y_col_names[i]).title()
        sns.lineplot(x=x_col_name, y=y_col_names[i], data=avg, palette=colors[y_col_names[i]], label=data_name)
        plt.fill_between(avg.index.values.tolist(),
                         [y - se for y, se in zip(avg[y_col_names[i]], std[y_col_names[i]])],
                         [y + se for y, se in zip(avg[y_col_names[i]], std[y_col_names[i]])],
                         alpha=.2, color=colors[y_col_names[i]])

    bool_flag = 0
    for c in y_col_names:
        if df[c].dtypes.name == 'bool':
            bool_flag = 1
            plt.yticks(ticks=[False, True], labels=["False", "True"], fontsize=F_AXES_NAME)

    if bool_flag == 0:
        min_val = min([round(df[c].min()) for c in y_col_names]) if y_tick_min is None else y_tick_min
        max_val = max([round(df[c].max()) for c in y_col_names]) if y_tick_max is None else y_tick_max
        if y_tick_labels is None:
            plt.yticks(np.arange(min_val, max_val + (y_tick_interval / 2), y_tick_interval), fontsize=F_AXES_NAME)
        else:
            t = df[y_col_names[0]].unique() if (y_tick_min is None or y_tick_max is None) else np.arange(min_val,
                                                                                                         max_val)
            plt.yticks(ticks=t, labels=y_tick_labels, fontsize=F_AXES_NAME)
    plt.xticks(fontsize=F_AXES_NAME)
    plt.title(plot_title, fontsize=F_TITLE, pad=LABELPAD)
    plt.xlabel(plot_x_name, fontsize=F_AXES_TITLE, labelpad=LABELPAD)
    plt.ylabel(plot_y_name, fontsize=F_AXES_TITLE, labelpad=LABELPAD)

    if show_legend == True:
        plt.legend()
    else:
        plt.legend().remove()
    figure = plt.gcf()  # get current figure
    plt.savefig(os.path.join(save_path, f"{LINE}_{save_name}.png"), dpi=DPI, bbox_inches='tight')

    del figure
    plt.clf()
    plt.cla()
    plt.close()
    gc.collect()
    return


def plot_avg_line(title, trial_df_list, avg_col_list, se_col_list=None, label_list=None, x_name="", y_name="",
                  color_list=None, x_tick_intervals=4, significance_bars_dict=None,
                  save=False, save_name="", save_path="", sub_folder=""):
    plt.clf()
    plt.figure()
    sns.reset_orig()
    # x axis
    trials_num = [t.shape[0] for t in trial_df_list]
    trials = [np.arange(0, n, 1) for n in trials_num]  # this will be the x axis

    # plot Y boundary range
    minimal = 1000000000
    maximal = -100000000
    if se_col_list is not None:
        for i in range(len(se_col_list)):
            if se_col_list[i] is not None:
                trial_df_list[i][se_col_list[i]].fillna(0, inplace=True)
                lower = [x - se for x, se in zip(trial_df_list[i][avg_col_list[i]], trial_df_list[i][se_col_list[i]])]
                upper = [x + se for x, se in zip(trial_df_list[i][avg_col_list[i]], trial_df_list[i][se_col_list[i]])]
            else:
                lower = trial_df_list[i][avg_col_list[i]]
                upper = trial_df_list[i][avg_col_list[i]]
            lowest = min(lower)
            highest = max(upper)
            minimal = min(minimal, lowest)
            maximal = max(maximal, highest)

    else:
        for i in range(len(avg_col_list)):
            if avg_col_list[i] is not None:
                lower = trial_df_list[i][avg_col_list[i]]
                upper = trial_df_list[i][avg_col_list[i]]
                lowest = min(lower)
                highest = max(upper)
                minimal = min(minimal, lowest)
                maximal = max(maximal, highest)

    # colors
    if color_list is None:
        colors = sns.color_palette("colorblind", len(avg_col_list))
    else:
        colors = color_list

    # plot
    for i in range(len(trial_df_list)):
        plt.plot(trials[i], trial_df_list[i][avg_col_list[i]], ls='-', color=colors[i], label=label_list[i])
        if se_col_list is not None:  # if data has SE
            if se_col_list[i] is not None:
                plt.fill_between(trials[i],
                                 [x - se for x, se in zip(trial_df_list[i][avg_col_list[i]], trial_df_list[i][se_col_list[i]])],
                                 [x + se for x, se in zip(trial_df_list[i][avg_col_list[i]], trial_df_list[i][se_col_list[i]])],
                                 alpha=.2, color=colors[i])

    # do we want to add gray bars that denote significance
    if significance_bars_dict is not None:
        for k in significance_bars_dict.keys():
            for x_start, x_end in significance_bars_dict[k]["x"]:
                xbar = [x for x in range(int(x_start), int(x_end) + 1)]
                ybar = [significance_bars_dict[k]["y"]] * len(xbar)
                plt.plot(xbar, ybar, color="gray", linewidth=6, alpha=0.5)

    plt.ylim([minimal - BUFFER/2, maximal + BUFFER/2])
    plt.yticks(np.arange(minimal - BUFFER/2, maximal + BUFFER/2, 0.004), labels=[f"{i:.3f}" for i in np.arange(minimal - BUFFER/2, maximal + BUFFER/2, 0.004)])

    plt.xlim([0, trial_df_list[0].shape[0]])
    x_tick_div = int(max(trials_num) / x_tick_intervals)
    plt.xticks(np.arange(0, max(trials_num) + x_tick_intervals, x_tick_div))

    plot_title = f"{title.title()}"
    plt.title(plot_title, fontsize=F_TITLE)
    plt.xlabel(x_name, fontsize=F_AXES_TITLE, labelpad=XLABELPAD)
    plt.ylabel(y_name, fontsize=F_AXES_TITLE, labelpad=YLABELPAD)
    plt.legend()

    figure = plt.gcf()  # get current figure
    figure.set_size_inches(W, H)
    plt.savefig(os.path.join(save_path, f"LINE_{save_name}.png"), dpi=DPI)
    plt.clf()
    plt.close()
    return


def plot_avg_line_dict(title, trial_df_dict, avg_col_list, se_col_list=None, label_list=None, x_name="", y_name="",
                       color_list=None, x_tick_intervals=4, y_tick_interval=1, significance_bars_dict=None,
                       save=False, save_name="", save_path="", sub_folder="", y_min=None, y_max=None):

    plt.clf()
    plt.figure()
    #sns.set_theme(style="whitegrid")
    sns.set_theme(style="white")
    # x axis
    trials_num = {key: trial_df_dict[key].shape[0] for key in trial_df_dict.keys()}
    trials = {key: np.arange(0, trials_num[key], 1) for key in trials_num.keys()}  # this will be the x axis

    # plot Y boundary range
    if y_min is None:
        minimal = 1000000000
        maximal = -100000000
        if se_col_list is not None:
            for key in se_col_list.keys():
                if se_col_list[key] is not None:
                    trial_df_dict[key][se_col_list[key]].fillna(0, inplace=True)
                    lower = [x - se for x, se in zip(trial_df_dict[key][avg_col_list[key]], trial_df_dict[key][se_col_list[key]])]
                    upper = [x + se for x, se in zip(trial_df_dict[key][avg_col_list[key]], trial_df_dict[key][se_col_list[key]])]
                else:
                    lower = trial_df_dict[key][avg_col_list[key]]
                    upper = trial_df_dict[key][avg_col_list[key]]
                lowest = min(lower)
                highest = max(upper)
                minimal = min(minimal, lowest)
                maximal = max(maximal, highest)

        else:
            for key in avg_col_list.keys():
                if avg_col_list[key] is not None:
                    lower = trial_df_dict[key][avg_col_list[key]]
                    upper = trial_df_dict[key][avg_col_list[key]]
                    lowest = min(lower)
                    highest = max(upper)
                    minimal = min(minimal, lowest)
                    maximal = max(maximal, highest)

    else:
        minimal = y_min
        maximal = y_max

    # colors
    if color_list is None:
        colors = sns.color_palette("colorblind", len(avg_col_list))
    else:
        colors = color_list

    # plot
    for key in trial_df_dict.keys():
        plt.plot(trials[key], trial_df_dict[key][avg_col_list[key]], ls='-', color=colors[key], label=label_list[key])
        if se_col_list is not None:  # if data has SE
            if se_col_list[key] is not None:
                plt.fill_between(trials[key],
                                 [x - se for x, se in zip(trial_df_dict[key][avg_col_list[key]], trial_df_dict[key][se_col_list[key]])],
                                 [x + se for x, se in zip(trial_df_dict[key][avg_col_list[key]], trial_df_dict[key][se_col_list[key]])],
                                 alpha=.2, color=colors[key])

    # do we want to add gray bars that denote significance
    if significance_bars_dict is not None:
        for k in significance_bars_dict.keys():
            for x_start, x_end in significance_bars_dict[k]["x"]:
                xbar = [x for x in range(int(x_start), int(x_end) + 1)]
                ybar = [significance_bars_dict[k]["y"]] * len(xbar)
                plt.plot(xbar, ybar, color="gray", linewidth=6, alpha=0.5)

    plt.ylim([minimal, maximal])
    #plt.yticks(np.arange(minimal - BUFFER/2, maximal + BUFFER/2, 0.004), labels=[f"{i:.3f}" for i in np.arange(minimal - BUFFER/2, maximal + BUFFER/2, 0.004)])
    plt.yticks(np.arange(y_min, y_max + (y_tick_interval / 2), y_tick_interval), fontsize=F_AXES_NAME)

    some_key = list(trial_df_dict.keys())[0]
    plt.xlim([0, trial_df_dict[some_key].shape[0]])
    x_tick_div = int(trials_num[some_key] / x_tick_intervals) # all length are the same, so it is OK not to calculate max of trials_num
    plt.xticks(np.arange(0, trials_num[some_key] + x_tick_intervals, x_tick_div))

    plt.title(title, fontsize=F_HEADER)
    plt.xlabel(x_name, fontsize=F_AXES_TITLE)
    plt.ylabel(y_name, fontsize=F_AXES_TITLE)
    plt.legend()

    figure = plt.gcf()  # get current figure
    #figure.set_size_inches(W, H)
    plt.savefig(os.path.join(save_path, f"LINE_{save_name}.png"), dpi=DPI)
    plt.clf()
    plt.close()
    return
