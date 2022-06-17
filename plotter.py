import os
import gc
import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np

# plot parameters
F_AXES_NAME = 8
F_AXES_TITLE = 12
F_TITLE = 15
LABELPAD = 5
DPI = 1000

# plot save name
BAR = 'bar'
RAIN = 'raincloud'
LINE = 'line'

JITTER_WIDTH = 0.12
VIOLIN_OFFSET = 0.15
SCATTER_MARKER_SIZE = 2
SCATTER_MARKER_ALPHA = 0.8
VIOLIN_ALPHA = 1
ALPHA_STEP = 0.3
LINE_WIDTH = 0.5


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
                 marker_alpha=SCATTER_MARKER_ALPHA, x_values=None):
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
            data = [y if not (np.isnan(y)) else 0 for y in data]  # get rid of nans
            if not(all(isinstance(x, (int, float)) for x in data)):
                data = np.where(data, 1, 0)  # replace True with 1, False with 0
            # violin_plot: just a single violin!
            violin = plt.violinplot(data, positions=[positions_list[c]], showmeans=False, showextrema=False,
                                    showmedians=False)
            # get the color of the first violin to be the legend color
            if legend_flag == 0:
                vio = violin['bodies'][0]
                legend_flag = 1
            # make it a half-violin plot (only to the LEFT of center)
            b = violin['bodies'][0]  # single violin = single body
            b.set_alpha(violin_alpha)
            m = np.mean(b.get_paths()[0].vertices[:, 0])  # get the center
            # modify the paths to not go further right than the center
            b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], -np.inf, m)
            if x_col_color_order is not None:
                b.set_color(x_col_color_order[c])
            # then scatter
            scat_x = (np.ones(len(data)) * positions_list[c]) + VIOLIN_OFFSET + (
                    np.random.rand(len(data)) * JITTER_WIDTH / 2.)
            plt.scatter(x=scat_x, y=data, marker="o", color=x_col_color_order[c], alpha=marker_alpha,
                        s=SCATTER_MARKER_SIZE)
            # complete with a boxplot
            plt.boxplot(data, positions=[positions_list[c] + VIOLIN_OFFSET], notch=False,
                        medianprops=dict(color='black', linewidth=LINE_WIDTH),
                        showfliers=False)
    return vio


def plot_raincloud(df, x_col_name, y_col_name, plot_title, plot_x_name, plot_y_name, save_path, save_name,
                   x_col_color_order=None, y_tick_interval=5, y_tick_min=None, y_tick_max=None,
                   x_axis_names=None, y_tick_names=None, group_col_name=None, group_name_mapping=None, x_values=None,
                   add_horizontal_line=None):
    gc.collect()
    plt.clf()
    plt.figure()
    sns.reset_orig()

    if group_col_name is not None:
        labels = list()
        valpha = VIOLIN_ALPHA
        malpha = SCATTER_MARKER_ALPHA
        group_name = re.sub('([A-Z])', r' \1', group_col_name).title()
        i = 0
        for val in df[group_col_name].unique():
            if group_name_mapping is not None:
                if not isinstance(val, str):
                    label = group_name_mapping[str(val)]
                else:
                    label = group_name_mapping[val]
            else:
                label = str(val)
            data = df[df[group_col_name] == val]
            vio = make_it_rain(data, x_col_name, y_col_name, x_col_color_order[i], violin_alpha=valpha,
                               marker_alpha=malpha, x_values=x_values)
            valpha -= ALPHA_STEP
            malpha -= ALPHA_STEP
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
    plt.ylabel(plot_y_name, fontsize=F_AXES_TITLE, labelpad=LABELPAD-1.5)

    if add_horizontal_line is not None:
        plt.axhline(y=add_horizontal_line, color='slategrey', linestyle='--', linewidth=1)

    figure = plt.gcf()  # get current figure
    if len(df[x_col_name].unique().tolist()) > 10:  # a wide plot
        figure.set_size_inches(20, 10)
    figure.savefig(os.path.join(save_path, f"{RAIN}_{save_name}.png"), dpi=DPI, bbox_inches='tight')

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
