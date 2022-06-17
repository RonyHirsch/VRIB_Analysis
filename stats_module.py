import pingouin as pg
import os
import re

CORR = "corr_"
IND_T = "ttest_ind_"
MWU = "mwu_"


def correlation(data, col1, col2, save_path, corr_method="pearson", save_name=None):
    """
    :param data: dataframe where the data is at
    :param col1: first parameter
    :param col2: second parameter
    :param corr_method: possible corr_methods: 'pearson', 'spearman', 'kendall', 'bicor', 'percbend', 'shepherd',
    'skipped'
    :return: correlation between data[col1] and data[col2]
    """
    try:
        stat = pg.corr(x=data[col1], y=data[col2], alternative='two-sided', method=corr_method)
    except AssertionError:
        print(f"{corr_method} CORRELATION ON {col1},{col2} FAILED: data needs to contain more than 1 element")
        return
    stat = stat.rename(columns={"n": "N", "r": "Correlation_Coeff_r",
                                "CI95%": "CI95", "r2": "r_squared", "adg_r2": "Adjusted_r_squared"})
    print(f"{corr_method} CORRELATION ON {col1},{col2} : FULL MATRIX")
    if save_name is None:
        save_name = re.sub(r'(?<!^)(?=[A-Z])', '_', col1).lower() + "_" + re.sub(r'(?<!^)(?=[A-Z])', '_', col2).lower()
    stat.to_csv(os.path.join(save_path, f"{CORR}{save_name}.csv"))
    return stat


def independent_t_test(data, col1, col2, save_path, save_name):
    stat = pg.ttest(x=data[col1], y=data[col2], paired=False)
    stat.columns = stat.columns.str.replace('%', '')
    stat.to_csv(os.path.join(save_path, f"{IND_T}{save_name}.csv"))
    return stat


def mann_whitney_test(data, col1, col2, save_path, save_name):
    stat = pg.mwu(x=data[col1], y=data[col2])
    stat.to_csv(os.path.join(save_path, f"{MWU}{save_name}.csv"))
    return stat

