import pandas as pd
import scipy
from scipy.stats import norm
import math
import random
import numpy as np
import sys

p_val = 0.05


def progress(count, total, status='Progress'):
    # credit to @vladignatyev, taken from : https://gist.github.com/vladignatyev/06860ec2040cb497f0f3
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.1 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write(f"\r[{bar}] {percents}% ...{status}")
    sys.stdout.flush()
    return


def permutation_cluster_paired_ttest(data_cond_1, data_cond_2, p_threshold=0.05, permutations=1024):
    """
    Perform a cluster permutation test PAIRED t-test, based on this paper:
    Eric Maris and Robert Oostenveld. Nonparametric statistical testing of EEG- and MEG-data.
    Journal of Neuroscience Methods, 164(1):177–190, 2007. doi:10.1016/j.jneumeth.2007.03.024.
    NOTE : this method assumes data_cond_1 and data_cond_2 contain the SAME COLUMNS, each column represents a subject
    and the same rows (each row represents a timepoint), with the only thing changing between 1 and 2 is the values.

    The way this function works:
    based on : https://benediktehinger.de/blog/science/statistics-cluster-permutation-test/, adjusted for TWO SIDED
    t-test and for acommodation of more than 1 cluster
    - Step 1: calculate the difference between conditions per time point
    - Step 2: test statistics  (t-test)
    - Step 3: clusters over time: define clusters by an arbitrary threshold and test whether these clusters are larger
    clusters that occur by chance. We do not want to do a statistical test for each time-point individually, because we
    would need to correct for multiple comparison for all timepoints. Instead, we define clusters by an arbitrary
    threshold (based on p_threshold) and test whether these clusters are larger clusters that occur by chance.
    - Step 4: calculate cluster-level statistics by taking the SUM of the t_values within each cluster
    (Maris & Oostenveld). The cluster mass is the largest of the cluster-level statistics (if there are multiple
    clusters).
    - Step 5: permutation of the data: shuffle the condition-label for each subject. Note that we actually do not need
    to go back to the two conditions, but we could just flip (multiply by -1) randomly every subject-difference curve in
    diff_df, and calculate the cluster size.

    How is the t-threshold calculated:
    The P-value (p_threshold) would be the probability under the density curve of Student's t distribution with n−1
    degrees of freedom beyond the observed t statistic. The value t we wish to reclaim from the reported p is then the
    inverse CDF (quantile) function of 1−p.
    # ppf: Percent Point Function (Inverse of CDF)
    # https://docs.scipy.org/doc/scipy/reference/tutorial/stats.html

    :param data_cond_1: dataframe where columns are subjects and rows are timepoints, in the first condition
    :param data_cond_2: dataframe where columns are subjects and rows are timepoints, in the second condition
    :param permutations: number of permutations to perform
    :param tail: 0 two tail, 1 one tail - right now not in use as I assume TWO TAIL
    :return:
    """

    # Calculate the difference between conditions per time point
    diff_df = pd.DataFrame()  # create a df that is the difference between data_cond 1 and 2 (= 2 - 1)
    for i in range(data_cond_1.shape[1]):  # iterate over columns = subjects (rows=timepoints)
        sub_name = data_cond_1.columns[i]
        sub_cond_1 = data_cond_1.iloc[:, i]
        sub_cond_2 = data_cond_2.iloc[:, i]
        sub_diff = sub_cond_2 - sub_cond_1
        diff_df.loc[:, sub_name] = sub_diff

    # Calculate the appropriate t-value threshold:
    deg_of_freedom = data_cond_1.shape[1]  # amount of subjects = amount of columns in original dataframe
    actual_p = p_threshold / 2  # TWO-SIDED HYPOTHESIS!!!!
    t_threshold_pos = abs(scipy.stats.t.ppf([actual_p], [deg_of_freedom])[0])
    t_threshold_neg = -t_threshold_pos

    # first, let's calculate the observed cluster/s that we have in the data. We will then test our permutation clusters
    # against EACH observed cluster
    ttest_sums, largest_cluster_ttest, cluster_full_df = ttest_clusters_over_time(diff_df=diff_df,
                                                                                    t_threshold_pos=t_threshold_pos,
                                                                                    t_threshold_neg=t_threshold_neg)
    # Permutations
    permutation_clusters_list = list()
    cluster_mass_list = list()
    for i in range(permutations):
        progress(count=i, total=permutations, status='Permutation cluster mass t-test progress')
        # create dataframe of diffs
        # Shuffle the condition-label for each SUBJECT (cond1 turns to cond 2 and vice versa) by flipping
        # (multiplying by -1) randomly every subject-difference curve
        perm_diff_df = diff_df.copy()
        for col in perm_diff_df.columns:
            perm_diff_df[col] = perm_diff_df[col] * random.choice([-1, 1])
        sum_df, cluster_mass, perm_cluster_full_df = ttest_clusters_over_time(diff_df=perm_diff_df,
                                                                              t_threshold_pos=t_threshold_pos,
                                                                              t_threshold_neg=t_threshold_neg)
        del perm_cluster_full_df  # we don't use it here
        # add to lists
        permutation_clusters_list.append(sum_df)
        cluster_mass_list.append(cluster_mass)

    # We now check whether our observed cluster mass (first run) is greater than p_threshold% of what we would expect
    # by chance (permutations)
    clusters = list()
    clusters_mass_size = list()
    cluster_p_vals = list()
    clusters_starts = list()
    clusters_ends = list()
    sig_list = list()
    for ind, row in ttest_sums.iterrows():
        # for each cluster we originally observed, check if its cluster mass is greater than the expected p_threshold%
        if ind != 0:  # 0 is NOT a cluster
            check_tail = [1 if row['tval'] > chance_mass else 0 for chance_mass in cluster_mass_list]
            # The exact value gives us the p-value of the cluster, the probability that cluster-mass with the observed size
            # would have occured when there was no actually difference between the conditions.
            probability = sum(check_tail) / len(check_tail)
            p = 1 - probability
            # div p-threshold because this is a TWO-TAIL hypothesis
            significance = f" < {p_threshold / 2} SIGNIFICANT" if p < (p_threshold / 2) else f" > {p_threshold / 2} NOT SIGNIFICANT"
            print(f"\nObserved cluster {ind}: mass={row['tval']:.5f} , cluster p-val={p}: {significance}")
            clusters.append(ind)
            cluster_p_vals.append(p)
            clusters_mass_size.append(row['tval'])
            clusters_starts.append(row['cluster_start'])
            clusters_ends.append(row['cluster_end'])
            sig_list.append(p < (p_threshold / 2))
    clusters_pvals = pd.DataFrame({"cluster": clusters, "mass_size": clusters_mass_size, "pval": cluster_p_vals,
                                   "significant": sig_list, "cluster_starts": clusters_starts, "cluster_ends": clusters_ends})
    cluster_mass_df = pd.DataFrame({"cluster_mass_in_permutation": cluster_mass_list})

    return clusters_pvals, cluster_full_df, cluster_mass_df


def ttest_clusters_over_time(diff_df, t_threshold_pos, t_threshold_neg):
    # calculate the test statistics per timepoint
    diff_df["mean"] = diff_df.mean(axis=1)  # average per row (timepoint) across columns (subject)
    diff_df["std"] = diff_df.std(axis=1)  # std per row (timepoint) across columns (subject)
    diff_df["tval"] = diff_df.apply(t_stat, axis=1)  # our test statistics


    # clusters over time. Cluster-mass statistics : the SUM of t-values
    diff_df["is_above_threshold"] = diff_df.apply(lambda row: thresh_check(row["tval"], t_threshold_pos,
                                                                                     t_threshold_neg), axis=1)
    perm_diff_df = clusters(diff_df)  # separate clusters
    # in sum_df, row per cluster (0 is NOT a cluster), tval is the sum of tvals in this cluster
    # (observed cluster mass)

    perm_diff_df['ind'] = perm_diff_df.index
    cluster_start = perm_diff_df.groupby(['cluster']).ind.idxmin()
    cluster_end = perm_diff_df.groupby(['cluster']).ind.idxmax()
    sum_df = perm_diff_df.groupby(['cluster']).apply(lambda c: c.abs().sum())  # TWO-SIDED HYPOTHESIS: ABSOLUTE SUM
    # If sum_df["is_above_threshold"] == 0 THEN THIS IS -NOT- A CLUSTER! NOTHING WAS ABOVE THRESHOLD
    sum_df.loc[sum_df.is_above_threshold == 0, ['tval']] = 0
    sum_df = sum_df["tval"].to_frame()  # only summation of tvals of the clusters is interesting
    for index, value in cluster_start.items():
        sum_df.loc[index, 'cluster_start'] = value
    for index, value in cluster_end.items():
        sum_df.loc[index, 'cluster_end'] = value
    # the cluster mass is the largest of the cluster-level statistics (if there are multiple clusters)
    cluster_mass = max(sum_df["tval"])
    return sum_df, cluster_mass, perm_diff_df


def t_stat(row):
    excess_columns = 2  # "mean" and "std" columns
    n = len(row) - excess_columns  # all subject columns
    if row["std"] == 0 or math.sqrt(n) == 0:
        t = 0
    else:
        t = row["mean"] / (row["std"] / math.sqrt(n))
    return t


def thresh_check(tval, thresh_pos=None, thresh_neg=None):
    # return 1 if tval is above threshold, 0 otherwise
    result = 0
    if thresh_pos is None:
        result = 1 if tval < thresh_neg else 0
    if thresh_neg is None:
        result = 1 if tval > thresh_pos else 0
    elif thresh_pos is not None and thresh_neg is not None:  # both thresh_pos and thresh_neg
        result = 1 if (tval > thresh_pos or tval < thresh_neg) else 0
    return result


def clusters(cluster_dataframe, cluster_col="is_above_threshold", tval_col="tval"):
    cluster_list = list()
    in_cluster = False  # boolean, are we right now in a cluster or not
    cluster_number = 0  # the index of the current (or last) cluster
    # from Maris & Oostenveld: "for a two-sided test, the clustering is performed separately for samples with a
    # positive and a negative t-value."
    cluster_sign = None  # the SIGN of the cluster

    for ind, row in cluster_dataframe.iterrows():
        if row[cluster_col] == 1:  # we are creating/in a cluster
            if not in_cluster:  # there wasn't an active cluster
                in_cluster = True
                cluster_number += 1
                cluster_sign = np.sign(row[tval_col])  # either 1 (pos), 0 (0), or -1 (neg)
            if in_cluster:  # there was an active cluster
                if cluster_sign is None:  # first time we have a cluster in the data
                    cluster_sign = np.sign(row[tval_col])
                elif cluster_sign != np.sign(row[tval_col]):  # if current cluster sign is NOT identical to our sign
                    in_cluster = True
                    cluster_number += 1  # we are in a SEPARATE cluster, with our sign now
                    cluster_sign = np.sign(row[tval_col])
            # but anyway:
            curr = cluster_number
        else:
            curr = 0  # not a cluster at all
            in_cluster = False
        cluster_list.append(curr)
    cluster_dataframe["cluster"] = cluster_list
    return cluster_dataframe
