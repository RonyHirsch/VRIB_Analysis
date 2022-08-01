import unittest
import os
import pandas as pd
from pathlib import Path
import beh_analysis


class TestBehaviorAnalysis(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.all_subs_df = pd.read_csv(Path(r"D:\VRIB\round 1\ANALYSIS_TEMP\raw_all_subs.csv"))

    def test_pas_comparison_cond(self):
        # load this function's latest output
        total_df = pd.read_csv(Path(r"D:\VRIB\round 1\ANALYSIS_TEMP\beh\PAS_comp_all_JASP_struct.csv"))
        PAS_task_comp_df = pd.read_csv(Path(r"D:\VRIB\round 1\ANALYSIS_TEMP\beh\PAS_comp_task.csv"))
        PAS_task_val_comp_df = pd.read_csv(Path(r"D:\VRIB\round 1\ANALYSIS_TEMP\beh\PAS_comp_task_valence.csv"))

        # UAT - AT
        for cond in ["Unattended", "Attended"]:
            cond_comp = PAS_task_comp_df[PAS_task_comp_df["condition"] == cond]
            subjects_comp = set(cond_comp["Subject"].tolist())
            cond_total = total_df[["Subject", f"{cond}_1", f"{cond}_2", f"{cond}_3", f"{cond}_4"]]
            subjects_total = set(cond_total["Subject"].tolist())
            # Do we have the same subject set?
            self.assertEqual(subjects_comp, subjects_total, msg=f"ERROR: {cond} : subjects are not matching:\n{len(subjects_comp)} "
                                                                f"in 'PAS_task_comp_df', {len(subjects_total)} in 'total_df'"
                                                                f"; difference: {subjects_comp.difference(subjects_total)}")
            # For each subject, do the PAS scores in the
            for sub in subjects_comp:
                sub_PAS_total = cond_total[cond_total["Subject"] == sub]
                sub_PAS_comp = cond_comp[cond_comp["Subject"] == sub]
                for rating in range(1, 5):
                    sub_rating_total = sub_PAS_total.loc[:, f"{cond}_{rating}"].tolist()[0]
                    sub_rating_comp = sub_PAS_comp[sub_PAS_comp["subjectiveAwareness"] == rating]["Average_PAS"].tolist()[0]
                    self.assertEqual(sub_rating_total, sub_rating_comp, msg=f"ERROR: Subject {sub} {cond} condition rating {rating}: in 'PAS_task_comp_df'={sub_rating_comp}, in total_df'={sub_rating_total}")
                    c = 4

        # AVERSIVE - NEUTRAL



