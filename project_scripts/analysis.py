import os
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use("TKAgg")
import matplotlib.pyplot as plt

path = os.getcwd()
analysis_path = os.path.join(path, r"analysis")
plots_path = os.path.join(path, r"plots")

class Analyser:
    """
    Analyse and display results from an experiment
    """
    def __init__(self,results_df):
        self.results = results_df.copy()
        self.proportion_correct = None
        self.chance_correct = None

    def clean_text(self,remove_list):
        for char in remove_list:
            self.results["answer_text"] = [
                x.replace(char,"") for x in self.results["answer_text"].astype(str)]
            self.results["correct_answer_text"] = [
                x.replace(char,"") for x in self.results["correct_answer_text"].astype(str)]

    def check_answers(self,**kwargs):
        """
        Specify method: "either_text_or_number" or "both_text_and_number"
        """
        params = {"method": "just_number",
                  "wrong_format_answer": np.nan,
                  "print_proportion": True}
        for key, value in kwargs.items():
            params[key] = value
        self.results["answer_num"] = self.results["answer_num"].astype(str)
        accepted_numbers = ["1","2","3","4"]
        to_remove = [".","_"," ",'"',"'"] + accepted_numbers
        self.clean_text(to_remove)
        self.results.loc[~self.results["answer_num"].isin(accepted_numbers),"answer_num"] = 0
        self.results["answer_num"] = self.results["answer_num"].astype(int)
        self.results["answer_option"] = self.results.apply(
            lambda row: 0 if row['answer_num'] == 0 else row[f"option_{row['answer_num']}"],axis=1)
        self.results["correct_answer_option"] = self.results.apply(
            lambda row: row[f"option_{row['correct_answer_number']}"], axis=1)
        self.results["llm_num_correct"] = (
                self.results.loc[:,"answer_num"] == self.results["correct_answer_number"])
        self.results["llm_text_correct"] = (
                self.results.loc[:,"answer_text"] == self.results["correct_answer_text"])
        if params["method"] == "just_number":
            self.results.loc[:,"llm_correct"] = self.results["llm_num_correct"].astype(int)
        elif params["method"] == "both_text_and_number":
            self.results.loc[:,"llm_correct"] = (
                self.results["llm_num_correct"] & self.results["llm_text_correct"]).astype(int)
        elif params["method"] == "either_text_or_number":
            self.results.loc[:,"llm_correct"] = (
                self.results["llm_num_correct"] | self.results["llm_text_correct"]).astype(int)
        else:
            raise Exception("Incorrect method.")
        self.results.loc[self.results["answer_num"] == 0, "llm_correct"] = params["wrong_format_answer"]
        if params["print_proportion"]:
            self.proportion_correct = self.results["llm_correct"].sum() / len(self.results)
            self.chance_correct = 1 / len(accepted_numbers)
            if params["method"] == "both_text_and_number":
                self.chance_correct = self.chance_correct * self.chance_correct
            print("proportion correct: " + str(round(self.proportion_correct,2)))
            print("chance: " + str(round(self.chance_correct,2)))

    def plot_accuracy(self,by="target_demand",
                      and_by = None,
                      subset = None,
                      title = "",
                      save_fig = True,
                      save_path = plots_path,
                      dpi = 200,
                      palette = "Blues_d",
                      colour = "b",
                      **kwargs):
        params = {}
        for key, value in kwargs.items():
            params[key] = value
        plot_df = self.results.copy()
        plot_df["instance_number"] = range(len(plot_df))
        if subset:
            subset_cols = subset.keys()
            for col in subset_cols:
                if col == "id":
                    plot_df = plot_df[plot_df[col].str.startswith(tuple(subset[col]))]
                else:
                    plot_df = plot_df[plot_df[col].isin(subset[col])]
        if and_by:
            cols = ["instance_number", by, and_by, "llm_correct"]
        else:
            palette = None
            cols = ["instance_number", by, "llm_correct"]
        plot_df = plot_df[cols]
        sns.set(style="darkgrid",font_scale=1.5)
        plt.subplots(figsize=(15,10))
        sns.barplot(plot_df,
                    x=by,
                    y="llm_correct",
                    hue=and_by,
                    palette=palette,
                    color=colour,
                    errorbar=("ci",95),
                    capsize=0.1)
        plt.axhline(y=self.chance_correct,
                    color='black',
                    linestyle='--',
                    label='Chance Level')
        plt.title(title,fontsize=32)
        if save_fig:
            if and_by:
                save_name = title + " by " + by + " and " + and_by + ".jpg"
            else:
                save_name = title + " by " + by + ".jpg"
            save_name = save_name.replace(" ","_")
            plt.savefig(os.path.join(save_path, save_name),
                        bbox_inches='tight',
                        format="jpg",
                        dpi=dpi)

    def plot_histogram(self, **kwargs):
        params = {"subset": None,
                  "title": "",
                  "save_fig": True,
                  "save_path": plots_path,
                  "dpi": 200}
        for key, value in kwargs.items():
            params[key] = value
        plot_df = self.results.copy()
        if params["subset"]:
            subset_cols = params["subset"].keys()
            for col in subset_cols:
                if col == "id":
                    plot_df = plot_df[plot_df[col].str.startswith(tuple(params["subset"][col]))]
                else:
                    plot_df = plot_df[plot_df[col].isin(params["subset"][col])]
        plt.figure(figsize=(8, 6))
        value_counts = plot_df["answer_option"].value_counts().sort_index()
        correct_answers = np.unique(plot_df["correct_answer_option"])
        colours = ['orange' if answer in correct_answers else 'blue' for answer in value_counts.index]
        plt.figure(figsize=(8, 6))
        value_counts.plot(kind='bar',color=colours)
        plt.title(params["title"])
        plt.xlabel("Answer Number")
        plt.xticks(rotation=0)
        plt.ylabel("Frequency")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        save_name = params["title"] + "_histogram.jpg"
        plt.savefig(os.path.join(params["save_path"], save_name),
                    bbox_inches='tight',
                    format="jpg",
                    dpi=params["dpi"])









