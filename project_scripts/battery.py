import os
import pandas as pd
import random as rd
import numpy as np
from itertools import product

from project_scripts.compiler import Compiler

path = os.getcwd()
results_path = os.path.join(path, r"results")


class batteryCreator:
    """
    Create a battery of vignettes from a specification dataframe
    """

    def __init__(self, content):
        self.content = content
        print("Battery complete.")

    @classmethod
    def create_battery(cls, battery_spec, difficulty_spec, template_df, variable_df, weights_df,
                       item_df, activity_df, demand_df, master_seed=42):
        """
        Create a battery of vignettes based on specifications.

        Parameters:
        -----------
        battery_spec : DataFrame
            Specifications for the battery
        difficulty_spec : dict
            Specifications for difficulty levels
        template_df : DataFrame
            Template dataframe
        variable_df : DataFrame
            Variable dataframe
        weights_df : DataFrame
            Weights dataframe
        item_df : DataFrame
            Item dataframe
        activity_df : DataFrame
            Activity dataframe
        demand_df : DataFrame
            Demand dataframe
        master_seed : int, optional
            Master random seed for reproducibility, by default 42
        """
        np.random.seed(master_seed)
        rd.seed(master_seed)

        cls.battery_spec = battery_spec
        cls.list_spec = False
        cls.difficulty_spec = difficulty_spec
        cls.template_df = template_df
        cls.variable_df = variable_df
        cls.weights_df = weights_df
        cls.item_df = item_df
        cls.activity_df = activity_df
        cls.demand_df = demand_df
        cls.used_templates = []
        cls.master_seed = master_seed
        cls.compilation_counter = 0

        cls.content = pd.DataFrame(columns=[
            "target_demand", "id", "number", "name", "version",
            "capability_type", "condition", "demand_condition", "vignette",
            "inference_level", "pov", "spacing_noise_proportion",
            "character_noise_number", "character_noise_proportion",
            "capitalisation_number", "capitalisation_proportion",
            "novelty", "causal_type", "causal_direction",
            "correct_answer_number", "correct_answer_text"])
        cls.check_difficulty_for_iterables()
        cls.demand_df["inference_component"] = cls.replace_space_with_underscore(cls.demand_df["inference_component"])
        for i, target_name in enumerate(cls.battery_spec["target_name"]):
            if target_name == "none":
                id_strings = cls.battery_spec.loc[0, "ids"]
            else:
                cls.demands = cls._get_demands(i)
                id_strings = cls._get_id_strings(target_name)
            for id_string in id_strings:
                print(f"Current ID: {id_string}")
                cls.v_info = cls._get_template_info(id_string)
                cls.define_conditions(id_string)
                difficulty_params = product(
                    cls.difficulty_spec["inference_level"],
                    cls.difficulty_spec["pov"],
                    cls.difficulty_spec["spacing_noise_proportion"],
                    cls.difficulty_spec["character_noise_number"],
                    cls.difficulty_spec["character_noise_proportion"],
                    cls.difficulty_spec["capitalisation_number"],
                    cls.difficulty_spec["capitalisation_proportion"],
                    cls.conditions)
                for il, p, snp, cnn, cnp, cn, cp, c in difficulty_params:
                    difficulty = {
                        "third_person": p,
                        "inference_level": il,
                        "double_spaces": snp,
                        "character_noise": (cnn, cnp),
                        "capitalisation": (cn, cp)}
                    mini_battery = cls._create_mini_battery(
                        target_name, id_string, c, difficulty, i)
                    cls.content = pd.concat(
                        [cls.content, mini_battery],
                        axis=0).reset_index(drop=True)
        return cls(cls.content)

    @classmethod
    def _get_template_info(cls, id_string):
        return (cls.template_df[cls.template_df["id"] == id_string]
                .reset_index(drop=True)
                .iloc[0]
                .to_dict())

    @classmethod
    def _generate_compiler_seed(cls):
        """
        Generate a deterministic but unique seed for each Compiler instance.
        """
        cls.compilation_counter += 1
        return cls.master_seed + cls.compilation_counter * 1000

    @classmethod
    def _create_mini_battery(cls, target_name, id_string, condition, difficulty, index):
        n = cls.battery_spec.loc[index, "label_variation_number"]
        demand_dict = cls.v_info["demands"]
        demand_list = cls.compute_demand_list(demand_dict, condition)
        demand_cols = cls.demand_df.columns[1:]
        demand_vector = cls.compute_demand_vector(demand_list, len(demand_cols))
        demand_matrix = pd.DataFrame(
            np.tile(demand_vector, (n, 1)),
            columns=demand_cols)
        compiler_seed = cls._generate_compiler_seed()

        vignette = Compiler(
            id_string, condition, difficulty,
            cls.template_df, cls.variable_df, cls.weights_df,
            cls.item_df, cls.activity_df, random_seed=compiler_seed)

        mini_battery = vignette.compile_battery(n)
        mini_battery['target_demand'] = target_name
        mini_battery['id'] = id_string
        mini_battery['number'] = cls.v_info["number"]
        mini_battery['name'] = cls.v_info["name"]
        mini_battery['version'] = cls.v_info["version"]
        mini_battery['capability_type'] = cls.v_info["capability_type"]
        mini_battery['condition'] = condition
        mini_battery['demand_condition'] = cls.demand_code
        mini_battery['inference_level'] = difficulty["inference_level"]
        mini_battery['pov'] = difficulty["third_person"]
        mini_battery['spacing_noise_proportion'] = difficulty["double_spaces"]
        mini_battery['character_noise_number'] = difficulty["character_noise"][0]
        mini_battery['character_noise_proportion'] = difficulty["character_noise"][1]
        mini_battery['capitalisation_number'] = difficulty["capitalisation"][0]
        mini_battery['capitalisation_proportion'] = difficulty["capitalisation"][1]
        mini_battery['novelty'] = cls.v_info["metadata"]["novelty"]
        mini_battery['causal_type'] = cls.v_info["metadata"]["causal_type"]
        mini_battery['causal_direction'] = cls.v_info["metadata"]["causal_direction"]
        return pd.concat([mini_battery, demand_matrix], axis=1).reset_index(drop=True)

    @classmethod
    def _randomise_rows(cls, df, weights=None):
        return df.sample(frac=1, weights=weights, random_state=np.random.randint(0, 100000)).reset_index(drop=True)

    @classmethod
    def replace_space_with_underscore(cls, data):
        return [x.replace(" ", "_") for x in data]

    @classmethod
    def _get_demands(cls, i):
        demands = cls.battery_spec.loc[i, "demand_profile"]
        curr_demand_df = cls.demand_df.copy()
        if curr_demand_df.empty:
            raise Exception("There are no eligible vignettes.")
        for demand in demands.keys():
            curr_demand_df = curr_demand_df[curr_demand_df[demand] == demands[demand]]
            curr_demand_df = cls._randomise_rows(curr_demand_df)
        return list(curr_demand_df["inference_component"])

    @classmethod
    def _get_id_strings(cls, target_name):
        base_ts = cls.template_df[cls.template_df["capability_type"] == "single"].reset_index(drop=True)
        eligible_ts = (
            base_ts[base_ts['demands'].apply(lambda x: bool(set(x['c1']) & set(cls.demands)))])
        base_ids = np.unique(eligible_ts["number"])
        available_ids = list(set(base_ids) - set(cls.used_templates))
        rd.shuffle(available_ids)
        if not available_ids:
            raise Exception("Battery creation failed on:\n" + target_name)
        selected_id = available_ids[0]
        cls.used_templates.append(selected_id)
        return cls.template_df.loc[cls.template_df["number"] == selected_id, "id"]

    @classmethod
    def define_conditions(cls, id_string):
        if id_string.split(".")[1] == "2":
            cls.conditions = ["00", "10", "01", "11"]
        else:
            cls.conditions = ["0", "1"]

    @classmethod
    def check_difficulty_for_iterables(cls):
        for key, value in cls.difficulty_spec.items():
            try:
                x = iter(value)
            except TypeError:
                cls.difficulty_spec[key] = [value]

    @classmethod
    def compute_demand_list(cls, demand_dict, condition):
        demand_list = demand_dict["c0"].copy()
        cls.demand_code = "c0"
        if condition in ["1", "10", "11"]:
            demand_list += demand_dict["c1"]
            cls.demand_code += " + c1"
        if condition in ["01", "11"]:
            demand_list += demand_dict["c2"]
            cls.demand_code += " + c2"
        return demand_list

    @classmethod
    def compute_demand_vector(cls, demand_list, vector_length):
        zero_vector = np.zeros([1, vector_length], dtype=int)
        if any(demand_list):
            for d in demand_list:
                new_demand = cls.demand_df[cls.demand_df["inference_component"] == d]
                zero_vector += new_demand.iloc[0, 1:].to_numpy(dtype=int)
        return zero_vector