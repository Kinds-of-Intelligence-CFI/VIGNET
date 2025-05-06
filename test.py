import os
import pandas as pd
import torch

from project_scripts.experiments import Experimenter
from project_scripts.prompts import default_preprompt
from project_scripts.models import SelectModel
from project_scripts.prompts import  default_preprompt

path = os.getcwd()
results_path = os.path.join(path, r"results")
data_path = os.path.join(path, r"dataframes")

## load battery
battery_df = pd.read_csv(os.path.join(data_path, "battery_for_ai_clean_pov_0.csv"))

## select models
token = os.getenv("HUGGING_FACE_HUB_TOKEN")
model_list = [
    "meta-llama/Llama-3.2-3B-Instruct"
]
temperature_list = [0.3]

## test model
if torch.cuda.is_available():
    log = pd.Series(["Running on GPU."])
else:
    log = pd.Series(["Torch not working."])
model = SelectModel(model_list[0],
                    token,
                    set_gpu_limit=False)
answer = model.generate_text(prompt="Why is the sky blue?",
                             system_prompt="",
                             temperature=temperature_list[0])
log[1] = answer
log.to_csv(os.path.join(results_path,"test_log"),index=False)
