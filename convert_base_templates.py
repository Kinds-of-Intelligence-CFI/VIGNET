import json
import os
from project_scripts.base_templates import templates
from modify_context_template import CONTEXT_TEMPLATE



# the purpose of this file is to convert the base template dictionary to a set of json files each in a sperate dir names after the key in the templates dict

for key, sub_dict in templates.items():
    # make the dir for the template
    output_dir = os.path.join("INTUIT", key)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # if template does nto already have a field for coinflips add one
    possible_versions = ["A", "B"] 
    for version in possible_versions:
        if "coinflips" not in sub_dict[version].keys():
            sub_dict[version]["coinflips"] = {}

    output_file_path = os.path.join(output_dir, f"{key}_{CONTEXT_TEMPLATE}")
    with open(output_file_path, "w") as outfile:
        json.dump(sub_dict, outfile, indent=4)


