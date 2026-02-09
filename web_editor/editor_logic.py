"""Pure-function business logic extracted from the tkinter JSONEditorGUI.

Every function here operates on plain Python dicts/lists and an EditorState --
no UI dependencies.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

import pandas as pd

from web_editor.state import (
    CONTEXT_TEMPLATE,
    QA_TEMPLATE,
    FORMAT_CHECKED_KEYS,
    NEVER_DELETE_KEYS,
    POTENTIAL_VERSIONS,
    EditorState,
)
from web_editor.tree_manager import collect_leaf_values


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def load_demand_data(path: str, state: EditorState) -> str | None:
    """Load demand CSV and update state.valid_values.  Returns error string or None."""
    try:
        if not path:
            return "Demand file path is empty."
        demand_data = pd.read_csv(path)
        demands = sorted(demand_data[demand_data.columns[0]].unique())
        demands_formatted = [d.replace(" ", "_") for d in demands]
        for key in ["provided", "required", "c0", "c1", "c2", "c3", "c4"]:
            state.valid_values[key] = demands_formatted
        state.current_demand_file = path
        return None
    except Exception as exc:
        return str(exc)


def load_items_data(path: str, state: EditorState) -> str | None:
    try:
        if not path:
            return "Item file path is empty."
        item_data = pd.read_csv(path)
        item_props = sorted(item_data.columns[1:])
        state.valid_values["items"] = [p.replace(" ", "_") for p in item_props]
        state.current_item_file = path
        return None
    except Exception as exc:
        return str(exc)


def load_activities_data(path: str, state: EditorState) -> str | None:
    try:
        if not path:
            return "Activity file path is empty."
        activity_data = pd.read_csv(path)
        activity_props = sorted(activity_data.columns[1:])
        state.valid_values["activities"] = [p.replace(" ", "_") for p in activity_props]
        state.current_activity_file = path
        return None
    except Exception as exc:
        return str(exc)


def load_variable_data(path: str, state: EditorState) -> str | None:
    try:
        if not path:
            return "Variable file path is empty."
        variable_data = pd.read_csv(path)
        state.variable_options = sorted(variable_data.columns[1:])
        state.current_variable_file = path
        return None
    except Exception as exc:
        return str(exc)


def load_weights_data(path: str, state: EditorState) -> str | None:
    try:
        if not path:
            return "Weights file path is empty."
        state.current_weights_file = path
        return None
    except Exception as exc:
        return str(exc)


# ---------------------------------------------------------------------------
# Populate variables from the loaded JSON template
# ---------------------------------------------------------------------------

def populate_variables(state: EditorState) -> None:
    """Read the current json_data (or associated context template) and fill
    state.context_variables / filler / items / activities / switches / povs."""
    state.povs = [f"pov_{x}" for x in range(10)]

    if state.is_context_template():
        tree_data = state.json_data
    else:
        tree_data = get_context_template_dict(state)

    if tree_data is None:
        return

    for version in POTENTIAL_VERSIONS:
        if version not in tree_data:
            continue
        state.context_variables[version] = tree_data[version]["variables"]
        filler_keys = tree_data[version]["filler"].keys()
        state.filler[version] = [f"filler_{x}" for x in filler_keys]
        state.items[version] = list(tree_data[version]["items"].keys())
        state.activities[version] = list(tree_data[version]["activities"].keys())
        state.switches[version] = list(tree_data[version]["switches"].keys())


def get_context_template_dict(state: EditorState) -> dict | None:
    """For a QA template, load and return the associated context template dict."""
    if state.json_data is None or state.current_file is None:
        return None
    if QA_TEMPLATE not in state.current_file:
        return None
    parent_dir = os.path.dirname(state.current_file)
    base_context = state.json_data.get("base_context", "")
    context_path = os.path.join(parent_dir, f"{base_context}_{CONTEXT_TEMPLATE}")
    if os.path.exists(context_path):
        with open(context_path, "r") as f:
            return json.load(f)
    return None


# ---------------------------------------------------------------------------
# Value conversion
# ---------------------------------------------------------------------------

def value_from_string(value_str: str, type_str: str) -> Any:
    """Convert an editor string + type tag to a proper Python value."""
    if type_str == "string":
        return value_str
    elif type_str == "number":
        try:
            return float(value_str) if "." in value_str else int(value_str)
        except ValueError:
            return 0
    elif type_str == "boolean":
        return value_str.lower() == "true"
    elif type_str == "null":
        return None
    elif type_str == "object":
        return {}
    elif type_str == "array":
        return []
    return value_str


def type_of_value(value: Any) -> str:
    """Return the type tag string for a JSON value."""
    if isinstance(value, dict):
        return "object"
    elif isinstance(value, list):
        return "array"
    elif isinstance(value, bool):
        return "boolean"
    elif isinstance(value, (int, float)):
        return "number"
    elif value is None:
        return "null"
    return "string"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def find_variable(var_name: str, version: str, state: EditorState) -> bool:
    """Check whether *var_name* is defined anywhere in the template for *version*."""
    if version in state.context_variables and var_name in state.context_variables[version]:
        return True
    if version in state.filler and var_name in state.filler[version]:
        return True
    if version in state.items and var_name in state.items[version]:
        return True
    if version in state.activities and var_name in state.activities[version]:
        return True
    if version in state.switches and var_name in state.switches[version]:
        return True
    if var_name in state.povs:
        return True
    return False


def validate_var_deletion(value: Any, del_var: str) -> bool:
    """Return True if deleting *del_var* would NOT break the string *value*."""
    if not isinstance(value, str):
        return True
    var_names = re.findall(r"vs\[(.*?)\]", value)
    return del_var not in var_names


def validate_str_format(value: Any, version: str | None, state: EditorState) -> tuple[bool, str]:
    """Validate that every vs[...] reference in *value* can be resolved."""
    if not isinstance(value, str):
        return True, "cannot test non string"
    var_names = re.findall(r"vs\[(.*?)\]", value)
    for var_name in var_names:
        if version is not None and not find_variable(var_name, version, state):
            return False, f"Could not find variable: {var_name}"
    return True, f"found all variables in string:\n{value}"


def validate_update(
    key: str,
    value: Any,
    parent_keys: list[str],
    parent_key: str,
    state: EditorState,
) -> tuple[bool, str]:
    """Validate that setting *key*=*value* under *parent_key* is allowed.

    *parent_keys* is the list of ancestor keys from parent up to (but excluding)
    root.  This mirrors the old get_parent_keys() output.
    """
    all_keys = list(parent_keys) + [parent_key, key]

    version = None
    if state.is_context_template():
        for sub_key in all_keys:
            if sub_key in POTENTIAL_VERSIONS:
                version = sub_key
                break
    else:
        if state.json_data is not None:
            version = state.json_data.get("base_context_version")

    # check valid_values
    test_valid_values = False
    key_to_test = None
    for sub_key in all_keys:
        if sub_key in state.valid_values:
            test_valid_values = True
            key_to_test = sub_key
            break

    if test_valid_values:
        if value in state.valid_values[key_to_test]:
            return True, f"found key: {key_to_test}, with value: {value}"
        elif key in state.valid_values[key_to_test]:
            return True, f"found key: {key_to_test}, with value: {key}"
        else:
            return False, (
                f"found key: {key_to_test}, but not value: {value} or key: {key} "
                f"in valid keys:\n {state.valid_values[key_to_test]}"
            )

    for sub_key in all_keys:
        if sub_key in FORMAT_CHECKED_KEYS:
            return validate_str_format(value, version, state)

    if key == "variables":
        if value not in state.variable_options:
            if version is not None and not find_variable(value, version, state):
                return False, f"variable: {value} not found"

    if "links" in all_keys:
        context_dict = (
            state.json_data
            if state.is_context_template()
            else get_context_template_dict(state)
        )
        if context_dict is not None and version is not None:
            switch_key = f"switch_{int(key) + 1}"
            switches = context_dict.get(version, {}).get("switches", {})
            if switch_key in switches:
                num_vals = len(switches[switch_key])
                return (
                    num_vals > int(value),
                    f"switch: {switch_key} contains {num_vals} values, tried to set value to {value}",
                )

    return True, f"key: {key}, not tracked in {list(state.valid_values.keys())}"


def validate_delete(
    del_key: str,
    del_value: Any,
    parent_keys: list[str],
    state: EditorState,
) -> tuple[bool, str]:
    """Check whether deleting *del_key* is allowed.  Returns (ok, reason)."""
    all_never = NEVER_DELETE_KEYS + list(state.valid_values.keys())
    if del_key in all_never:
        return False, f"Cannot delete key: {del_key} -- it is needed in the structure of the JSON file."
    return True, ""


# ---------------------------------------------------------------------------
# Cascading operations on QA template link/answer tables
# ---------------------------------------------------------------------------

def _qa_template_paths_in_dir(current_file: str) -> list[str]:
    """Return full paths to all QA template files in the same directory."""
    current_dir = os.path.dirname(current_file)
    paths = []
    for name in os.listdir(current_dir):
        if "_diff" not in name and "_context" not in name and name.endswith(".json"):
            paths.append(os.path.join(current_dir, name))
    return paths


def append_new_switch_to_qa_files(state: EditorState, version: str) -> None:
    """When a new switch is added in the context template, append 0 to every
    link table in the sibling QA templates."""
    for qa_path in _qa_template_paths_in_dir(state.current_file):
        with open(qa_path, "r") as f:
            qa = json.load(f)
        for _key, link_table in qa.get("links", {}).items():
            link_table.append(0)
        with open(qa_path, "w") as f:
            json.dump(qa, f, indent=4)


def delete_switch_from_qa_files(state: EditorState, deleted_index: int) -> None:
    """When a switch is deleted, remove its column from every QA link table."""
    for qa_path in _qa_template_paths_in_dir(state.current_file):
        with open(qa_path, "r") as f:
            qa = json.load(f)
        for _key, link_table in qa.get("links", {}).items():
            if deleted_index - 1 < len(link_table):
                del link_table[deleted_index - 1]
        with open(qa_path, "w") as f:
            json.dump(qa, f, indent=4)


def delete_switch_value_from_qa_files(
    state: EditorState,
    switch_name: str,
    del_index: int,
    num_values: int,
) -> None:
    """When a value is removed from a switch array, decrement any link table
    entries that pointed at or beyond the deleted index."""
    switch_index = int(switch_name.split("_")[1]) - 1
    del_value = int(del_index)

    for qa_path in _qa_template_paths_in_dir(state.current_file):
        with open(qa_path, "r") as f:
            qa = json.load(f)
        for _key, link_table in qa.get("links", {}).items():
            if switch_index < len(link_table):
                if link_table[switch_index] > del_value:
                    link_table[switch_index] -= 1
                elif link_table[switch_index] == del_value and del_value == num_values - 1:
                    link_table[switch_index] -= 1
        with open(qa_path, "w") as f:
            json.dump(qa, f, indent=4)


# ---------------------------------------------------------------------------
# Switch reindexing inside json_data
# ---------------------------------------------------------------------------

def reindex_switches_after_delete(switches_dict: dict, deleted_key: str) -> None:
    """After deleting *deleted_key* (e.g. "switch_3") from a switches dict,
    rename all higher-numbered switches to fill the gap.

    Mutates *switches_dict* in place.
    """
    deleted_index = int(deleted_key.split("_")[1])
    keys_to_rename = []
    for k in list(switches_dict.keys()):
        idx = int(k.split("_")[1])
        if idx > deleted_index:
            keys_to_rename.append((k, idx))
    keys_to_rename.sort(key=lambda t: t[1])
    for old_key, idx in keys_to_rename:
        new_key = f"switch_{idx - 1}"
        switches_dict[new_key] = switches_dict.pop(old_key)


# ---------------------------------------------------------------------------
# Add / delete helpers (the complex special-case logic)
# ---------------------------------------------------------------------------

def handle_add_special_cases(
    json_data: dict,
    parent_keys: list[str],
    parent_key: str,
    new_key: str,
    new_value: Any,
    state: EditorState,
) -> tuple[bool, str]:
    """Process special cascading side-effects when adding an item.

    Returns (ok, error_message).  If ok is False, the add should be aborted.
    """
    if parent_key == "switches":
        if not isinstance(new_value, list):
            return False, "Switch values must be an array."
        version = None
        for k in parent_keys:
            if k in POTENTIAL_VERSIONS:
                version = k
                break
        if version is not None:
            append_new_switch_to_qa_files(state, version)

    elif "switches" in parent_keys:
        parent_path_keys = parent_keys  # ancestor keys from root to parent
        # find the parent array in json_data to check length
        # parent_key is the switch name (e.g. "switch_1"), parent holds the array
        # new_key should equal the current length
        version = None
        for k in parent_keys:
            if k in POTENTIAL_VERSIONS:
                version = k
                break
        if version is not None:
            switch_array = json_data[version]["switches"].get(parent_key, [])
            if int(new_key) != len(switch_array):
                return False, (
                    f"Cannot insert switch value in the middle, "
                    f"change key from {new_key} to {len(switch_array)}."
                )

    elif parent_key == "options":
        # adding a new answer option -- mirror into all answer tables
        answer_length = len(json_data.get("question", {}).get("options", []))
        if int(new_key) != answer_length:
            return False, (
                f"Cannot insert answer value in the middle, "
                f"change key from {new_key} to {answer_length}."
            )
        for _answer_key, answer_table in json_data.get("answers", {}).items():
            answer_table.append(0)

    elif parent_key == "links":
        # adding a new link table -- also add a matching answers table
        num_answers = len(json_data.get("question", {}).get("options", []))
        json_data.setdefault("answers", {})[new_key] = [0] * num_answers

    return True, ""


def handle_delete_special_cases(
    json_data: dict,
    parent_keys: list[str],
    del_key: str,
    del_value: Any,
    state: EditorState,
) -> tuple[bool, str]:
    """Process cascading side-effects when deleting an item.

    Returns (ok, error_message).
    """
    if state.is_context_template():
        version = None
        for k in parent_keys:
            if k in POTENTIAL_VERSIONS:
                version = k
                break

        if version is not None:
            # validate no string references to the deleted key/value
            version_data = json_data.get(version, {})
            all_values = collect_leaf_values(version_data)
            for val in all_values:
                if not validate_var_deletion(val, del_key):
                    return False, f"Cannot delete key: {del_key} -- it is referenced in text: {val}"
                if not validate_var_deletion(val, str(del_value)):
                    return False, f"Cannot delete value: {del_value} -- it is referenced in text: {val}"

            if parent_keys and parent_keys[0] == "switches":
                # deleting a whole switch
                deleted_index = int(del_key.split("_")[1])
                reindex_switches_after_delete(json_data[version]["switches"], del_key)
                delete_switch_from_qa_files(state, deleted_index)

            elif "switches" in parent_keys:
                # deleting a value from within a switch array
                switch_name = parent_keys[0]
                switch_array = json_data[version]["switches"].get(switch_name, [])
                num_values = len(switch_array)
                if num_values <= 1:
                    return False, "Cannot delete the last value in a switch."
                delete_switch_value_from_qa_files(state, switch_name, int(del_key), num_values)

    else:
        # QA template deletions
        # Check deeper nesting first so "links" -> "0" -> element is caught
        # before the top-level "links" -> table deletion.
        if len(parent_keys) >= 2 and "links" in parent_keys:
            return False, "Cannot delete item from a link table. Link tables are only modified from the context template."

        elif len(parent_keys) >= 2 and "answers" in parent_keys:
            return False, "Cannot delete item from an answer table. Answer tables are only modified by changing the options array."

        elif parent_keys and parent_keys[-1] == "links":
            json_data.get("answers", {}).pop(del_key, None)

        elif parent_keys and parent_keys[-1] == "answers":
            json_data.get("links", {}).pop(del_key, None)

        elif "options" in parent_keys:
            # removing an answer option -- shuffle answer tables
            del_idx = int(del_key)
            for _answer_key, answer_table in json_data.get("answers", {}).items():
                if del_idx < len(answer_table):
                    answer_table.pop(del_idx)

    return True, ""
