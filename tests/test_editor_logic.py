import copy
import json
import os
import tempfile

import pytest

from web_editor.state import EditorState, CONTEXT_TEMPLATE, QA_TEMPLATE
from web_editor.editor_logic import (
    find_variable,
    validate_var_deletion,
    validate_str_format,
    validate_update,
    validate_delete,
    value_from_string,
    type_of_value,
    populate_variables,
    reindex_switches_after_delete,
    handle_add_special_cases,
    handle_delete_special_cases,
)


CONTEXT_DATA = {
    "metadata": {"novelty": 1, "causal_type": "intervention", "causal_direction": "forward",
                 "provided": ["agent_action_sequence"], "required": ["physical_knowledge_gravity"]},
    "A": {
        "context": "Hello {vs[name_1]} in {vs[city]}",
        "filler": {"0": "some filler", "1": "", "2": ""},
        "variables": ["name_1", "city", "filler"],
        "items": {"item_1": {"wearable": True}},
        "activities": {},
        "switches": {
            "switch_1": ["val_a", "val_b"],
            "switch_2": ["x", "y", "z"],
        },
        "coinflips": {},
    },
    "B": {
        "context": "Version B text",
        "filler": {"0": "b filler"},
        "variables": ["name_1", "filler"],
        "items": {},
        "activities": {"activity_1": {"two-person": True}},
        "switches": {"switch_1": ["ba", "bb"]},
        "coinflips": {},
    },
}


def _make_state(file_name="test_context_template.json") -> EditorState:
    s = EditorState()
    s.json_data = copy.deepcopy(CONTEXT_DATA)
    s.current_file = f"/tmp/{file_name}"
    s.variable_options = ["name_1", "city", "colour_1"]
    populate_variables(s)
    return s


class TestValueConversion:
    def test_string(self):
        assert value_from_string("hello", "string") == "hello"

    def test_number_int(self):
        assert value_from_string("42", "number") == 42

    def test_number_float(self):
        assert value_from_string("3.14", "number") == 3.14

    def test_boolean(self):
        assert value_from_string("true", "boolean") is True
        assert value_from_string("False", "boolean") is False

    def test_null(self):
        assert value_from_string("anything", "null") is None

    def test_object(self):
        assert value_from_string("", "object") == {}

    def test_array(self):
        assert value_from_string("", "array") == []


class TestTypeOfValue:
    def test_dict(self):
        assert type_of_value({}) == "object"

    def test_list(self):
        assert type_of_value([]) == "array"

    def test_bool(self):
        assert type_of_value(True) == "boolean"

    def test_int(self):
        assert type_of_value(42) == "number"

    def test_none(self):
        assert type_of_value(None) == "null"

    def test_str(self):
        assert type_of_value("hello") == "string"


class TestFindVariable:
    def test_in_variables(self):
        s = _make_state()
        assert find_variable("name_1", "A", s) is True

    def test_in_filler(self):
        s = _make_state()
        assert find_variable("filler_0", "A", s) is True

    def test_in_items(self):
        s = _make_state()
        assert find_variable("item_1", "A", s) is True

    def test_in_pov(self):
        s = _make_state()
        assert find_variable("pov_1", "A", s) is True

    def test_not_found(self):
        s = _make_state()
        assert find_variable("nonexistent", "A", s) is False


class TestValidateVarDeletion:
    def test_no_reference(self):
        assert validate_var_deletion("some plain text", "switch_1") is True

    def test_has_reference(self):
        assert validate_var_deletion("use {vs[switch_1]} here", "switch_1") is False

    def test_non_string(self):
        assert validate_var_deletion(42, "switch_1") is True


class TestValidateStrFormat:
    def test_all_found(self):
        s = _make_state()
        ok, _ = validate_str_format("Hello {vs[name_1]}", "A", s)
        assert ok is True

    def test_missing_var(self):
        s = _make_state()
        ok, reason = validate_str_format("Hello {vs[nonexistent_var]}", "A", s)
        assert ok is False
        assert "nonexistent_var" in reason


class TestValidateUpdate:
    def test_valid_causal_type(self):
        s = _make_state()
        ok, _ = validate_update("causal_type", "counterfactual", ["metadata"], "metadata", s)
        assert ok is True

    def test_invalid_causal_type(self):
        s = _make_state()
        ok, _ = validate_update("causal_type", "invalid_value", ["metadata"], "metadata", s)
        assert ok is False


class TestValidateDelete:
    def test_protected_key(self):
        s = _make_state()
        ok, _ = validate_delete("metadata", {}, [], s)
        assert ok is False

    def test_normal_key(self):
        s = _make_state()
        ok, _ = validate_delete("switch_1", ["val_a", "val_b"], ["switches"], s)
        assert ok is True


class TestPopulateVariables:
    def test_populates_both_versions(self):
        s = _make_state()
        assert "name_1" in s.context_variables["A"]
        assert "name_1" in s.context_variables["B"]
        assert "switch_1" in s.switches["A"]
        assert "item_1" in s.items["A"]
        assert "activity_1" in s.activities["B"]


class TestReindexSwitches:
    def test_reindex(self):
        switches = {"switch_1": ["a"], "switch_2": ["b"], "switch_3": ["c"]}
        reindex_switches_after_delete(switches, "switch_1")
        assert list(switches.keys()) == ["switch_1", "switch_2"]
        assert switches["switch_1"] == ["b"]
        assert switches["switch_2"] == ["c"]


class TestHandleAddSpecialCases:
    def test_add_option_mirrors_answers(self):
        qa_data = {
            "question": {"options": ["A", "B"], "prompt": "Q?"},
            "answers": {"0": [1, 0]},
            "links": {"0": [0]},
        }
        s = EditorState()
        s.json_data = qa_data
        s.current_file = "/tmp/test_qa_template.json"
        ok, err = handle_add_special_cases(qa_data, ["question"], "options", "2", "C", s)
        assert ok is True
        assert qa_data["answers"]["0"] == [1, 0, 0]

    def test_switch_value_must_append_at_end(self):
        s = _make_state()
        ok, err = handle_add_special_cases(
            s.json_data, ["A", "switches"], "switch_1", "0", "new_val", s
        )
        assert ok is False
        assert "middle" in err


class TestHandleDeleteSpecialCases:
    def test_delete_link_also_deletes_answer(self):
        qa_data = {
            "links": {"0": [0, 1], "1": [1, 0]},
            "answers": {"0": [1, 0, 0, 0], "1": [0, 1, 0, 0]},
            "question": {"options": ["A", "B", "C", "D"], "prompt": "Q?"},
        }
        s = EditorState()
        s.json_data = qa_data
        s.current_file = "/tmp/test_qa_template.json"
        ok, _ = handle_delete_special_cases(qa_data, ["links"], "1", [1, 0], s)
        assert ok is True
        assert "1" not in qa_data["answers"]

    def test_cannot_delete_from_link_table_directly(self):
        qa_data = {"links": {"0": [0, 1]}, "answers": {"0": [1, 0, 0, 0]}}
        s = EditorState()
        s.json_data = qa_data
        s.current_file = "/tmp/test_qa_template.json"
        ok, err = handle_delete_special_cases(qa_data, ["links", "0"], "0", 0, s)
        assert ok is False
        assert "link table" in err.lower()
