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
    validate_scenario_name,
    list_scenario_dirs,
    create_context_template,
    validate_qa_id,
    get_base_context_number,
    get_next_qa_id,
    create_qa_template,
    SKELETON_CONTEXT_TEMPLATE,
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


# ===================================================================
# Tests for template / folder creation functions
# ===================================================================

class TestValidateScenarioName:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()

    def test_valid_name(self):
        ok, _ = validate_scenario_name("my_scenario", self.tmpdir)
        assert ok is True

    def test_uppercase_allowed(self):
        ok, _ = validate_scenario_name("just_what_I_wanted", self.tmpdir)
        assert ok is True

    def test_empty_rejected(self):
        ok, reason = validate_scenario_name("", self.tmpdir)
        assert ok is False
        assert "empty" in reason.lower()

    def test_spaces_rejected(self):
        ok, _ = validate_scenario_name("bad name", self.tmpdir)
        assert ok is False

    def test_special_chars_rejected(self):
        ok, _ = validate_scenario_name("bad-name!", self.tmpdir)
        assert ok is False

    def test_leading_underscore_rejected(self):
        ok, _ = validate_scenario_name("_leading", self.tmpdir)
        assert ok is False

    def test_trailing_underscore_rejected(self):
        ok, _ = validate_scenario_name("trailing_", self.tmpdir)
        assert ok is False

    def test_double_underscore_rejected(self):
        ok, _ = validate_scenario_name("bad__name", self.tmpdir)
        assert ok is False

    def test_duplicate_rejected(self):
        os.makedirs(os.path.join(self.tmpdir, "existing"))
        ok, reason = validate_scenario_name("existing", self.tmpdir)
        assert ok is False
        assert "already exists" in reason


class TestListScenarioDirs:
    def test_empty_dir(self):
        tmpdir = tempfile.mkdtemp()
        assert list_scenario_dirs(tmpdir) == []

    def test_lists_dirs_not_files(self):
        tmpdir = tempfile.mkdtemp()
        os.makedirs(os.path.join(tmpdir, "beta"))
        os.makedirs(os.path.join(tmpdir, "alpha"))
        with open(os.path.join(tmpdir, "file.txt"), "w") as f:
            f.write("")
        result = list_scenario_dirs(tmpdir)
        assert result == ["alpha", "beta"]

    def test_nonexistent_dir(self):
        assert list_scenario_dirs("/tmp/nonexistent_xyz_123") == []


class TestCreateContextTemplate:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.scenario = os.path.join(self.tmpdir, "test_scenario")
        os.makedirs(self.scenario)

    def test_creates_file(self):
        path, err = create_context_template(self.scenario, "test_scenario")
        assert err is None
        assert path is not None
        assert os.path.exists(path)
        assert path.endswith("test_scenario_context_template.json")

    def test_correct_structure(self):
        path, _ = create_context_template(self.scenario, "test_scenario")
        with open(path) as f:
            data = json.load(f)
        assert "metadata" in data
        assert "A" in data
        assert "B" in data
        assert data["A"]["variables"] == []
        assert data["A"]["switches"] == {}
        assert list(data["A"]["filler"].keys()) == ["0", "1", "2", "3"]

    def test_refuses_duplicate(self):
        create_context_template(self.scenario, "test_scenario")
        _, err = create_context_template(self.scenario, "test_scenario")
        assert err is not None
        assert "already exists" in err

    def test_nonexistent_dir(self):
        _, err = create_context_template("/tmp/no_such_dir_xyz", "foo")
        assert err is not None


class TestValidateQaId:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()

    def test_valid_id(self):
        ok, _ = validate_qa_id("3.0.0.0.a", self.tmpdir)
        assert ok is True

    def test_missing_version_letter(self):
        ok, _ = validate_qa_id("3.0.0.0", self.tmpdir)
        assert ok is False

    def test_uppercase_version_rejected(self):
        ok, _ = validate_qa_id("3.0.0.0.A", self.tmpdir)
        assert ok is False

    def test_invalid_format(self):
        ok, _ = validate_qa_id("bad_id", self.tmpdir)
        assert ok is False

    def test_duplicate_file_rejected(self):
        with open(os.path.join(self.tmpdir, "3.0.0.0.a_qa_template.json"), "w") as f:
            json.dump({}, f)
        ok, reason = validate_qa_id("3.0.0.0.a", self.tmpdir)
        assert ok is False
        assert "already exists" in reason


class TestGetBaseContextNumber:
    def setup_method(self):
        self.dataset = tempfile.mkdtemp()

    def _make_scenario(self, name, number):
        d = os.path.join(self.dataset, name)
        os.makedirs(d, exist_ok=True)
        qa = {"base_context_number": number, "id": f"{number}.0.0.0.a"}
        with open(os.path.join(d, f"{number}.0.0.0.a_qa_template.json"), "w") as f:
            json.dump(qa, f)
        return d

    def test_no_existing_files(self):
        new_dir = os.path.join(self.dataset, "new_scenario")
        os.makedirs(new_dir)
        num = get_base_context_number(new_dir, self.dataset)
        assert num == 1

    def test_reuses_existing_number(self):
        d = self._make_scenario("existing", 5)
        num = get_base_context_number(d, self.dataset)
        assert num == 5

    def test_skips_used_numbers(self):
        self._make_scenario("s1", 1)
        self._make_scenario("s2", 2)
        new_dir = os.path.join(self.dataset, "s3")
        os.makedirs(new_dir)
        num = get_base_context_number(new_dir, self.dataset)
        assert num == 3

    def test_fills_gap(self):
        self._make_scenario("s1", 1)
        self._make_scenario("s3", 3)
        new_dir = os.path.join(self.dataset, "new")
        os.makedirs(new_dir)
        num = get_base_context_number(new_dir, self.dataset)
        assert num == 2


class TestGetNextQaId:
    def setup_method(self):
        self.dataset = tempfile.mkdtemp()
        self.scenario = os.path.join(self.dataset, "test")
        os.makedirs(self.scenario)

    def test_no_existing_files(self):
        result = get_next_qa_id(self.scenario, self.dataset, "A")
        assert result == "1.0.0.0.a"

    def test_increments_second_digit(self):
        qa = {"base_context_number": 5, "id": "5.0.0.0.a"}
        with open(os.path.join(self.scenario, "5.0.0.0.a_qa_template.json"), "w") as f:
            json.dump(qa, f)
        result = get_next_qa_id(self.scenario, self.dataset, "A")
        assert result == "5.1.0.0.a"

    def test_version_b(self):
        result = get_next_qa_id(self.scenario, self.dataset, "B")
        assert result == "1.0.0.0.b"


class TestCreateQaTemplate:
    def setup_method(self):
        self.dataset = tempfile.mkdtemp()
        self.scenario = os.path.join(self.dataset, "test_scenario")
        os.makedirs(self.scenario)
        # write a context template with 2 switches
        ctx = copy.deepcopy(SKELETON_CONTEXT_TEMPLATE)
        ctx["A"]["switches"] = {"switch_1": ["a", "b"], "switch_2": ["x", "y"]}
        with open(os.path.join(self.scenario, "test_scenario_context_template.json"), "w") as f:
            json.dump(ctx, f)

    def test_creates_file(self):
        path, err = create_qa_template(
            self.scenario, "test_scenario", "A", "1.0.0.0.a", 1
        )
        assert err is None
        assert path is not None
        assert os.path.exists(path)

    def test_links_match_switch_count(self):
        path, _ = create_qa_template(
            self.scenario, "test_scenario", "A", "1.0.0.0.a", 1
        )
        with open(path) as f:
            data = json.load(f)
        assert len(data["links"]["0"]) == 2

    def test_correct_base_fields(self):
        path, _ = create_qa_template(
            self.scenario, "test_scenario", "B", "1.0.0.0.b", 7
        )
        with open(path) as f:
            data = json.load(f)
        assert data["base_context"] == "test_scenario"
        assert data["base_context_version"] == "B"
        assert data["base_context_number"] == 7
        assert data["id"] == "1.0.0.0.b"

    def test_refuses_without_context_template(self):
        empty_dir = os.path.join(self.dataset, "empty")
        os.makedirs(empty_dir)
        _, err = create_qa_template(empty_dir, "empty", "A", "1.0.0.0.a", 1)
        assert err is not None
        assert "context template" in err.lower()

    def test_refuses_duplicate(self):
        create_qa_template(self.scenario, "test_scenario", "A", "1.0.0.0.a", 1)
        _, err = create_qa_template(self.scenario, "test_scenario", "A", "1.0.0.0.a", 1)
        assert err is not None
        assert "already exists" in err

    def test_zero_switches(self):
        # make a scenario with no switches
        empty_ctx_dir = os.path.join(self.dataset, "no_switches")
        os.makedirs(empty_ctx_dir)
        ctx = copy.deepcopy(SKELETON_CONTEXT_TEMPLATE)
        with open(os.path.join(empty_ctx_dir, "no_switches_context_template.json"), "w") as f:
            json.dump(ctx, f)
        path, err = create_qa_template(
            empty_ctx_dir, "no_switches", "A", "1.0.0.0.a", 1
        )
        assert err is None
        with open(path) as f:
            data = json.load(f)
        assert data["links"]["0"] == []
