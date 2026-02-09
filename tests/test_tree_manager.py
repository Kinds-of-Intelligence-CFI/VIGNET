import pytest
from web_editor.tree_manager import (
    json_to_tree_nodes,
    id_to_path,
    get_at_path,
    set_at_path,
    delete_at_path,
    rename_key_at_path,
    get_parent_keys_from_path,
    find_node_by_id,
    collect_leaf_values,
)


SAMPLE_DATA = {
    "metadata": {"novelty": 1, "causal_type": "intervention"},
    "A": {
        "context": "some text",
        "switches": {
            "switch_1": ["val_a", "val_b"],
            "switch_2": ["x", "y", "z"],
        },
        "variables": ["name_1", "city"],
    },
}


class TestJsonToTreeNodes:
    def test_root_is_object(self):
        nodes = json_to_tree_nodes(SAMPLE_DATA)
        assert nodes["type_tag"] == "object"
        assert nodes["label"] == "root"
        assert nodes["id"] == "root"

    def test_children_count(self):
        nodes = json_to_tree_nodes(SAMPLE_DATA)
        assert len(nodes["children"]) == 2  # metadata, A

    def test_nested_path_ids(self):
        nodes = json_to_tree_nodes(SAMPLE_DATA)
        a_node = nodes["children"][1]
        assert a_node["id"] == "root.A"
        switches = [c for c in a_node["children"] if c["label"] == "switches"][0]
        assert switches["id"] == "root.A.switches"
        sw1 = switches["children"][0]
        assert sw1["id"] == "root.A.switches.switch_1"
        assert sw1["type_tag"] == "array"
        assert sw1["children"][0]["id"] == "root.A.switches.switch_1.0"
        assert sw1["children"][0]["type_tag"] == "val_a"

    def test_leaf_values(self):
        nodes = json_to_tree_nodes(SAMPLE_DATA)
        meta = nodes["children"][0]
        novelty = meta["children"][0]
        assert novelty["label"] == "novelty"
        assert novelty["type_tag"] == "1"
        assert novelty["children"] == []


class TestIdToPath:
    def test_simple(self):
        assert id_to_path("root.A.switches") == ["root", "A", "switches"]


class TestGetAtPath:
    def test_root(self):
        assert get_at_path(SAMPLE_DATA, ["root"]) is SAMPLE_DATA

    def test_nested(self):
        assert get_at_path(SAMPLE_DATA, ["root", "A", "context"]) == "some text"

    def test_list_index(self):
        assert get_at_path(SAMPLE_DATA, ["root", "A", "switches", "switch_1", "1"]) == "val_b"


class TestSetAtPath:
    def test_set_leaf(self):
        import copy
        d = copy.deepcopy(SAMPLE_DATA)
        set_at_path(d, ["root", "A", "context"], "new text")
        assert d["A"]["context"] == "new text"

    def test_set_list_item(self):
        import copy
        d = copy.deepcopy(SAMPLE_DATA)
        set_at_path(d, ["root", "A", "switches", "switch_1", "0"], "changed")
        assert d["A"]["switches"]["switch_1"][0] == "changed"


class TestDeleteAtPath:
    def test_delete_dict_key(self):
        import copy
        d = copy.deepcopy(SAMPLE_DATA)
        val = delete_at_path(d, ["root", "A", "switches", "switch_2"])
        assert val == ["x", "y", "z"]
        assert "switch_2" not in d["A"]["switches"]

    def test_delete_list_item(self):
        import copy
        d = copy.deepcopy(SAMPLE_DATA)
        val = delete_at_path(d, ["root", "A", "switches", "switch_1", "0"])
        assert val == "val_a"
        assert d["A"]["switches"]["switch_1"] == ["val_b"]


class TestRenameKeyAtPath:
    def test_rename(self):
        import copy
        d = copy.deepcopy(SAMPLE_DATA)
        rename_key_at_path(d, ["root", "A", "switches", "switch_1"], "switch_9")
        assert "switch_9" in d["A"]["switches"]
        assert "switch_1" not in d["A"]["switches"]
        # order preserved: switch_9 comes before switch_2
        assert list(d["A"]["switches"].keys()) == ["switch_9", "switch_2"]


class TestGetParentKeysFromPath:
    def test_basic(self):
        path = ["root", "A", "switches", "switch_1", "0"]
        assert get_parent_keys_from_path(path) == ["A", "switches", "switch_1"]

    def test_shallow(self):
        path = ["root", "A"]
        assert get_parent_keys_from_path(path) == []


class TestFindNodeById:
    def test_find_deep(self):
        nodes = json_to_tree_nodes(SAMPLE_DATA)
        found = find_node_by_id(nodes, "root.A.switches.switch_1.0")
        assert found is not None
        assert found["type_tag"] == "val_a"

    def test_not_found(self):
        nodes = json_to_tree_nodes(SAMPLE_DATA)
        assert find_node_by_id(nodes, "root.Z.nonexistent") is None


class TestCollectLeafValues:
    def test_flat(self):
        assert set(collect_leaf_values({"a": 1, "b": "x"})) == {1, "x"}

    def test_nested(self):
        vals = collect_leaf_values(SAMPLE_DATA)
        assert "some text" in vals
        assert 1 in vals
        assert "val_a" in vals
