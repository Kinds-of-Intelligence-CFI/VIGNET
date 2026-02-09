"""Converts between plain Python dicts and the NiceGUI tree node format,
and provides path-based accessors into nested dicts/lists."""

from __future__ import annotations

from typing import Any

SEPARATOR = "."


def json_to_tree_nodes(data: Any, key: str = "root", path: list[str] | None = None) -> dict:
    """Convert a JSON-like value into a single NiceGUI tree node dict.

    Each node has:
        id       -- dot-separated path (e.g. "root.A.switches.switch_1.0")
        label    -- the key name shown in the tree
        type_tag -- "object" / "array" for containers, or the stringified leaf value
        children -- list of child node dicts (empty for leaves)
    """
    if path is None:
        path = [key]

    node_id = SEPARATOR.join(path)

    if isinstance(data, dict):
        children = []
        for k, v in data.items():
            child_path = path + [k]
            children.append(json_to_tree_nodes(v, key=k, path=child_path))
        return {
            "id": node_id,
            "label": key,
            "type_tag": "object",
            "children": children,
        }
    elif isinstance(data, list):
        children = []
        for i, item in enumerate(data):
            child_path = path + [str(i)]
            children.append(json_to_tree_nodes(item, key=str(i), path=child_path))
        return {
            "id": node_id,
            "label": key,
            "type_tag": "array",
            "children": children,
        }
    else:
        return {
            "id": node_id,
            "label": key,
            "type_tag": str(data) if data is not None else "null",
            "children": [],
        }


def id_to_path(node_id: str) -> list[str]:
    """Split a node id back into a path list."""
    return node_id.split(SEPARATOR)


def get_at_path(data: Any, path: list[str]) -> Any:
    """Retrieve a value from a nested dict/list using a path.

    The first element of *path* is the root label (e.g. "root") and is skipped.
    Remaining elements are dict keys or list indices (as strings).
    """
    current = data
    for key in path[1:]:  # skip the root label
        if isinstance(current, list):
            current = current[int(key)]
        else:
            current = current[key]
    return current


def set_at_path(data: Any, path: list[str], value: Any) -> None:
    """Set a value inside a nested dict/list at the given path.

    Navigates to the parent then sets the final key/index.
    """
    parent = get_at_path(data, path[:-1])
    final_key = path[-1]
    if isinstance(parent, list):
        parent[int(final_key)] = value
    else:
        parent[final_key] = value


def delete_at_path(data: Any, path: list[str]) -> Any:
    """Delete a key/index inside a nested dict/list.  Returns the deleted value."""
    parent = get_at_path(data, path[:-1])
    final_key = path[-1]
    if isinstance(parent, list):
        return parent.pop(int(final_key))
    else:
        return parent.pop(final_key)


def rename_key_at_path(data: Any, path: list[str], new_key: str) -> None:
    """Rename a dict key in-place, preserving insertion order."""
    parent = get_at_path(data, path[:-1])
    old_key = path[-1]
    if not isinstance(parent, dict):
        raise TypeError("Can only rename keys in a dict")
    if old_key == new_key:
        return
    new_dict = {}
    for k, v in parent.items():
        if k == old_key:
            new_dict[new_key] = v
        else:
            new_dict[k] = v
    parent.clear()
    parent.update(new_dict)


def get_parent_keys_from_path(path: list[str]) -> list[str]:
    """Return the list of ancestor key names (excluding root and the node itself).

    For path ["root", "A", "switches", "switch_1", "0"] returns
    ["A", "switches", "switch_1"].
    """
    return list(path[1:-1])


def find_node_by_id(nodes: list[dict] | dict, target_id: str) -> dict | None:
    """Depth-first search through tree nodes to find one by id."""
    if isinstance(nodes, dict):
        nodes = [nodes]
    for node in nodes:
        if node["id"] == target_id:
            return node
        result = find_node_by_id(node.get("children", []), target_id)
        if result is not None:
            return result
    return None


def collect_leaf_values(data: Any) -> list:
    """Recursively collect all leaf (non-dict, non-list) values from nested data."""
    values = []
    if isinstance(data, dict):
        for v in data.values():
            values.extend(collect_leaf_values(v))
    elif isinstance(data, list):
        for item in data:
            values.extend(collect_leaf_values(item))
    else:
        values.append(data)
    return values
