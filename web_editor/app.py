"""NiceGUI application: JSON template editor.

This replaces the tkinter-based JSONEditorGUI.  The JSON dict is the single
source of truth; the tree display is rebuilt from it after every mutation.
"""

from __future__ import annotations

import json
import os
import traceback

import pandas as pd
from nicegui import ui

from project_scripts.compiler import Compiler
from web_editor.editor_logic import (
    append_new_switch_to_qa_files,
    create_context_template,
    create_qa_template,
    get_base_context_number,
    get_context_template_dict,
    get_next_qa_id,
    handle_add_special_cases,
    handle_delete_special_cases,
    list_scenario_dirs,
    load_activities_data,
    load_demand_data,
    load_items_data,
    load_variable_data,
    load_weights_data,
    populate_variables,
    validate_delete,
    validate_qa_id,
    validate_scenario_name,
    validate_update,
    value_from_string,
    type_of_value,
    _has_context_template,
)
from web_editor.file_picker import LocalFilePicker
from web_editor.state import (
    CONTEXT_TEMPLATE,
    NEVER_DELETE_KEYS,
    POTENTIAL_VERSIONS,
    QA_TEMPLATE,
    EditorState,
)
from web_editor.tree_manager import (
    collect_leaf_values,
    delete_at_path,
    find_node_by_id,
    get_at_path,
    get_parent_keys_from_path,
    id_to_path,
    json_to_tree_nodes,
    set_at_path,
)


def build_ui(
    default_demand_file: str,
    default_item_file: str,
    default_activity_file: str,
    default_variable_file: str,
    default_weights_file: str,
) -> None:
    """Construct the full editor page.  Call once at import time."""

    state = EditorState()

    # load default CSV files
    load_demand_data(default_demand_file, state)
    load_items_data(default_item_file, state)
    load_activities_data(default_activity_file, state)
    load_variable_data(default_variable_file, state)
    load_weights_data(default_weights_file, state)

    # --- currently selected tree node ---
    selected_node_id: dict = {"value": None}

    # ------------------------------------------------------------------
    # Helper: rebuild tree from json_data
    # ------------------------------------------------------------------

    def rebuild_tree(restore_id: str | None = None) -> None:
        """Regenerate tree nodes from state.json_data and refresh the widget.

        Sets tree_widget._props['nodes'] directly because NiceGUI wraps
        assigned lists in an ObservableList (a copy), so mutating a plain
        Python list that was passed to ui.tree() has no effect on the widget.
        """
        if state.json_data is not None:
            root = json_to_tree_nodes(state.json_data)
            tree_widget._props['nodes'] = [root]
        else:
            tree_widget._props['nodes'] = []
        tree_widget.update()
        if restore_id is not None:
            tree_widget._props["selected"] = restore_id
            selected_node_id["value"] = restore_id
            _populate_editor_from_id(restore_id)
        tree_widget.update()

    def _populate_editor_from_id(node_id: str) -> None:
        path = id_to_path(node_id)
        try:
            value = get_at_path(state.json_data, path)
        except (KeyError, IndexError, TypeError):
            return
        key_input.set_value(path[-1])
        t = type_of_value(value)
        if t in ("object", "array"):
            value_textarea.set_value("")
            type_select.set_value(t)
        else:
            value_textarea.set_value(str(value) if value is not None else "null")
            type_select.set_value(t)

    # ------------------------------------------------------------------
    # Tree selection handler
    # ------------------------------------------------------------------

    def on_tree_select(e) -> None:
        node_id = e.value
        if not node_id:
            return
        selected_node_id["value"] = node_id
        _populate_editor_from_id(node_id)

    # ------------------------------------------------------------------
    # File operations
    # ------------------------------------------------------------------

    async def open_json() -> None:
        picker = LocalFilePicker(
            start_dir=os.path.join(os.getcwd(), "dataset"),
            allowed_extensions=[".json"],
            title="Open JSON template",
        )
        path = await picker.pick()
        if not path:
            return
        try:
            with open(path, "r") as f:
                state.json_data = json.load(f)
            state.current_file = path
            populate_variables(state)
            rebuild_tree()
            status_label.set_text(f"Loaded: {os.path.basename(path)}")
        except json.JSONDecodeError:
            ui.notify("Invalid JSON file", type="negative")
        except Exception:
            ui.notify(f"Error loading file:\n{traceback.format_exc()}", type="negative")

    def save_json() -> None:
        if state.json_data is None:
            ui.notify("Nothing to save", type="warning")
            return
        if state.current_file is None:
            ui.notify("No file path set -- open a file first", type="warning")
            return
        try:
            with open(state.current_file, "w") as f:
                json.dump(state.json_data, f, indent=4)
            status_label.set_text(f"Saved: {os.path.basename(state.current_file)}")
        except Exception:
            ui.notify(f"Error saving file:\n{traceback.format_exc()}", type="negative")

    async def _open_csv(loader, label: str) -> None:
        picker = LocalFilePicker(
            start_dir=os.path.join(os.getcwd(), "dataframes"),
            allowed_extensions=[".csv"],
            title=f"Open {label} file",
        )
        path = await picker.pick()
        if not path:
            return
        err = loader(path, state)
        if err:
            ui.notify(err, type="negative")
        else:
            status_label.set_text(f"{label} loaded: {os.path.basename(path)}")

    # ------------------------------------------------------------------
    # Template / folder creation dialogs
    # ------------------------------------------------------------------

    dataset_dir = os.path.join(os.getcwd(), "dataset")

    async def new_scenario_folder() -> None:
        with ui.dialog() as dialog, ui.card().classes("w-96"):
            ui.label("New Scenario Folder").classes("text-lg font-bold")
            name_input = ui.input(label="Folder name").classes("w-full")
            error_label = ui.label("").classes("text-red-500 text-sm")

            with ui.row().classes("w-full justify-end gap-2"):
                ui.button("Cancel", on_click=dialog.close)

                def on_ok():
                    name = name_input.value.strip()
                    ok, reason = validate_scenario_name(name, dataset_dir)
                    if not ok:
                        error_label.set_text(reason)
                        return
                    os.makedirs(os.path.join(dataset_dir, name))
                    ui.notify(f"Created folder: {name}", type="positive")
                    status_label.set_text(f"Created scenario folder: {name}")
                    dialog.close()

                ui.button("OK", on_click=on_ok).props("color=primary")
        dialog.open()

    async def new_context_template() -> None:
        dirs = list_scenario_dirs(dataset_dir)
        available = [
            d for d in dirs
            if not _has_context_template(os.path.join(dataset_dir, d))
        ]
        if not available:
            ui.notify(
                "No scenario folders available (all already have a context template, or none exist).",
                type="warning",
            )
            return

        with ui.dialog() as dialog, ui.card().classes("w-96"):
            ui.label("New Context Template").classes("text-lg font-bold")
            folder_select = ui.select(
                options=available, label="Scenario folder"
            ).classes("w-full")
            error_label = ui.label("").classes("text-red-500 text-sm")

            with ui.row().classes("w-full justify-end gap-2"):
                ui.button("Cancel", on_click=dialog.close)

                def on_ok():
                    scenario_name = folder_select.value
                    if not scenario_name:
                        error_label.set_text("Select a folder.")
                        return
                    scenario_dir = os.path.join(dataset_dir, scenario_name)
                    file_path, err = create_context_template(scenario_dir, scenario_name)
                    if err:
                        error_label.set_text(err)
                        return
                    # auto-load into editor
                    with open(file_path, "r") as f:
                        state.json_data = json.load(f)
                    state.current_file = file_path
                    populate_variables(state)
                    rebuild_tree()
                    ui.notify(f"Created: {os.path.basename(file_path)}", type="positive")
                    status_label.set_text(f"Created: {os.path.basename(file_path)}")
                    dialog.close()

                ui.button("OK", on_click=on_ok).props("color=primary")
        dialog.open()

    async def new_qa_template() -> None:
        dirs = list_scenario_dirs(dataset_dir)
        available = [
            d for d in dirs
            if _has_context_template(os.path.join(dataset_dir, d))
        ]
        if not available:
            ui.notify(
                "No scenario folders with a context template found.",
                type="warning",
            )
            return

        with ui.dialog() as dialog, ui.card().classes("w-96"):
            ui.label("New QA Template").classes("text-lg font-bold")
            folder_select = ui.select(
                options=available, label="Scenario folder"
            ).classes("w-full")
            version_sel = ui.select(
                options=POTENTIAL_VERSIONS, label="Version", value="A"
            ).classes("w-full")
            id_input = ui.input(label="QA ID").classes("w-full")
            error_label = ui.label("").classes("text-red-500 text-sm")

            def refresh_suggestion():
                scenario_name = folder_select.value
                ver = version_sel.value
                if scenario_name and ver:
                    scenario_dir = os.path.join(dataset_dir, scenario_name)
                    suggestion = get_next_qa_id(scenario_dir, dataset_dir, ver)
                    id_input.set_value(suggestion)

            folder_select.on_value_change(lambda _: refresh_suggestion())
            version_sel.on_value_change(lambda _: refresh_suggestion())

            with ui.row().classes("w-full justify-end gap-2"):
                ui.button("Cancel", on_click=dialog.close)

                def on_ok():
                    scenario_name = folder_select.value
                    ver = version_sel.value
                    qa_id = id_input.value.strip()
                    if not scenario_name:
                        error_label.set_text("Select a folder.")
                        return
                    if not ver:
                        error_label.set_text("Select a version.")
                        return
                    scenario_dir = os.path.join(dataset_dir, scenario_name)
                    ok, reason = validate_qa_id(qa_id, scenario_dir)
                    if not ok:
                        error_label.set_text(reason)
                        return
                    bcn = get_base_context_number(scenario_dir, dataset_dir)
                    file_path, err = create_qa_template(
                        scenario_dir, scenario_name, ver, qa_id, bcn
                    )
                    if err:
                        error_label.set_text(err)
                        return
                    # auto-load into editor
                    with open(file_path, "r") as f:
                        state.json_data = json.load(f)
                    state.current_file = file_path
                    populate_variables(state)
                    rebuild_tree()
                    ui.notify(f"Created: {os.path.basename(file_path)}", type="positive")
                    status_label.set_text(f"Created: {os.path.basename(file_path)}")
                    dialog.close()

                ui.button("OK", on_click=on_ok).props("color=primary")
        dialog.open()

    # ------------------------------------------------------------------
    # Editor actions
    # ------------------------------------------------------------------

    def _get_parent_context() -> tuple[list[str], str, list[str]] | None:
        """Return (parent_keys, parent_key, full_path) for the selected node's parent.

        Returns None and shows a notification if nothing is selected.
        """
        nid = selected_node_id["value"]
        if not nid:
            ui.notify("No node selected", type="warning")
            return None
        path = id_to_path(nid)
        parent_keys = get_parent_keys_from_path(path)
        parent_key = path[-2] if len(path) >= 2 else "root"
        return parent_keys, parent_key, path

    def update_item() -> None:
        if state.json_data is None:
            return
        ctx = _get_parent_context()
        if ctx is None:
            return
        parent_keys, parent_key, path = ctx
        new_key = key_input.value
        new_value = value_from_string(value_textarea.value, type_select.value)

        ok, reason = validate_update(new_key, new_value, parent_keys, parent_key, state)
        if not ok:
            ui.notify(f"Error updating ({new_key}, {new_value}):\n{reason}", type="negative")
            return

        old_key = path[-1]
        if isinstance(new_value, (dict, list)):
            old_value = get_at_path(state.json_data, path)
            if not isinstance(old_value, type(new_value)):
                set_at_path(state.json_data, path, new_value)
        else:
            set_at_path(state.json_data, path, new_value)

        # handle key rename
        if new_key != old_key:
            parent = get_at_path(state.json_data, path[:-1])
            if isinstance(parent, dict):
                new_dict = {}
                for k, v in parent.items():
                    new_dict[new_key if k == old_key else k] = v
                parent.clear()
                parent.update(new_dict)
                # update path for tree restore
                path[-1] = new_key

        new_id = ".".join(path)
        rebuild_tree(restore_id=new_id)
        status_label.set_text("Item updated")

    def add_item() -> None:
        if state.json_data is None:
            return
        nid = selected_node_id["value"]
        if not nid:
            ui.notify("Select a container node (object/array) first", type="warning")
            return
        path = id_to_path(nid)

        # the selected node is the parent to add into
        try:
            parent_val = get_at_path(state.json_data, path)
        except (KeyError, IndexError, TypeError):
            ui.notify("Cannot resolve selected node", type="negative")
            return
        if not isinstance(parent_val, (dict, list)):
            ui.notify("Can only add items to objects or arrays", type="negative")
            return

        parent_keys = get_parent_keys_from_path(path)
        parent_key = path[-1]

        new_key = key_input.value
        new_value = value_from_string(value_textarea.value, type_select.value)

        ok, reason = validate_update(new_key, new_value, parent_keys, parent_key, state)
        if not ok:
            ui.notify(f"Error adding ({new_key}, {new_value}):\n{reason}", type="negative")
            return

        # special-case cascading logic
        ok, err = handle_add_special_cases(
            state.json_data, parent_keys, parent_key, new_key, new_value, state
        )
        if not ok:
            ui.notify(err, type="negative")
            return

        # perform the actual add
        if isinstance(parent_val, dict):
            parent_val[new_key] = new_value
            # for new switches auto-insert an empty first value
            if parent_key == "switches" and isinstance(new_value, list):
                parent_val[new_key] = [""]
            # for new links auto-fill with zeros matching existing link length
            if parent_key == "links" and isinstance(new_value, list):
                existing = state.json_data.get("links", {})
                first_link = next(iter(existing.values()), [])
                parent_val[new_key] = [0] * len(first_link) if first_link else [0]
        elif isinstance(parent_val, list):
            parent_val.append(new_value)

        populate_variables(state)
        new_child_id = nid + "." + new_key
        rebuild_tree(restore_id=new_child_id)
        status_label.set_text("Item added")

    def delete_item() -> None:
        if state.json_data is None:
            return
        nid = selected_node_id["value"]
        if not nid:
            ui.notify("No node selected", type="warning")
            return
        path = id_to_path(nid)
        if len(path) <= 1:
            ui.notify("Cannot delete the root node", type="negative")
            return

        del_key = path[-1]
        try:
            del_value = get_at_path(state.json_data, path)
        except (KeyError, IndexError, TypeError):
            ui.notify("Cannot resolve selected node", type="negative")
            return

        parent_keys = get_parent_keys_from_path(path)

        # basic disallowed-key check
        ok, reason = validate_delete(del_key, del_value, parent_keys, state)
        if not ok:
            ui.notify(reason, type="negative")
            return

        # cascading side-effects
        ok, err = handle_delete_special_cases(
            state.json_data, parent_keys, del_key, del_value, state
        )
        if not ok:
            ui.notify(err, type="negative")
            return

        # perform the deletion
        delete_at_path(state.json_data, path)

        # if we deleted from an array, the indices shift -- no need to reindex
        # because json_to_tree_nodes enumerates freshly

        populate_variables(state)
        # select the parent after deletion
        parent_id = ".".join(path[:-1])
        rebuild_tree(restore_id=parent_id)
        status_label.set_text("Item deleted")

    # ------------------------------------------------------------------
    # Sample generation
    # ------------------------------------------------------------------

    @ui.refreshable
    def switch_selectors() -> None:
        """Build one selector per switch for the currently chosen version."""
        ver = version_select.value
        if not ver or state.json_data is None:
            return
        if state.is_context_template():
            context_data = state.json_data
        else:
            context_data = get_context_template_dict(state)
        if context_data is None or ver not in context_data:
            return
        switches = context_data[ver].get("switches", {})
        switch_selects.clear()
        for sw_name, sw_values in switches.items():
            with ui.column().classes("gap-0"):
                ui.label(sw_name).classes("text-xs")
                sel = ui.select(
                    options={i: str(v) for i, v in enumerate(sw_values)},
                    label=sw_name,
                ).classes("w-36")
                switch_selects[sw_name] = sel

    switch_selects: dict[str, ui.select] = {}

    def on_version_change(e) -> None:
        switch_selectors.refresh()

    def generate_sample() -> None:
        if state.json_data is None:
            ui.notify("Load a template first", type="warning")
            return
        ver = version_select.value
        inf = inference_select.value
        tp = third_person_select.value
        if not ver or inf is None or tp is None:
            ui.notify("Select version, inference level, and third person first", type="warning")
            return

        link_table = []
        for sw_name in sorted(switch_selects.keys()):
            sel = switch_selects[sw_name]
            link_table.append(sel.value if sel.value is not None else 0)

        if state.is_context_template():
            context_tree = state.json_data
        else:
            context_tree = get_context_template_dict(state)
        if context_tree is None:
            ui.notify("Cannot load context template", type="negative")
            return

        id_string = "EXAMPLE"
        condition = "0"
        difficulty = {
            "third_person": tp,
            "inference_level": int(inf),
            "double_spaces": True,
            "character_noise": (0, 0.1),
            "capitalisation": (0, 0.8),
        }

        qa_template_dict = {
            "id": id_string,
            "name": id_string,
            "version": ver,
            "number": 1,
            "metadata": context_tree["metadata"],
            "context": context_tree[ver]["context"],
            "filler": context_tree[ver]["filler"],
            "variables": context_tree[ver]["variables"],
            "items": context_tree[ver]["items"],
            "activities": context_tree[ver]["activities"],
            "coinflips": context_tree[ver]["coinflips"],
            "switches": context_tree[ver]["switches"],
            "capability_type": "comprehension_check",
            "demands": {"c0": [], "c1": [], "c2": []},
            "links": {"0": link_table},
            "question": {"prompt": "QUESTION HERE", "options": ["Answer 1", "Answer 2", "Answer 3", "Answer 4"]},
            "answers": {"0": [1, 0, 0, 0]},
        }

        template_df = pd.DataFrame(
            [qa_template_dict],
            columns=[
                "id", "number", "name", "version", "capability_type", "metadata",
                "context", "filler", "variables", "items", "activities", "coinflips",
                "switches", "demands", "links", "question", "answers",
            ],
        )

        try:
            variable_df = pd.read_csv(state.current_variable_file)
            weights_df = pd.read_csv(state.current_weights_file)
            item_df = pd.read_csv(state.current_item_file)
            activity_df = pd.read_csv(state.current_activity_file)
        except Exception as exc:
            ui.notify(f"Error loading CSV data:\n{exc}", type="negative")
            return

        try:
            vignette = Compiler(
                id_string, condition, difficulty,
                template_df, variable_df, weights_df,
                item_df, activity_df,
            )
            mini_battery = vignette.compile_battery(1)
            sample = mini_battery["vignette"].iloc[0]
            sample_display.set_value(sample)
        except Exception as exc:
            ui.notify(f"Error generating sample:\n{traceback.format_exc()}", type="negative")

    # ------------------------------------------------------------------
    # Build the UI layout
    # ------------------------------------------------------------------

    ui.page_title("INTUIT JSON Editor")

    # --- Top button bar ---
    with ui.row().classes("w-full items-center gap-2 p-2"):
        ui.button("Open JSON", on_click=open_json)
        with ui.dropdown_button("New...", auto_close=True):
            ui.item("New Scenario Folder", on_click=new_scenario_folder)
            ui.item("New Context Template", on_click=new_context_template)
            ui.item("New QA Template", on_click=new_qa_template)
        ui.button("Save", on_click=save_json)
        ui.button("Demand CSV", on_click=lambda: _open_csv(load_demand_data, "Demand"))
        ui.button("Item CSV", on_click=lambda: _open_csv(load_items_data, "Item"))
        ui.button("Activity CSV", on_click=lambda: _open_csv(load_activities_data, "Activity"))
        ui.button("Variable CSV", on_click=lambda: _open_csv(load_variable_data, "Variable"))
        ui.button("Weights CSV", on_click=lambda: _open_csv(load_weights_data, "Weights"))

    # --- Three column layout ---
    with ui.row().classes("w-full flex-nowrap gap-0").style("height: calc(100vh - 120px)"):

        # Column 1: Tree view
        with ui.column().classes("w-1/3 h-full border-r p-2 overflow-auto"):
            ui.label("JSON Tree").classes("text-lg font-bold")
            tree_widget = ui.tree(
                [],
                node_key="id",
                label_key="label",
                children_key="children",
                on_select=on_tree_select,
            ).classes("w-full")

            # custom body slot to show type/value beside the label
            tree_widget.add_slot(
                "default-header",
                '''
                <span>
                    <strong>{{ props.node.label }}</strong>
                    <span v-if="props.node.type_tag !== 'object' && props.node.type_tag !== 'array'"
                          style="color: #666; margin-left: 8px;">
                        = {{ props.node.type_tag }}
                    </span>
                    <span v-else style="color: #999; margin-left: 8px; font-size: 0.85em;">
                        ({{ props.node.type_tag }})
                    </span>
                </span>
                ''',
            )

        # Column 2: Editor panel
        with ui.column().classes("w-1/3 h-full border-r p-4 overflow-auto gap-3"):
            ui.label("Edit Selected Item").classes("text-lg font-bold")
            key_input = ui.input(label="Key").classes("w-full")
            value_textarea = ui.textarea(label="Value").classes("w-full").props("rows=8")
            type_select = ui.select(
                options=["string", "number", "boolean", "null", "object", "array"],
                label="Type",
                value="string",
            ).classes("w-full")
            with ui.row().classes("gap-2"):
                ui.button("Add Item", on_click=add_item)
                ui.button("Update", on_click=update_item)
                ui.button("Delete", on_click=delete_item).props("color=negative")

        # Column 3: Sample generator
        with ui.column().classes("w-1/3 h-full p-4 overflow-auto gap-3"):
            ui.label("Generate Sample").classes("text-lg font-bold")
            sample_display = ui.textarea(label="Sample output").classes("w-full").props("rows=12 readonly")

            with ui.row().classes("gap-2 flex-wrap"):
                version_select = ui.select(
                    options=POTENTIAL_VERSIONS,
                    label="Version",
                    on_change=on_version_change,
                ).classes("w-24")
                inference_select = ui.select(
                    options=[0, 1, 2, 3],
                    label="Inference Level",
                ).classes("w-32")
                third_person_select = ui.select(
                    options={True: "True", False: "False"},
                    label="Third Person",
                ).classes("w-28")

            switch_selectors()

            ui.button("Generate", on_click=generate_sample)

    # --- Status bar ---
    status_label = ui.label("Ready").classes("p-2 text-sm text-gray-600")
