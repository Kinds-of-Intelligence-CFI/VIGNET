"""Entry point for the INTUIT JSON template editor (NiceGUI web UI).

Run with:
    python modify_template.py

Opens a browser at http://localhost:8080
"""

from nicegui import ui
from web_editor.app import build_ui

DEFAULT_DEMAND_FILE = "./dataframes/demand_df.csv"
DEFAULT_ITEM_FILE = "./dataframes/item_df.csv"
DEFAULT_ACTIVITY_FILE = "./dataframes/activity_df.csv"
DEFAULT_VARIABLE_FILE = "./dataframes/variable_df.csv"
DEFAULT_WEIGHTS_FILE = "./dataframes/variable_weights_df.csv"


@ui.page("/")
def index():
    build_ui(
        DEFAULT_DEMAND_FILE,
        DEFAULT_ITEM_FILE,
        DEFAULT_ACTIVITY_FILE,
        DEFAULT_VARIABLE_FILE,
        DEFAULT_WEIGHTS_FILE,
    )


if __name__ in {"__main__", "__mp_main__"}:
    ui.run(title="INTUIT JSON Editor", port=8080)
