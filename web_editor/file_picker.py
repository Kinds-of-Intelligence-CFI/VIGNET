"""Local filesystem file picker dialog for NiceGUI.

Shows a dialog that lets the user browse directories on the server machine
and select a file.  Supports filtering by file extension.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

from nicegui import ui


class LocalFilePicker:
    """A dialog-based local file picker.

    Usage::

        async def pick():
            path = await LocalFilePicker("~/Documents", allowed_extensions=[".json"])
            if path:
                ui.notify(f"Selected: {path}")

        ui.button("Open", on_click=pick)
    """

    def __init__(
        self,
        start_dir: str = ".",
        *,
        allowed_extensions: list[str] | None = None,
        title: str = "Select a file",
    ) -> None:
        self.start_dir = os.path.abspath(os.path.expanduser(start_dir))
        self.allowed_extensions = allowed_extensions
        self.title = title
        self._result: str | None = None
        self._dialog: ui.dialog | None = None

    async def pick(self) -> str | None:
        """Open the picker dialog and return the selected path (or None)."""
        self._result = None

        with ui.dialog() as self._dialog, ui.card().classes("w-96"):
            ui.label(self.title).classes("text-lg font-bold")
            self._path_label = ui.label(self.start_dir).classes("text-xs text-gray-500 break-all")
            self._file_list = ui.column().classes("w-full max-h-80 overflow-y-auto gap-0")
            with ui.row().classes("w-full justify-end gap-2 mt-2"):
                ui.button("Cancel", on_click=lambda: self._close(None)).props("flat")

        self._current_dir = self.start_dir
        self._refresh_list()
        self._dialog.open()
        result = await self._dialog
        return self._result

    def _refresh_list(self) -> None:
        self._path_label.set_text(self._current_dir)
        self._file_list.clear()

        try:
            entries = sorted(os.listdir(self._current_dir))
        except PermissionError:
            with self._file_list:
                ui.label("Permission denied").classes("text-red-500")
            return

        dirs = []
        files = []
        for entry in entries:
            full = os.path.join(self._current_dir, entry)
            if os.path.isdir(full):
                dirs.append(entry)
            elif os.path.isfile(full):
                if self.allowed_extensions is None:
                    files.append(entry)
                else:
                    ext = os.path.splitext(entry)[1].lower()
                    if ext in self.allowed_extensions:
                        files.append(entry)

        with self._file_list:
            # parent directory link
            if self._current_dir != "/":
                ui.button(
                    ".. (parent directory)",
                    on_click=lambda: self._navigate(os.path.dirname(self._current_dir)),
                ).props("flat dense no-caps").classes("w-full justify-start text-blue-600")

            for d in dirs:
                full_path = os.path.join(self._current_dir, d)
                ui.button(
                    f"[dir] {d}",
                    on_click=lambda fp=full_path: self._navigate(fp),
                ).props("flat dense no-caps").classes("w-full justify-start text-blue-600")

            for f in files:
                full_path = os.path.join(self._current_dir, f)
                ui.button(
                    f,
                    on_click=lambda fp=full_path: self._close(fp),
                ).props("flat dense no-caps").classes("w-full justify-start")

    def _navigate(self, path: str) -> None:
        self._current_dir = path
        self._file_list.clear()
        self._refresh_list()

    def _close(self, path: str | None) -> None:
        self._result = path
        if self._dialog is not None:
            self._dialog.submit(path)
