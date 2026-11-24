"""Brief drive of the Tk launcher: processing runs in a thread and the UI stays responsive."""

import threading
import time
import tkinter as tk

import pytest

from line_counter import gui
from line_counter.config import ProcessingConfig
from line_counter.counter import CountingLine


def test_processing_runs_off_the_ui_thread(monkeypatch):
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no display available")
    root.withdraw()
    seen = {}
    release = threading.Event()

    def fake_process(video, output, lines, config):
        seen["thread"] = threading.current_thread()
        seen["display"] = config.display
        release.wait(5)

    import line_counter.pipeline as pipeline
    monkeypatch.setattr(pipeline, "process_video", fake_process)
    monkeypatch.setattr(gui.filedialog, "askdirectory", lambda **k: "out")
    shown = []
    monkeypatch.setattr(gui.messagebox, "showinfo", lambda *a: shown.append(a))
    monkeypatch.setattr(gui.messagebox, "showerror", lambda *a: shown.append(a))
    app = gui.LauncherApp(root, ProcessingConfig(display=True))
    app.video_path, app.lines = "v.mp4", [CountingLine("Line_1", (0, 0), (5, 5))]
    app.process_video()
    root.update()  # the UI answers while the worker is still blocked
    assert app.worker.is_alive() and str(app.btn_process["state"]) == "disabled"
    release.set()
    deadline = time.time() + 5
    while not shown and time.time() < deadline:
        root.update()
        time.sleep(0.05)
    root.destroy()
    assert seen["thread"] is not threading.main_thread() and seen["display"] is False
    assert shown and shown[0][0] == "Finished"
