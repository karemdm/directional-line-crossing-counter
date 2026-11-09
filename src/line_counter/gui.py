"""Tkinter launcher: select video -> draw lines -> process and save."""

from __future__ import annotations

import logging
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from typing import List, Optional

from line_counter.config import ProcessingConfig
from line_counter.counter import CountingLine

logger = logging.getLogger(__name__)

VIDEO_TYPES = [("Video Files", "*.mp4 *.avi *.mov *.mkv")]


class LauncherApp:
    """Three-step launcher window.

    Args:
        root: Tk root window.
        config: Processing settings used when the video is processed.
    """

    def __init__(self, root: tk.Tk, config: ProcessingConfig) -> None:
        self.root = root
        self.config = config
        self.video_path = ""
        self.lines: List[CountingLine] = []

        root.title("Directional Line Crossing Counter")
        root.geometry("450x300")
        tk.Label(root, text="Directional Line Crossing Counter",
                 font=("Arial", 14, "bold")).pack(pady=10)
        tk.Button(root, text="1. Select Input Video", command=self.select_video,
                  width=32).pack(pady=5)
        self.lbl_video = tk.Label(root, text="No video selected", fg="gray")
        self.lbl_video.pack()
        self.btn_draw = tk.Button(root, text="2. Draw Lines on Frame", command=self.draw_lines,
                                  width=32, state=tk.DISABLED)
        self.btn_draw.pack(pady=5)
        self.btn_process = tk.Button(root, text="3. Process and Save Video",
                                     command=self.process_video, width=32, state=tk.DISABLED,
                                     bg="#4CAF50", fg="white")
        self.btn_process.pack(pady=15)

    def select_video(self) -> None:
        """Ask for an input video and reset any previously drawn lines."""
        path = filedialog.askopenfilename(title="Select Video File", filetypes=VIDEO_TYPES)
        if path:
            self.video_path = path
            self.lbl_video.config(text=os.path.basename(path), fg="black")
            self.btn_draw.config(state=tk.NORMAL)
            self.btn_process.config(state=tk.DISABLED)
            self.lines = []

    def draw_lines(self) -> None:
        """Open the OpenCV line editor on the first frame."""
        from line_counter.line_editor import draw_lines, read_first_frame

        frame = read_first_frame(self.video_path)
        if frame is None:
            messagebox.showerror("Error", "Could not read the first frame.")
            return
        draw_lines(frame, self.lines)
        if self.lines:
            self.btn_process.config(state=tk.NORMAL)
            messagebox.showinfo("Success", f"{len(self.lines)} line(s) defined successfully!")
        else:
            messagebox.showwarning("Warning", "No lines were drawn.")

    def process_video(self) -> None:
        """Ask for an output folder, then process and save the video."""
        from line_counter.pipeline import default_output_path, process_video

        output_dir = filedialog.askdirectory(title="Select Output Directory to Save Processed Video")
        if not output_dir:
            return
        output_path = default_output_path(self.video_path, output_dir)
        try:
            process_video(self.video_path, output_path, self.lines, self.config)
        except Exception as exc:  # surface any failure to the GUI user
            logger.exception("Processing failed")
            messagebox.showerror("Error", f"Processing failed:\n{exc}")
            return
        messagebox.showinfo("Finished", f"Video processing completed!\nSaved to: {output_path}")


def run_gui(config: Optional[ProcessingConfig] = None) -> None:
    """Start the Tkinter launcher and block until it is closed.

    Args:
        config: Processing settings; defaults to :class:`ProcessingConfig`.
    """
    root = tk.Tk()
    LauncherApp(root, config or ProcessingConfig())
    root.mainloop()
