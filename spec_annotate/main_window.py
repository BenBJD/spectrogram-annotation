from PySide6.QtCore import Qt, QObject, Signal, Slot, QThread, QSettings, QByteArray, QBuffer, QSize, QTimer
from PySide6.QtGui import QAction, QIcon
from PySide6.QtWidgets import (
    QMainWindow,
    QFileDialog,
    QMessageBox,
    QProgressDialog,
    QToolBar,
    QSlider,
    QLabel,
    QStatusBar,
    QWidget,
    QHBoxLayout,
    QStyle,
    QSizePolicy,
)
from PySide6.QtMultimedia import (
    QMediaPlayer,
    QAudioOutput,
    QAudioSink,
    QAudioFormat,
    QMediaDevices,
)
from PySide6.QtCore import QUrl

from pathlib import Path
import numpy as np
import librosa

from spec_annotate.spectrogram_widget import SpectrogramWidget
from spec_annotate.synth import SynthEngine
from spec_annotate.cqt_settings_dialog import CQTSettingsDialog
from spec_annotate.utils.cqt import generate_spectrogram
from spec_annotate.utils import midi as midi_utils


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SpecAnnotate")
        # Set a sensible default window size
        self.resize(1200, 800)

        self.widget = SpectrogramWidget()
        self.setCentralWidget(self.widget)

        # Settings storage
        self._settings = QSettings("SpecAnnotate")

        self._audio_path: Path | None = None
        self._midi_path: Path | None = None
        self._sample_rate: int | None = None
        # CQT parameters (editable)
        self._hop_length: int = 128
        self._n_bins: int = 240
        self._bins_per_octave: int = 36
        # Use a musical note for f_min, derive MIDI for labeling
        self._f_min_note: str = "C2"
        self._f_min_midi: int = int(librosa.note_to_midi(self._f_min_note))
        self._power_scaling: float | None = 2

        # Build UI actions and menus/toolbar
        self._build_toolbar()
        # Ensure title/labels reflect initial state
        try:
            self._update_titles()
        except Exception:
            pass

        # Async loading members
        self._loader_thread: QThread | None = None
        # Narrow type to reduce warnings; use a forward reference to avoid NameError at import time
        self._loader: 'LoadWorker | None' = None
        self._progress: QProgressDialog | None = None

        # Persist horizontal scale changes
        self._restoring_settings: bool = False
        try:
            self.widget.x_scale_changed.connect(self._on_x_scale_changed)
        except Exception:
            pass

        # Connect note-created for preview playback
        try:
            self.widget.note_created.connect(self._on_note_created)
        except Exception:
            pass
        # Live pitch preview while dragging
        try:
            self.widget.pitch_preview_started.connect(self._on_pitch_preview_started)
            self.widget.pitch_preview_updated.connect(self._on_pitch_preview_updated)
            self.widget.pitch_preview_ended.connect(self._on_pitch_preview_ended)
        except Exception:
            pass

        # Set up media player for audio playback
        self._player = QMediaPlayer(self)
        self._player_output = QAudioOutput(self)
        self._player.setAudioOutput(self._player_output)
        # Sync playhead with playback
        self._suppress_seek = False
        try:
            self._player.positionChanged.connect(self._on_player_position_changed)
        except Exception:
            pass
        try:
            self.widget.marker_moved_seconds.connect(self._on_marker_moved)
        except Exception:
            pass
        # Keep references to active tone preview playback objects
        self._active_tones: list[tuple[QAudioSink, QBuffer]] = []

        # Built-in audition synth and scheduling state
        self._synth = SynthEngine(self)
        self._audition_enabled: bool = True
        # Map note index -> list of active voice ids (supports overlapping notes)
        self._active_note_voices: dict[int, list[int]] = {}
        # Active voice id for live pitch preview while dragging
        self._drag_preview_voice_id: int | None = None

        # Volume controls (0..1)
        self._track_volume: float = 1.0         # original track playback
        self._audition_volume: float = 0.2      # drawn notes during playback (synth master gain)
        self._preview_volume: float = 0.2       # short tone when drawing a note

        # Apply initial volumes to outputs
        try:
            self._player_output.setVolume(float(self._track_volume))
        except Exception:
            pass
        try:
            self._synth.master_gain = float(self._audition_volume)
        except Exception:
            pass

        # Load persisted settings and optionally auto-open last file
        self._load_settings()
        # Defer auto-loading until the window is shown so the progress dialog is visible
        self._startup_attempted = False


    def _build_toolbar(self):
        tb = QToolBar("Controls", self)
        tb.setMovable(False)
        # Compact, icon-only toolbar
        try:
            tb.setToolButtonStyle(Qt.ToolButtonIconOnly)
            tb.setIconSize(QSize(20, 20))
        except Exception:
            pass
        self.addToolBar(Qt.TopToolBarArea, tb)

        # Action setup
        self.action_open = QAction("Open Audio…", self)
        try:
            self.action_open.setIcon(QIcon.fromTheme("document-open"))
            if self.action_open.icon().isNull():
                self.action_open.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogOpenButton))
        except Exception:
            pass
        self.action_open.setShortcut("Ctrl+O")
        self.action_open.triggered.connect(self.open_audio)

        self.action_open_midi = QAction("Open MIDI…", self)
        try:
            self.action_open_midi.setIcon(QIcon.fromTheme("media-playlist-audio"))
            if self.action_open_midi.icon().isNull():
                # Fallback to generic open icon
                self.action_open_midi.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogOpenButton))
        except Exception:
            pass
        self.action_open_midi.setShortcut("Ctrl+M")
        self.action_open_midi.triggered.connect(self.open_midi)

        # New/blank MIDI action
        self.action_new_midi = QAction("New MIDI", self)
        try:
            # Try a sensible themed icon, otherwise use a generic new-file icon
            self.action_new_midi.setIcon(QIcon.fromTheme("document-new"))
            if self.action_new_midi.icon().isNull():
                self.action_new_midi.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_FileIcon))
        except Exception:
            pass
        # Use Ctrl+N for convenience; if it conflicts in the environment, it's still accessible via the toolbar
        self.action_new_midi.setShortcut("Ctrl+N")
        self.action_new_midi.triggered.connect(self.new_midi)

        self.action_save_midi = QAction("Save MIDI", self)
        try:
            self.action_save_midi.setIcon(QIcon.fromTheme("document-save"))
            if self.action_save_midi.icon().isNull():
                self.action_save_midi.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogSaveButton))
        except Exception:
            pass
        self.action_save_midi.setShortcut("Ctrl+S")
        self.action_save_midi.triggered.connect(self.save_midi)

        self.action_clear_notes = QAction("Clear Notes", self)
        try:
            self.action_clear_notes.setIcon(QIcon.fromTheme("edit-clear"))
            if self.action_clear_notes.icon().isNull():
                self.action_clear_notes.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_TrashIcon))
        except Exception:
            pass
        self.action_clear_notes.setShortcut("Ctrl+K")
        self.action_clear_notes.triggered.connect(self.widget.clear_notes)

        self.action_quit = QAction("Quit", self)
        try:
            self.action_quit.setIcon(QIcon.fromTheme("application-exit"))
        except Exception:
            pass
        self.action_quit.setShortcut("Ctrl+Q")
        self.action_quit.triggered.connect(self.close)

        self.action_cqt_settings = QAction("CQT Settings…", self)
        try:
            self.action_cqt_settings.setIcon(QIcon.fromTheme("preferences-system"))
            if self.action_cqt_settings.icon().isNull():
                self.action_cqt_settings.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_FileDialogDetailedView))
        except Exception:
            pass
        self.action_cqt_settings.setShortcut("Ctrl+,")
        self.action_cqt_settings.triggered.connect(self.open_cqt_settings)

        self.action_wider = QAction("Wider (increase X scale)", self)
        try:
            self.action_wider.setIcon(QIcon.fromTheme("zoom-in"))
            if self.action_wider.icon().isNull():
                self.action_wider.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_ArrowUp))
        except Exception:
            pass
        self.action_wider.setShortcut("Ctrl+=")
        self.action_wider.triggered.connect(
            lambda: self.widget.scale_x_by(1.25))

        self.action_narrower = QAction("Narrower (decrease X scale)", self)
        try:
            self.action_narrower.setIcon(QIcon.fromTheme("zoom-out"))
            if self.action_narrower.icon().isNull():
                self.action_narrower.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_ArrowDown))
        except Exception:
            pass
        self.action_narrower.setShortcut("Ctrl+-")
        self.action_narrower.triggered.connect(
            lambda: self.widget.scale_x_by(1.0 / 1.25))

        self.action_reset_width = QAction("Reset Width", self)
        try:
            self.action_reset_width.setIcon(QIcon.fromTheme("zoom-original"))
            if self.action_reset_width.icon().isNull():
                self.action_reset_width.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_BrowserReload))
        except Exception:
            pass
        self.action_reset_width.setShortcut("Ctrl+0")
        self.action_reset_width.triggered.connect(
            lambda: self.widget.reset_x_scale(4.0))

        self.action_undo = QAction("Undo", self)
        try:
            self.action_undo.setIcon(QIcon.fromTheme("edit-undo"))
            if self.action_undo.icon().isNull():
                self.action_undo.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_ArrowBack))
        except Exception:
            pass
        self.action_undo.setShortcut("Ctrl+Z")
        self.action_undo.triggered.connect(self.widget.undo_last_note)

        # Playback controls
        self.action_play = QAction("Play", self)
        try:
            self.action_play.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        except Exception:
            pass
        self.action_play.setShortcut("Space")
        # Space should toggle play/pause depending on current state
        self.action_play.triggered.connect(self.toggle_play_pause)

        self.action_pause = QAction("Pause", self)
        try:
            self.action_pause.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPause))
        except Exception:
            pass
        self.action_pause.triggered.connect(self.pause)

        self.action_stop = QAction("Stop", self)
        try:
            self.action_stop.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaStop))
        except Exception:
            pass
        self.action_stop.triggered.connect(self.stop)

        self.action_back1 = QAction("-1s", self)
        try:
            self.action_back1.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaSeekBackward))
        except Exception:
            pass
        self.action_back1.setShortcut("Left")
        self.action_back1.triggered.connect(lambda: self.nudge_seconds(-1.0))

        self.action_fwd1 = QAction("+1s", self)
        try:
            self.action_fwd1.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaSeekForward))
        except Exception:
            pass
        self.action_fwd1.setShortcut("Right")
        self.action_fwd1.triggered.connect(lambda: self.nudge_seconds(+1.0))

        # Audition toggle
        self.action_audition = QAction("Audition Notes", self)
        self.action_audition.setCheckable(True)
        self.action_audition.setChecked(True)
        try:
            # Use volume icon as a proxy for audition
            self.action_audition.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaVolume))
        except Exception:
            pass
        self.action_audition.toggled.connect(self._on_audition_toggled)

        # File/actions
        tb.addAction(self.action_open)
        tb.addAction(self.action_open_midi)
        tb.addAction(self.action_new_midi)
        tb.addAction(self.action_save_midi)
        tb.addAction(self.action_clear_notes)
        # Undo button
        tb.addAction(self.action_undo)
        tb.addSeparator()
        # View zoom/aspect
        tb.addAction(self.action_wider)
        tb.addAction(self.action_narrower)
        tb.addAction(self.action_reset_width)
        tb.addSeparator()
        # Playback controls
        tb.addAction(self.action_play)
        tb.addAction(self.action_pause)
        tb.addAction(self.action_stop)
        tb.addAction(self.action_back1)
        tb.addAction(self.action_fwd1)
        tb.addAction(self.action_audition)
        # (settings and quit added in final section)

        # Final section
        tb.addSeparator()
        tb.addAction(self.action_cqt_settings)
        tb.addAction(self.action_quit)

        # Build status bar with filenames and volume controls
        self._build_statusbar()

    def _build_statusbar(self):
        sb = self.statusBar()
        if sb is None:
            sb = QStatusBar(self)
            self.setStatusBar(sb)

        # Container for custom widgets
        container = QWidget(self)
        lay = QHBoxLayout(container)
        lay.setContentsMargins(6, 0, 6, 0)
        lay.setSpacing(10)

        # Filenames (left)
        self._lbl_audio_name = QLabel("Audio: (none)", self)
        self._lbl_audio_name.setToolTip("Currently opened audio file")
        lay.addWidget(self._lbl_audio_name)

        self._lbl_midi_name = QLabel("MIDI: untitled.mid", self)
        self._lbl_midi_name.setToolTip("Currently opened MIDI file")
        lay.addWidget(self._lbl_midi_name)

        # Spacer
        spacer = QWidget(self)
        # Use class-level QSizePolicy values (instance does not have Expanding/Preferred attributes)
        try:
            spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        except Exception:
            # Fallback for older bindings where enum is exposed directly on QSizePolicy
            spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        lay.addWidget(spacer)

        # Volume controls (right)
        # Track
        try:
            icon_track = self.style().standardIcon(QStyle.StandardPixmap.SP_MediaVolume)
        except Exception:
            icon_track = QIcon()
        lbl_track_icon = QLabel(self)
        lbl_track_icon.setToolTip("Original track volume")
        if not icon_track.isNull():
            lbl_track_icon.setPixmap(icon_track.pixmap(16, 16))
        else:
            lbl_track_icon.setText("Trk")
        lay.addWidget(lbl_track_icon)

        self._slider_track_vol = QSlider(Qt.Orientation.Horizontal, self)
        self._slider_track_vol.setMinimum(0)
        self._slider_track_vol.setMaximum(100)
        self._slider_track_vol.setFixedWidth(90)
        self._slider_track_vol.setToolTip("Original track volume")
        try:
            self._slider_track_vol.setValue(int(round(self._track_volume * 100)))
        except Exception:
            self._slider_track_vol.setValue(100)
        self._slider_track_vol.valueChanged.connect(self._on_track_volume_changed)
        lay.addWidget(self._slider_track_vol)

        # Audition
        lbl_aud_icon = QLabel(self)
        lbl_aud_icon.setToolTip("Drawn notes audition volume")
        if not icon_track.isNull():
            lbl_aud_icon.setPixmap(icon_track.pixmap(16, 16))
        else:
            lbl_aud_icon.setText("Aud")
        lay.addWidget(lbl_aud_icon)

        self._slider_aud_vol = QSlider(Qt.Orientation.Horizontal, self)
        self._slider_aud_vol.setMinimum(0)
        self._slider_aud_vol.setMaximum(100)
        self._slider_aud_vol.setFixedWidth(90)
        self._slider_aud_vol.setToolTip("Drawn notes audition volume")
        try:
            self._slider_aud_vol.setValue(int(round(self._audition_volume * 100)))
        except Exception:
            self._slider_aud_vol.setValue(20)
        self._slider_aud_vol.valueChanged.connect(self._on_audition_volume_changed)
        lay.addWidget(self._slider_aud_vol)

        # Preview
        lbl_prev_icon = QLabel(self)
        lbl_prev_icon.setToolTip("Preview tone volume (when drawing)")
        if not icon_track.isNull():
            lbl_prev_icon.setPixmap(icon_track.pixmap(16, 16))
        else:
            lbl_prev_icon.setText("Prv")
        lay.addWidget(lbl_prev_icon)

        self._slider_prev_vol = QSlider(Qt.Orientation.Horizontal, self)
        self._slider_prev_vol.setMinimum(0)
        self._slider_prev_vol.setMaximum(100)
        self._slider_prev_vol.setFixedWidth(90)
        self._slider_prev_vol.setToolTip("Preview tone volume (when drawing)")
        try:
            self._slider_prev_vol.setValue(int(round(self._preview_volume * 100)))
        except Exception:
            self._slider_prev_vol.setValue(20)
        self._slider_prev_vol.valueChanged.connect(self._on_preview_volume_changed)
        lay.addWidget(self._slider_prev_vol)

        container.setLayout(lay)
        sb.addPermanentWidget(container, 1)

    def _update_titles(self):
        try:
            # Derive display names
            audio_name = "(none)"
            if isinstance(self._audio_path, Path) and self._audio_path is not None:
                try:
                    audio_name = self._audio_path.name
                except Exception:
                    audio_name = str(self._audio_path)

            midi_name = "untitled.mid" if self._midi_path is None else (
                self._midi_path.name if isinstance(self._midi_path, Path) else str(self._midi_path)
            )

            # Update window title
            self.setWindowTitle(f"SpecAnnotate — {audio_name} — {midi_name}")

            # Update toolbar labels if they exist
            if hasattr(self, "_lbl_audio_name") and self._lbl_audio_name is not None:
                self._lbl_audio_name.setText(f"Audio: {audio_name}")
            if hasattr(self, "_lbl_midi_name") and self._lbl_midi_name is not None:
                self._lbl_midi_name.setText(f"MIDI: {midi_name}")
        except Exception:
            # Best-effort; ignore UI update errors
            pass

    def open_audio(self):
        path_str, _ = QFileDialog.getOpenFileName(
            self,
            "Open Audio File",
            "",
            "Audio Files (*.wav *.mp3 *.flac *.ogg *.m4a);;All Files (*)",
        )
        if not path_str:
            return
        self._audio_path = Path(path_str)
        self._save_settings()
        self._update_titles()
        self._start_load_for_path(self._audio_path)

    def open_midi(self):
        path_str, _ = QFileDialog.getOpenFileName(
            self,
            "Open MIDI File",
            "",
            "MIDI Files (*.mid *.midi);;All Files (*)",
        )
        if not path_str:
            return
        # Guard: require spectrogram so we can place notes on a timeline
        if self.widget is None:
            QMessageBox.warning(self, "No View", "Spectrogram view is not ready.")
            return
        if self.widget._cqt is None:
            QMessageBox.information(self, "Spectrogram not loaded", "Load an audio file first so MIDI notes can be placed on the timeline.")
            return

        # Prepare progress dialog (like audio loading)
        self._midi_progress = QProgressDialog("Loading MIDI…", "Cancel", 0, 0, self)
        self._midi_progress.setWindowTitle("Please wait")
        self._midi_progress.setWindowModality(Qt.ApplicationModal)
        self._midi_progress.setMinimumDuration(0)
        self._midi_progress.show()
        # Disable menus while loading
        self.menuBar().setEnabled(False)

        # Compute target duration from current CQT to align MIDI timeline
        try:
            n_frames = int(getattr(self.widget, "_n_frames", 0))
            hop = int(getattr(self.widget, "_hop_length", 0))
            sr = int(getattr(self.widget, "_sample_rate", 0))
            target_duration = (n_frames * hop / float(sr)) if (n_frames > 0 and hop > 0 and sr > 0) else None
        except Exception:
            target_duration = None

        # Record current MIDI path and spin up worker thread for MIDI parsing
        try:
            self._midi_path = Path(path_str)
        except Exception:
            self._midi_path = None
        self._update_titles()

        # Spin up worker thread for MIDI parsing
        self._midi_loader_thread = QThread(self)
        self._midi_loader = MidiLoadWorker(path=str(path_str), target_duration_sec=target_duration)
        self._midi_loader.moveToThread(self._midi_loader_thread)

        # Wire signals
        self._midi_loader_thread.started.connect(self._midi_loader.process)
        self._midi_loader.stage_changed.connect(self._on_midi_stage_changed)
        self._midi_loader.finished.connect(self._on_midi_finished)
        self._midi_loader.error.connect(self._on_midi_error)
        self._midi_loader.done.connect(self._midi_loader_thread.quit)
        self._midi_loader.done.connect(self._midi_loader.deleteLater)
        self._midi_loader_thread.finished.connect(self._midi_loader_thread.deleteLater)

        # Cancel handling
        if self._midi_progress is not None:
            self._midi_progress.canceled.connect(self._midi_loader.request_cancel)

        self._midi_loader_thread.start()

    def _start_load_for_path(self, path: Path):
        # Prepare progress dialog
        self._progress = QProgressDialog("Loading audio…", "Cancel", 0, 0, self)
        self._progress.setWindowTitle("Please wait")
        self._progress.setWindowModality(Qt.ApplicationModal)
        self._progress.setMinimumDuration(0)
        self._progress.show()
        # Prevent re-entry while loading
        self.menuBar().setEnabled(False)

        # Spin up worker thread
        self._loader_thread = QThread(self)
        self._loader = LoadWorker(
            path=str(path),
            hop_length=self._hop_length,
            n_bins=self._n_bins,
            bins_per_octave=self._bins_per_octave,
            f_min_note=self._f_min_note,
            power_scaling=self._power_scaling,
        )
        self._loader.moveToThread(self._loader_thread)

        # Wire signals
        self._loader_thread.started.connect(self._loader.process)
        self._loader.stage_changed.connect(self._on_stage_changed)
        self._loader.finished.connect(self._on_load_finished)
        self._loader.error.connect(self._on_load_error)
        self._loader.done.connect(self._loader_thread.quit)
        self._loader.done.connect(self._loader.deleteLater)
        self._loader_thread.finished.connect(self._loader_thread.deleteLater)

        # Cancel handling
        if self._progress is not None:
            self._progress.canceled.connect(self._loader.request_cancel)

        self._loader_thread.start()

    def open_cqt_settings(self):
        dlg = CQTSettingsDialog(
            hop_length=self._hop_length,
            n_bins=self._n_bins,
            bins_per_octave=self._bins_per_octave,
            f_min_note=self._f_min_note,
            power_scaling=self._power_scaling,
            parent=self,
        )
        if dlg.exec():
            vals = dlg.values()
            self._hop_length = vals["hop_length"]
            self._n_bins = vals["n_bins"]
            self._bins_per_octave = vals["bins_per_octave"]
            # Store note string; derive MIDI for labeling
            self._f_min_note = vals["f_min_note"]
            try:
                self._f_min_midi = int(librosa.note_to_midi(self._f_min_note))
            except Exception:
                # Fallback to C0 if something went wrong
                self._f_min_note = "C0"
                self._f_min_midi = int(librosa.note_to_midi(self._f_min_note))
            self._power_scaling = vals["power_scaling"]
            self._save_settings()
            # If an audio file is already loaded/opened, recompute using new settings
            if self._audio_path is not None and self._audio_path.exists():
                self._start_load_for_path(self._audio_path)

    def save_midi(self):
        if not self.widget.has_notes():
            QMessageBox.information(self, "No Notes", "There are no notes to save.")
            return
        # Decide path: save directly to currently opened MIDI if available; otherwise Save As
        save_path: str | None
        if self._midi_path is not None:
            save_path = str(self._midi_path)
        else:
            path_str, _ = QFileDialog.getSaveFileName(
                self,
                "Save MIDI",
                "untitled.mid",
                "MIDI Files (*.mid *.midi)",
            )
            if not path_str:
                return
            try:
                self._midi_path = Path(path_str)
            except Exception:
                self._midi_path = Path(path_str)
            save_path = path_str
            self._update_titles()
        # Calculate track duration to ensure MIDI file length matches the audio
        duration_sec = None
        try:
            n_frames = int(getattr(self.widget, "_n_frames", 0))
            hop = int(getattr(self.widget, "_hop_length", 0))
            sr = int(getattr(self.widget, "_sample_rate", 0))
            if n_frames > 0 and hop > 0 and sr > 0:
                duration_sec = n_frames * hop / float(sr)
        except Exception:
            pass
        notes = self.widget.export_notes_seconds()
        try:
            midi_utils.export_notes_to_midi(notes, save_path, tempo_bpm=120, ppq=480, duration_sec=duration_sec)
            QMessageBox.information(self, "Saved", f"MIDI saved to: {save_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save MIDI: {e}")

    def new_midi(self):
        """
        Replace the currently opened MIDI with a blank (untitled) one.
        This clears any drawn/imported notes and resets the MIDI path/state.
        """
        try:
            if getattr(self, "widget", None) is not None:
                # Clear any existing notes from the scene
                self.widget.clear_notes()
        except Exception:
            # Best-effort: continue with resetting state even if UI clearing failed
            pass

        # Reset MIDI path to indicate a new, untitled MIDI session
        self._midi_path = None

        # Refresh window/status labels
        self._update_titles()

        # Optional: give subtle feedback without interrupting flow
        try:
            self.statusBar().showMessage("New blank MIDI ready", 2000)
        except Exception:
            pass

    # Slots for async loader
    @Slot(str)
    def _on_stage_changed(self, text: str):
        if self._progress is not None:
            self._progress.setLabelText(text)

    @Slot(object, int)
    def _on_load_finished(self, cqt_img, sample_rate: int):
        # Close progress
        if self._progress is not None:
            self._progress.close()
            self._progress = None
        self.menuBar().setEnabled(True)

        self._sample_rate = int(sample_rate)
        # Update widget
        self.widget.set_spectrogram(
            cqt_img,
            sample_rate=self._sample_rate,
            hop_length=self._hop_length,
            f_min_midi=self._f_min_midi,
            bins_per_octave=self._bins_per_octave,
        )
        # Fit the spectrogram to the current window on initial load for a good first view
        self.widget.fit_to_window()
        # Ensure saved x-scale is applied after first image build (handled in _load_settings too)
        # Set media player source if we know the audio path
        if self._audio_path is not None and self._audio_path.exists():
            try:
                self._player.setSource(QUrl.fromLocalFile(str(self._audio_path)))
            except Exception:
                pass
        # Refresh title bar/labels
        self._update_titles()

    @Slot(str)
    def _on_load_error(self, message: str):
        if self._progress is not None:
            self._progress.close()
            self._progress = None
        self.menuBar().setEnabled(True)
        QMessageBox.critical(self, "Error", message)

    # MIDI loading slots ---------------------------------------------------
    @Slot(str)
    def _on_midi_stage_changed(self, text: str):
        if getattr(self, "_midi_progress", None) is not None:
            self._midi_progress.setLabelText(text)

    @Slot(object)
    def _on_midi_finished(self, tuples_obj):
        # Close progress
        if getattr(self, "_midi_progress", None) is not None:
            try:
                self._midi_progress.close()
            except Exception:
                pass
            self._midi_progress = None
        self.menuBar().setEnabled(True)

        try:
            tuples = list(tuples_obj)
        except Exception:
            tuples = []
        # Apply to widget if still available
        if self.widget is not None and self.widget._cqt is not None:
            try:
                self.widget.set_notes_seconds(tuples)
                QMessageBox.information(self, "MIDI Loaded", f"Loaded {len(tuples)} notes from MIDI.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to apply MIDI: {e}")
        # Update titles now that MIDI is considered open in session
        self._update_titles()

    @Slot(str)
    def _on_midi_error(self, message: str):
        if getattr(self, "_midi_progress", None) is not None:
            try:
                self._midi_progress.close()
            except Exception:
                pass
            self._midi_progress = None
        self.menuBar().setEnabled(True)
        QMessageBox.critical(self, "Error", message)

    # Playback controls ---------------------------------------------------
    def play(self):
        if self._player.source().isEmpty():
            if self._audio_path is not None and self._audio_path.exists():
                self._player.setSource(QUrl.fromLocalFile(str(self._audio_path)))
        try:
            self._player.play()
            # When starting playback, resync audition to current position
            if self._audition_enabled:
                t = float(self._player.position()) / 1000.0
                self._audition_update_for_time(t)
        except Exception:
            pass

    def pause(self):
        try:
            self._player.pause()
        except Exception:
            pass
        # Stop all audition voices on pause (prevents hanging tones)
        self._audition_stop_all()

    def stop(self):
        try:
            self._player.stop()
        except Exception:
            pass
        # Stop all audition voices on stop
        self._audition_stop_all()

    def toggle_play_pause(self):
        """Toggle playback state: if currently playing, pause; otherwise play.
        Bound to Space via the Play action's shortcut.
        """
        try:
            state = self._player.playbackState()
        except Exception:
            state = None

        try:
            if state == QMediaPlayer.PlaybackState.PlayingState:
                self.pause()
            else:
                self.play()
        except Exception:
            # Fallback: attempt to play if anything goes wrong
            try:
                self.play()
            except Exception:
                pass

    def nudge_seconds(self, delta: float):
        try:
            cur_ms = int(self._player.position())
            new_ms = max(0, cur_ms + int(delta * 1000.0))
            self._suppress_seek = True
            self._player.setPosition(new_ms)
            self._suppress_seek = False
            self.widget.set_playback_position_seconds(new_ms / 1000.0)
        except Exception:
            pass

    @Slot(int)
    def _on_player_position_changed(self, pos_ms: int):
        # Avoid feedback when we programmatically seek
        if self._suppress_seek:
            return
        self.widget.set_playback_position_seconds(pos_ms / 1000.0)
        if self._audition_enabled:
            self._audition_update_for_time(float(pos_ms) / 1000.0)

    @Slot(float)
    def _on_marker_moved(self, seconds: float):
        # Seek the player when user drags the marker
        try:
            self._suppress_seek = True
            self._player.setPosition(int(max(0.0, seconds) * 1000.0))
        finally:
            self._suppress_seek = False
        # Immediately resync audition to the new position as well
        if self._audition_enabled:
            self._audition_update_for_time(float(seconds))

    # Note preview playback -----------------------------------------------
    @Slot(int, int, int, int)
    def _on_note_created(self, pitch: int, start_frame: int, end_frame: int, velocity: int):
        # Preview the created note using the SynthEngine (short note-on/note-off)
        try:
            # Skip if preview volume is effectively muted
            if float(self._preview_volume) <= 0.0:
                return
            freq = float(librosa.midi_to_hz(int(pitch)))
            secs_per_frame = self._hop_length / float(self._sample_rate or 22050)
            dur = max(0.05, min(0.6, (end_frame - start_frame) * secs_per_frame))
            # Map preview volume (0..1) to MIDI velocity 1..127
            vel = max(1, min(127, int(round(float(self._preview_volume) * 127.0))))
            vid = self._synth.note_on(freq, vel)
            if vid is not None:
                # Schedule note_off after the short preview duration
                QTimer.singleShot(int(dur * 1000.0), lambda v=vid: self._synth.note_off(int(v)))
        except Exception:
            pass

    def _play_tone(self, freq_hz: float, duration_sec: float, volume: float | None = None):
        try:
            sr = 44100
            t = np.arange(int(duration_sec * sr), dtype=np.float32) / sr
            # Generate base sine at full scale; use sink volume to control loudness
            wave = np.sin(2 * np.pi * float(freq_hz) * t)
            # Simple attack/decay envelope
            n = len(wave)
            if n > 8:
                env = np.ones(n, dtype=np.float32)
                a = max(1, int(0.01 * sr))
                d = max(1, int(0.03 * sr))
                env[:a] = np.linspace(0, 1, a, dtype=np.float32)
                env[-d:] = np.linspace(1, 0, d, dtype=np.float32)
                wave *= env
            # Convert to 16-bit PCM mono
            pcm = np.clip(wave, -1.0, 1.0)
            pcm_i16 = (pcm * 32767.0).astype(np.int16)
            data = QByteArray(bytes(pcm_i16))
            buf = QBuffer()
            buf.setData(data)
            buf.open(QBuffer.ReadOnly)

            fmt = QAudioFormat()
            fmt.setSampleRate(sr)
            fmt.setChannelCount(1)
            fmt.setSampleFormat(QAudioFormat.SampleFormat.Int16)

            device = QMediaDevices.defaultAudioOutput()
            sink = QAudioSink(device, fmt)
            # Apply preview volume (0..1)
            try:
                sink.setVolume(float(self._preview_volume))
            except Exception:
                pass
            sink.start(buf)

            # Keep references until finished; schedule cleanup when buffer drains
            self._active_tones.append((sink, buf))
            def _cleanup_state():
                # QAudio::IdleState when finished
                try:
                    if sink.bytesFree() > 0 and buf.atEnd():
                        sink.stop()
                except Exception:
                    pass
            try:
                # Some PySide6 builds have trouble converting QAudio::State to Python types in slots.
                # Use a lambda to ignore the enum argument and call our no-arg cleanup safely.
                sink.stateChanged.connect(lambda _state: _cleanup_state())
            except Exception:
                pass
        except Exception:
            pass

    # Volume handlers -----------------------------------------------------
    def _on_track_volume_changed(self, value: int):
        try:
            self._track_volume = max(0.0, min(1.0, float(value) / 100.0))
            self._player_output.setVolume(float(self._track_volume))
            if not self._restoring_settings:
                self._save_settings()
        except Exception:
            pass

    def _on_audition_volume_changed(self, value: int):
        try:
            self._audition_volume = max(0.0, min(1.0, float(value) / 100.0))
            self._synth.master_gain = float(self._audition_volume)
            if not self._restoring_settings:
                self._save_settings()
        except Exception:
            pass

    def _on_preview_volume_changed(self, value: int):
        try:
            self._preview_volume = max(0.0, min(1.0, float(value) / 100.0))
            # Update any active preview sinks
            for sink, _buf in list(self._active_tones):
                try:
                    sink.setVolume(float(self._preview_volume))
                except Exception:
                    pass
            if not self._restoring_settings:
                self._save_settings()
        except Exception:
            pass

    # Live pitch preview handlers ----------------------------------------
    def _on_pitch_preview_started(self, pitch: int):
        try:
            # Stop any existing preview voice first
            if self._drag_preview_voice_id is not None:
                try:
                    self._synth.note_off(int(self._drag_preview_voice_id))
                except Exception:
                    pass
                self._drag_preview_voice_id = None
            # If preview volume effectively zero, skip starting
            if float(self._preview_volume) <= 0.0:
                return
            freq = float(librosa.midi_to_hz(int(pitch)))
            # Map preview volume (0..1) to velocity (1..127)
            vel = max(1, min(127, int(round(float(self._preview_volume) * 127.0))))
            vid = self._synth.note_on(freq, vel)
            if vid is not None:
                self._drag_preview_voice_id = int(vid)
        except Exception:
            pass

    def _on_pitch_preview_updated(self, pitch: int):
        try:
            if self._drag_preview_voice_id is None:
                # If no active voice (e.g., volume toggled during drag), start one
                self._on_pitch_preview_started(pitch)
                return
            freq = float(librosa.midi_to_hz(int(pitch)))
            self._synth.set_voice_freq(int(self._drag_preview_voice_id), freq)
        except Exception:
            pass

    def _on_pitch_preview_ended(self):
        try:
            if self._drag_preview_voice_id is not None:
                try:
                    self._synth.note_off(int(self._drag_preview_voice_id))
                except Exception:
                    pass
                self._drag_preview_voice_id = None
        except Exception:
            pass

    # Audition engine integration -----------------------------------------
    def _on_audition_toggled(self, enabled: bool):
        self._audition_enabled = bool(enabled)
        if not self._audition_enabled:
            self._audition_stop_all()
        else:
            # Resync to current time
            t = float(self._player.position()) / 1000.0
            self._audition_update_for_time(t)

    def _audition_stop_all(self):
        # Send note_off to all active voices and clear state
        try:
            for idx, vids in list(self._active_note_voices.items()):
                for vid in vids:
                    try:
                        if vid is not None:
                            self._synth.note_off(int(vid))
                    except Exception:
                        pass
            self._active_note_voices.clear()
            # Allow envelopes to release naturally; no hard stop to avoid clicks
        except Exception:
            pass

    def _audition_update_for_time(self, t_seconds: float):
        """
        Start/stop synth voices so that notes overlapping t_seconds are sounding.
        Notes are derived from the widget's current drawn notes.
        """
        try:
            if not self._audition_enabled:
                return
            if not hasattr(self.widget, 'export_notes_seconds'):
                return
            notes = self.widget.export_notes_seconds()  # list of (pitch, start, end, velocity)
            # Build set of currently active note indices for the given time
            want_active_idxs: set[int] = set()
            for i, (pitch, start, end, vel) in enumerate(notes):
                if start <= t_seconds < end:
                    want_active_idxs.add(i)

            # Stop voices for notes that are no longer active
            for idx in list(self._active_note_voices.keys()):
                if idx not in want_active_idxs:
                    vids = self._active_note_voices.pop(idx, [])
                    for vid in vids:
                        try:
                            if vid is not None:
                                self._synth.note_off(int(vid))
                        except Exception:
                            pass

            # Start voices for newly active notes
            for i in sorted(want_active_idxs):
                if i in self._active_note_voices:
                    continue  # already sounding
                try:
                    pitch, start, end, vel = notes[i]
                    freq = float(librosa.midi_to_hz(int(pitch)))
                except Exception:
                    continue
                vid = self._synth.note_on(freq, int(vel))
                if vid is not None:
                    self._active_note_voices[i] = [int(vid)]
        except Exception:
            pass

    # Settings persistence -------------------------------------------------
    def _load_settings(self):
        try:
            # Prevent signal-driven saves while we are restoring settings
            self._restoring_settings = True
            # Horizontal zoom (robust parsing)
            try:
                x_val = self._settings.value("view/x_scale", 4.0)
                x_str = str(x_val).strip()
                x_scale = float(x_str) if x_str != "" else 4.0
                self.widget.set_x_scale(float(x_scale))
            except Exception:
                self.widget.set_x_scale(4.0)

            # CQT params (robust parsing per key)
            try:
                hop_val = self._settings.value("cqt/hop_length", self._hop_length)
                self._hop_length = int(hop_val) if str(hop_val).strip() != "" else self._hop_length
            except Exception:
                pass
            try:
                nb_val = self._settings.value("cqt/n_bins", self._n_bins)
                self._n_bins = int(nb_val) if str(nb_val).strip() != "" else self._n_bins
            except Exception:
                pass
            try:
                bpo_val = self._settings.value("cqt/bins_per_octave", self._bins_per_octave)
                self._bins_per_octave = int(bpo_val) if str(bpo_val).strip() != "" else self._bins_per_octave
            except Exception:
                pass
            try:
                fmin_val = self._settings.value("cqt/f_min_note", self._f_min_note)
                self._f_min_note = str(fmin_val) if str(fmin_val).strip() != "" else self._f_min_note
            except Exception:
                pass
            try:
                power_val = self._settings.value("cqt/power_scaling", "")
                if power_val == "" or power_val is None:
                    self._power_scaling = None
                else:
                    self._power_scaling = float(power_val)
            except Exception:
                self._power_scaling = None
            # Derive MIDI for labels
            try:
                self._f_min_midi = int(librosa.note_to_midi(self._f_min_note))
            except Exception:
                self._f_min_note = "C0"
                self._f_min_midi = int(librosa.note_to_midi(self._f_min_note))

            # Last audio path
            try:
                last_path = self._settings.value("session/last_audio_path", "")
            except Exception:
                last_path = ""
            p = Path(str(last_path)) if (isinstance(last_path, str) and last_path) else None
            self._audio_path = p if (p is not None and p.exists()) else None

            # Volumes
            try:
                tv = self._settings.value("audio/track_volume", self._track_volume)
                self._track_volume = float(tv)
            except Exception:
                pass
            try:
                av = self._settings.value("audio/audition_volume", self._audition_volume)
                self._audition_volume = float(av)
            except Exception:
                pass
            try:
                pv = self._settings.value("audio/preview_volume", self._preview_volume)
                self._preview_volume = float(pv)
            except Exception:
                pass

            # Apply restored volumes to UI/components
            try:
                self._player_output.setVolume(float(self._track_volume))
            except Exception:
                pass
            try:
                self._synth.master_gain = float(self._audition_volume)
            except Exception:
                pass
            try:
                if hasattr(self, "_slider_track_vol"):
                    self._slider_track_vol.setValue(int(round(self._track_volume * 100)))
                if hasattr(self, "_slider_aud_vol"):
                    self._slider_aud_vol.setValue(int(round(self._audition_volume * 100)))
                if hasattr(self, "_slider_prev_vol"):
                    self._slider_prev_vol.setValue(int(round(self._preview_volume * 100)))
            except Exception:
                pass
        except Exception:
            # Ignore corrupt settings
            pass
        finally:
            self._restoring_settings = False

    def _save_settings(self):
        try:
            # View
            self._settings.setValue("view/x_scale", float(self.widget.get_x_scale()))
            # CQT
            self._settings.setValue("cqt/hop_length", self._hop_length)
            self._settings.setValue("cqt/n_bins", self._n_bins)
            self._settings.setValue("cqt/bins_per_octave", self._bins_per_octave)
            self._settings.setValue("cqt/f_min_note", self._f_min_note)
            self._settings.setValue("cqt/power_scaling", "" if self._power_scaling is None else float(self._power_scaling))
            # Session
            self._settings.setValue("session/last_audio_path", "" if self._audio_path is None else str(self._audio_path))
            # Audio volumes
            self._settings.setValue("audio/track_volume", float(self._track_volume))
            self._settings.setValue("audio/audition_volume", float(self._audition_volume))
            self._settings.setValue("audio/preview_volume", float(self._preview_volume))
            self._settings.sync()
        except Exception:
            pass

    # Ensure settings are saved when closing
    def closeEvent(self, event):  # type: ignore[override]
        try:
            self._save_settings()
            try:
                # Gracefully stop the synth engine's audio stream
                if hasattr(self, "_synth") and self._synth is not None:
                    self._synth.stop()
            except Exception:
                pass
        finally:
            super().closeEvent(event)

    # Internal slot to persist x-scale changes without triggering during restore
    @Slot(float)
    def _on_x_scale_changed(self, _: float):
        if self._restoring_settings:
            return
        self._save_settings()

    # Defer auto-opening the last session until first show, so users see the progress dialog
    def showEvent(self, event):  # type: ignore[override]
        super().showEvent(event)
        if not getattr(self, "_startup_attempted", False):
            self._startup_attempted = True
            if self._audio_path is not None and self._audio_path.exists():
                self._start_load_for_path(self._audio_path)


class LoadWorker(QObject):
    stage_changed = Signal(str)
    finished = Signal(object, int)  # (cqt, sample_rate)
    error = Signal(str)
    done = Signal()

    def __init__(self, *, path: str, hop_length: int, n_bins: int, bins_per_octave: int, f_min_note: str, power_scaling: float | None):
        super().__init__()
        self._path = path
        self._hop_length = int(hop_length)
        self._n_bins = int(n_bins)
        self._bins_per_octave = int(bins_per_octave)
        self._f_min_note = str(f_min_note)
        self._power_scaling = None if power_scaling is None else float(power_scaling)
        self._cancel = False

    @Slot()
    def process(self):
        try:
            self.stage_changed.emit("Loading audio…")
            # Keep stereo/multichannel; our generator will mono-ize
            y, sr = librosa.load(self._path, mono=False)
            if self._cancel:
                self.done.emit()
                return

            self.stage_changed.emit("Computing CQT…")
            cqt_img = generate_spectrogram(
                y,
                hop_length=self._hop_length,
                sample_rate=int(sr),
                n_bins=self._n_bins,
                bins_per_octave=self._bins_per_octave,
                f_min=librosa.note_to_hz(self._f_min_note),
                power_scaling=self._power_scaling,
            )
            if self._cancel:
                self.done.emit()
                return

            self.finished.emit(cqt_img, int(sr))
        except Exception as e:
            self.error.emit(f"Failed to load audio: {e}")
        finally:
            self.done.emit()

    @Slot()
    def request_cancel(self):
        self._cancel = True


class MidiLoadWorker(QObject):
    stage_changed = Signal(str)
    finished = Signal(object)  # list of tuples (pitch, start_s, end_s, velocity)
    error = Signal(str)
    done = Signal()

    def __init__(self, *, path: str, target_duration_sec: float | None = None):
        super().__init__()
        self._path = str(path)
        self._target_duration_sec = None if target_duration_sec is None else float(target_duration_sec)
        self._cancel = False

    @Slot()
    def process(self):
        try:
            self.stage_changed.emit("Parsing MIDI…")
            notes = midi_utils.import_notes_from_midi(self._path, target_duration_sec=self._target_duration_sec)
            if self._cancel:
                self.done.emit()
                return
            tuples = [(n.pitch, n.start, n.end, n.velocity) for n in notes]
            self.finished.emit(tuples)
        except Exception as e:
            self.error.emit(f"Failed to load MIDI: {e}")
        finally:
            self.done.emit()

    @Slot()
    def request_cancel(self):
        self._cancel = True
