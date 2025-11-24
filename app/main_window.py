from pathlib import Path

import librosa
from PySide6.QtCore import Qt, QObject, Signal, Slot, QThread, QSettings, QByteArray, QBuffer
from PySide6.QtGui import QAction, QIcon
from PySide6.QtWidgets import (
    QMainWindow,
    QFileDialog,
    QMessageBox,
    QProgressDialog,
    QToolBar,
)
from PySide6.QtMultimedia import (
    QMediaPlayer,
    QAudioOutput,
    QAudioSink,
    QAudioFormat,
    QMediaDevices,
)
from PySide6.QtCore import QUrl
import numpy as np

from .spectrogram_widget import SpectrogramWidget
from .cqt_settings_dialog import CQTSettingsDialog
from utils.cqt import generate_spectrogram
from utils import midi as midi_utils


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Spectrogram Annotation")
        # Set a sensible default window size
        self.resize(1200, 800)

        self.widget = SpectrogramWidget()
        self.setCentralWidget(self.widget)

        # Settings storage
        self._settings = QSettings("bendavison", "SpectrogramAnnotation")

        self._audio_path: Path | None = None
        self._sample_rate: int | None = None
        # CQT parameters (editable)
        self._hop_length: int = 128
        self._n_bins: int = 128
        self._bins_per_octave: int = 12
        # Use a musical note for f_min, derive MIDI for labeling
        self._f_min_note: str = "C0"
        self._f_min_midi: int = int(librosa.note_to_midi(self._f_min_note))
        self._power_scaling: float | None = 0.8

        # Build UI actions and menus/toolbar
        self._build_toolbar()

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

        # Load persisted settings and optionally auto-open last file
        self._load_settings()
        # Defer auto-loading until the window is shown so the progress dialog is visible
        self._startup_attempted = False


    def _build_toolbar(self):
        tb = QToolBar("Controls", self)
        tb.setMovable(False)
        self.addToolBar(Qt.TopToolBarArea, tb)

        # Action setup
        self.action_open = QAction("Open Audio…", self)
        self.action_open.setShortcut("Ctrl+O")
        self.action_open.triggered.connect(self.open_audio)

        self.action_open_midi = QAction("Open MIDI…", self)
        self.action_open_midi.setShortcut("Ctrl+M")
        self.action_open_midi.triggered.connect(self.open_midi)

        self.action_save_midi = QAction("Save MIDI…", self)
        self.action_save_midi.setShortcut("Ctrl+S")
        self.action_save_midi.triggered.connect(self.save_midi)

        self.action_clear_notes = QAction("Clear Notes", self)
        self.action_clear_notes.setShortcut("Ctrl+K")
        self.action_clear_notes.triggered.connect(self.widget.clear_notes)

        self.action_quit = QAction("Quit", self)
        self.action_quit.setShortcut("Ctrl+Q")
        self.action_quit.triggered.connect(self.close)

        self.action_cqt_settings = QAction("CQT Settings…", self)
        self.action_cqt_settings.setShortcut("Ctrl+,")
        self.action_cqt_settings.triggered.connect(self.open_cqt_settings)

        self.action_wider = QAction("Wider (increase X scale)", self)
        self.action_wider.setShortcut("Ctrl+=")
        self.action_wider.triggered.connect(
            lambda: self.widget.scale_x_by(1.25))

        self.action_narrower = QAction("Narrower (decrease X scale)", self)
        self.action_narrower.setShortcut("Ctrl+-")
        self.action_narrower.triggered.connect(
            lambda: self.widget.scale_x_by(1.0 / 1.25))

        self.action_reset_width = QAction("Reset Width", self)
        self.action_reset_width.setShortcut("Ctrl+0")
        self.action_reset_width.triggered.connect(
            lambda: self.widget.reset_x_scale(4.0))

        self.action_undo = QAction("Undo", self)
        self.action_undo.setShortcut("Ctrl+Z")
        self.action_undo.triggered.connect(self.widget.undo_last_note)

        # Playback controls
        self.action_play = QAction("Play", self)
        self.action_play.setShortcut("Space")
        self.action_play.triggered.connect(self.play)

        self.action_pause = QAction("Pause", self)
        self.action_pause.triggered.connect(self.pause)

        self.action_stop = QAction("Stop", self)
        self.action_stop.triggered.connect(self.stop)

        self.action_back1 = QAction("-1s", self)
        self.action_back1.setShortcut("Left")
        self.action_back1.triggered.connect(lambda: self.nudge_seconds(-1.0))

        self.action_fwd1 = QAction("+1s", self)
        self.action_fwd1.setShortcut("Right")
        self.action_fwd1.triggered.connect(lambda: self.nudge_seconds(+1.0))

        # File/actions
        tb.addAction(self.action_open)
        tb.addAction(self.action_open_midi)
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
        tb.addSeparator()
        tb.addAction(self.action_cqt_settings)

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
        path_str, _ = QFileDialog.getSaveFileName(
            self,
            "Save MIDI",
            "annotations.mid",
            "MIDI Files (*.mid *.midi)",
        )
        if not path_str:
            return
        notes = self.widget.export_notes_seconds()
        try:
            midi_utils.export_notes_to_midi(notes, path_str, tempo_bpm=120, ppq=480)
            QMessageBox.information(self, "Saved", f"MIDI saved to: {path_str}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save MIDI: {e}")

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
        except Exception:
            pass

    def pause(self):
        try:
            self._player.pause()
        except Exception:
            pass

    def stop(self):
        try:
            self._player.stop()
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

    @Slot(float)
    def _on_marker_moved(self, seconds: float):
        # Seek the player when user drags the marker
        try:
            self._suppress_seek = True
            self._player.setPosition(int(max(0.0, seconds) * 1000.0))
        finally:
            self._suppress_seek = False

    # Note preview playback -----------------------------------------------
    @Slot(int, int, int, int)
    def _on_note_created(self, pitch: int, start_frame: int, end_frame: int, velocity: int):
        # Play a short sine tone at the note's frequency; cap duration to 600ms
        try:
            freq = float(librosa.midi_to_hz(pitch))
        except Exception:
            return
        secs_per_frame = self._hop_length / float(self._sample_rate or 22050)
        dur = max(0.05, min(0.6, (end_frame - start_frame) * secs_per_frame))
        self._play_tone(freq, dur, volume=0.2)

    def _play_tone(self, freq_hz: float, duration_sec: float, volume: float = 0.5):
        try:
            sr = 44100
            t = np.arange(int(duration_sec * sr), dtype=np.float32) / sr
            wave = np.sin(2 * np.pi * float(freq_hz) * t) * float(volume)
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
            sink.start(buf)

            # Keep references until finished; schedule cleanup when buffer drains
            self._active_tones.append((sink, buf))
            def _cleanup_state(_state):
                # QAudio::IdleState when finished
                try:
                    if sink.bytesFree() > 0 and buf.atEnd():
                        sink.stop()
                except Exception:
                    pass
            try:
                sink.stateChanged.connect(_cleanup_state)
            except Exception:
                pass
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
            self._settings.sync()
        except Exception:
            pass

    # Ensure settings are saved when closing
    def closeEvent(self, event):  # type: ignore[override]
        try:
            self._save_settings()
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
            from utils import midi as midi_utils
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
