from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from PySide6.QtCore import Qt, QRectF, QPointF, Signal
from PySide6.QtGui import QImage, QPixmap, QColor, QMouseEvent, QPainter, QPen, QFont, QWheelEvent
from PySide6.QtWidgets import (
    QGraphicsView,
    QGraphicsScene,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsLineItem,
)
from matplotlib import cm


@dataclass
class DrawnNote:
    pitch: int           # MIDI pitch number
    start_frame: int     # inclusive
    end_frame: int       # exclusive
    velocity: int = 64


class SpectrogramWidget(QGraphicsView):
    # Emitted whenever horizontal pixel-per-frame scale changes (float pixels/frame)
    x_scale_changed = Signal(float)
    # Emitted when a note is finalized by drawing: (pitch, start_frame, end_frame, velocity)
    note_created = Signal(int, int, int, int)
    # Emitted when the playback marker/playhead is moved by the user (seconds)
    marker_moved_seconds = Signal(float)
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRenderHints(self.renderHints())
        self.setDragMode(QGraphicsView.NoDrag)
        self.setMouseTracking(True)

        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        # Important for pinned Y-axis labels drawn in viewport coordinates:
        # ensure the entire viewport repaints on scroll/drag to avoid label "ghosting" artifacts
        # that can happen with the default scroll-optimized update modes.
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)

        self._pix_item: QGraphicsPixmapItem | None = None
        self._notes: List[DrawnNote] = []
        self._note_items: List[QGraphicsRectItem] = []
        # Playback marker
        self._playhead_item: QGraphicsLineItem | None = None
        self._playhead_seconds: float = 0.0
        self._dragging_playhead: bool = False

        # Spectrogram metadata
        self._cqt: np.ndarray | None = None  # shape (n_bins, n_frames) in [0,1]
        self._n_bins = 0
        self._n_frames = 0
        self._sample_rate = 22050
        self._hop_length = 128
        self._f_min_midi = 0
        self._bins_per_octave = 12

        # Visual scaling so it’s visible
        self._x_scale = 4.0  # pixels per frame (logical; image will be rebuilt)
        self._y_scale = 4  # pixels per bin (will be adjusted to fit height)
        # Reserve a left gutter in the scene for pinned Y-axis labels
        self._left_gutter = 64  # pixels

        # Zoom state (view transform-based)
        self._user_zoomed = False
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)

        # Interactive drawing
        self._drawing = False
        self._drag_start_scene: QPointF | None = None
        self._rubberband_item: QGraphicsRectItem | None = None

        # Note interaction (drag/move/delete)
        self._dragging_note_item: QGraphicsRectItem | None = None
        self._dragging_note_index: int | None = None
        self._dragging_note_grab_frame_offset: int = 0
        self._dragging_note_length: int = 0

    # Public API
    def set_spectrogram(self, cqt: np.ndarray, *, sample_rate: int, hop_length: int, f_min_midi: int, bins_per_octave: int):
        assert cqt.ndim == 2, "CQT must be 2D (n_bins, n_frames)"
        self._cqt = cqt
        self._n_bins, self._n_frames = cqt.shape
        self._sample_rate = int(sample_rate)
        self._hop_length = int(hop_length)
        self._f_min_midi = int(f_min_midi)
        self._bins_per_octave = int(bins_per_octave)

        # Build initial image/pixmap
        self._rebuild_pixmap()

        # Reset any prior transform so fit_to_window starts from identity
        self.resetTransform()
        self._user_zoomed = False

        # Fit vertically to the viewport height immediately
        self.fit_to_window()
        # Ensure playhead exists/updated for new scene
        self._ensure_playhead()
        self.set_playback_position_seconds(self._playhead_seconds)

    def clear_notes(self):
        for it in self._note_items:
            self._scene.removeItem(it)
        self._note_items.clear()
        self._notes.clear()

    def has_notes(self) -> bool:
        return len(self._notes) > 0

    def export_notes_seconds(self) -> List[Tuple[int, float, float, int]]:
        """
        Returns a list of tuples: (pitch, start_seconds, end_seconds, velocity)
        """
        secs_per_frame = self._hop_length / float(self._sample_rate)
        out = []
        for n in self._notes:
            start = n.start_frame * secs_per_frame
            end = n.end_frame * secs_per_frame
            out.append((n.pitch, start, end, n.velocity))
        return out

    def set_notes_seconds(self, notes: List[Tuple[int, float, float, int]]):
        """
        Replace current notes with the provided list defined in seconds.
        Each entry is (pitch, start_seconds, end_seconds, velocity).
        """
        # Clear current note visuals and data
        self.clear_notes()
        # If no spectrogram set yet, nothing to map against
        if self._cqt is None or self._n_frames <= 0:
            return
        secs_per_frame = self._hop_length / float(self._sample_rate)
        for pitch, start_s, end_s, vel in notes:
            print(f"Processing note: pitch={pitch}, start={start_s:.2f}s, end={end_s:.2f}s, velocity={vel}")
            # Convert seconds to frame indices
            start_frame = int(np.floor(max(0.0, start_s) / secs_per_frame))
            end_frame = int(np.ceil(max(start_s, end_s) / secs_per_frame))
            if end_frame <= start_frame:
                end_frame = start_frame + 1
            # Clamp frames to scene
            start_frame = int(np.clip(start_frame, 0, max(0, self._n_frames - 1)))
            end_frame = int(np.clip(end_frame, start_frame + 1, self._n_frames))
            # Clamp pitch within available semitone range
            midi_min = int(self._f_min_midi)
            midi_max = int(np.floor(self._f_min_midi + (self._n_bins * 12.0 / float(max(1, self._bins_per_octave)))))
            p = int(np.clip(int(pitch), midi_min, midi_max))
            # Add visual item
            rect = self._frame_pitch_to_rect(start_frame, end_frame, p)
            item = QGraphicsRectItem(rect)
            item.setBrush(QColor(0, 255, 0, 80))
            item.setPen(QColor(0, 200, 0, 180))
            self._scene.addItem(item)
            self._note_items.append(item)
            self._notes.append(DrawnNote(pitch=p, start_frame=start_frame, end_frame=end_frame, velocity=int(vel)))

    # Internals
    def _to_qimage(self, cqt: np.ndarray) -> QImage:
        """
        Convert normalized CQT [0,1] to a QImage using the 'inferno' colormap
        (matching librosa.display default).
        """
        # Defensive clip
        c = np.clip(cqt, 0.0, 1.0).astype(np.float32)
        # Map to RGBA using matplotlib's inferno
        cmap = cm.get_cmap('inferno')
        rgba = cmap(c)  # shape (bins, frames, 4), floats in [0,1]
        # Drop alpha and convert to uint8
        rgb = (rgba[..., :3] * 255.0).astype(np.uint8)  # (bins, frames, 3)
        # Flip vertically so lowest bin is at the bottom
        img_arr = np.flipud(rgb)
        # Determine target pixel size
        bins, frames, _ = img_arr.shape
        target_w = max(1, int(round(frames * float(self._x_scale))))
        target_h = max(1, int(round(bins * int(self._y_scale))))

        # Horizontal resampling to target_w
        if target_w != frames:
            # Use nearest-neighbor sampling for speed
            idx = np.linspace(0, frames - 1, target_w).astype(np.int64)
            img_arr = img_arr[:, idx, :]
        # Vertical upscaling by integer repeat (y_scale is integer)
        if self._y_scale > 1:
            img_arr = np.repeat(img_arr, self._y_scale, axis=0)

        height, width, _ = img_arr.shape
        # Create QImage from RGB888 buffer
        bytes_per_line = 3 * width
        qimg = QImage(img_arr.data, width, height, bytes_per_line, QImage.Format_RGB888)
        return qimg.copy()

    def _bins_per_semitone(self) -> float:
        return 12.0 / float(max(1, self._bins_per_octave))  # bins per semitone (may be <1 if bpo>12)

    def _midi_at_bin(self, bin_index: int) -> float:
        """
        Fractional MIDI pitch corresponding to the center of the given CQT bin index
        (0 = lowest bin). Assumes linear mapping in log2 frequency: each octave
        is `bins_per_octave` bins, so 12 semitones span `bins_per_octave` bins.
        """
        return float(self._f_min_midi) + (float(bin_index) * 12.0 / float(max(1, self._bins_per_octave)))

    def _bin_range_for_midi(self, midi_pitch: int) -> Tuple[int, int]:
        """
        Return [start_bin, end_bin) span (in bin indices, low→high) that corresponds
        to the given integer MIDI semitone. This honors `bins_per_octave` even when it
        is not a multiple of 12 by using ceil boundaries, and guarantees at least 1 bin.
        """
        # Bin index boundaries for this semitone relative to f_min
        bpo = float(max(1, self._bins_per_octave))
        rel_semitones_start = float(midi_pitch - self._f_min_midi)
        rel_semitones_end = rel_semitones_start + 1.0
        start_f = rel_semitones_start * (bpo / 12.0)
        end_f = rel_semitones_end * (bpo / 12.0)
        start_bin = int(np.floor(start_f + 1e-9))
        end_bin = int(np.ceil(end_f - 1e-9))
        # Clamp and ensure at least one bin
        start_bin = int(np.clip(start_bin, 0, max(0, self._n_bins - 1)))
        end_bin = int(np.clip(max(start_bin + 1, end_bin), 1, self._n_bins))
        return start_bin, end_bin

    def _scene_pos_to_frame_pitch(self, pos: QPointF) -> Tuple[int, int]:
        # Map scene x to frame
        frame = int(np.clip((pos.x() - self._left_gutter) / self._x_scale, 0, max(0, self._n_frames - 1)))
        # Map scene y to bin; remember we flipped vertically: scene y=0 is top (highest bin), bottom is lowest.
        bin_from_top = int(np.clip(pos.y() / self._y_scale, 0, max(0, self._n_bins - 1)))
        bin_index = self._n_bins - 1 - bin_from_top
        # Convert bin to fractional MIDI and snap to nearest integer semitone
        midi_f = self._midi_at_bin(int(np.clip(bin_index, 0, max(0, self._n_bins - 1))))
        # Total MIDI range covered by the spectrogram
        midi_min = int(self._f_min_midi)
        midi_max = int(np.floor(self._f_min_midi + (self._n_bins * 12.0 / float(max(1, self._bins_per_octave)))) )
        pitch = int(np.clip(int(round(midi_f)), midi_min, midi_max))
        return frame, pitch

    def _frame_pitch_to_rect(self, start_frame: int, end_frame: int, pitch: int) -> QRectF:
        # Convert a time range and integer MIDI pitch to a scene rectangle spanning
        # the bins that fall within this semitone, honoring bins_per_octave.
        start_bin, end_bin = self._bin_range_for_midi(int(pitch))  # [start, end)
        # y from top corresponds to higher bin indices nearer the top after flip
        # Top y is at the top of the highest bin in the range, which is end_bin-1
        bin_from_top_top = self._n_bins - end_bin
        x = self._left_gutter + start_frame * self._x_scale
        y = bin_from_top_top * self._y_scale
        w = max(1.0, (end_frame - start_frame) * self._x_scale)
        h = max(1.0, (end_bin - start_bin) * self._y_scale)
        return QRectF(float(x), float(y), float(w), float(h))

    # Playback marker API -------------------------------------------------
    def _ensure_playhead(self):
        if self._cqt is None:
            return
        if self._playhead_item is None:
            self._playhead_item = QGraphicsLineItem()
            self._playhead_item.setZValue(10_000)  # above everything
            self._scene.addItem(self._playhead_item)
            pen = QPen(QColor(50, 200, 255, 220))
            pen.setWidth(2)
            self._playhead_item.setPen(pen)
        # Update its height and position
        self._update_playhead_geometry()

    def _update_playhead_geometry(self):
        if self._playhead_item is None or self._cqt is None:
            return
        # Compute X from seconds without snapping to integer frames, to preserve
        # exact alignment with the mouse under any view zoom level.
        secs_per_frame = self._hop_length / float(self._sample_rate)
        frame_f = self._playhead_seconds / max(1e-12, secs_per_frame)
        # Clamp to valid [0, n_frames]
        frame_f = float(np.clip(frame_f, 0.0, float(max(0, self._n_frames))))
        x = self._left_gutter + frame_f * self._x_scale
        # Full scene height
        height = self._n_bins * self._y_scale
        self._playhead_item.setLine(x, 0, x, height)

    def set_playback_position_seconds(self, t: float):
        """
        Update the playhead position (in seconds). Does not emit marker_moved_seconds.
        """
        self._playhead_seconds = max(0.0, float(t))
        if self._cqt is None:
            return
        self._ensure_playhead()
        self._update_playhead_geometry()

    def get_playback_position_seconds(self) -> float:
        return float(self._playhead_seconds)

    # View controls
    def fit_to_window(self):
        if self._pix_item is None:
            return
        # Adjust Y scale so the spectrogram height fits the current viewport height
        self._update_y_scale_to_fit_height()
        self._rebuild_pixmap()
        # Ensure no additional view scaling is applied; we only fit to height via pixel scaling
        self.resetTransform()
        self._user_zoomed = False
        # Playhead needs to be updated after any rebuild
        self._ensure_playhead()
        self._update_playhead_geometry()

    def set_x_scale(self, x_scale: float):
        """
        Adjust horizontal pixel-per-frame scale without recomputing the CQT.
        Rebuilds the pixmap and resizes existing note rectangles accordingly.
        """
        # Allow zooming out below 1 px/frame down to 0.05; max 128 px/frame
        x_scale = float(max(0.05, min(128.0, x_scale)))
        if x_scale == self._x_scale:
            return
        self._x_scale = x_scale
        self._rebuild_pixmap()
        # Update playhead after geometry change
        self._ensure_playhead()
        self._update_playhead_geometry()
        try:
            self.x_scale_changed.emit(float(self._x_scale))
        except Exception:
            pass

    def increase_x_scale(self, delta: int = 1):
        # Backward-compatible additive change
        self.set_x_scale(float(self._x_scale) + float(delta))

    def scale_x_by(self, factor: float):
        self.set_x_scale(float(self._x_scale) * float(factor))

    def reset_x_scale(self, value: float = 4.0):
        self.set_x_scale(float(value))

    def get_x_scale(self) -> float:
        return float(self._x_scale)

    def wheelEvent(self, event: QWheelEvent):  # type: ignore[override]
        if self._pix_item is None:
            return
        modifiers = event.modifiers() if hasattr(event, 'modifiers') else Qt.NoModifier
        if modifiers & Qt.ControlModifier:
            # Ctrl + wheel = view zoom
            delta = event.angleDelta().y() if hasattr(event, 'angleDelta') else 0
            if delta == 0:
                return
            factor = 1.25 if delta > 0 else 1.0 / 1.25
            self.scale(factor, factor)
            self._user_zoomed = True
            event.accept()
            return
        # Default wheel pans horizontally
        delta = 0
        if hasattr(event, 'angleDelta'):
            ad = event.angleDelta()
            # Prefer vertical wheel delta to translate to horizontal pan
            delta = ad.y() if ad.y() != 0 else ad.x()
        elif hasattr(event, 'pixelDelta'):
            pd = event.pixelDelta()
            delta = pd.y() if pd.y() != 0 else pd.x()
        # Invert so wheel up scrolls right (natural feel can be adjusted)
        if delta != 0 and hasattr(self, 'horizontalScrollBar'):
            sb = self.horizontalScrollBar()
            sb.setValue(sb.value() - delta)
            event.accept()
            return
        super().wheelEvent(event)

    def resizeEvent(self, event):  # type: ignore[override]
        super().resizeEvent(event)
        # If user hasn't zoomed, keep the spectrogram fitted as the window resizes
        if not self._user_zoomed:
            self.fit_to_window()

    # Mouse events for drawing notes as horizontal rectangles (single bin height)
    def mousePressEvent(self, event: QMouseEvent):
        if self._cqt is None:
            return

        # Helper: determine if we clicked on an existing note item
        def _note_hit(scene_pos: QPointF) -> tuple[int, QGraphicsRectItem] | None:
            # Prefer using scene items at position for accurate hit-test
            for it in self._scene.items(scene_pos):
                if isinstance(it, QGraphicsRectItem) and it in self._note_items:
                    return (self._note_items.index(it), it)
            # Fallback by manual check
            for idx in reversed(range(len(self._note_items))):
                it = self._note_items[idx]
                if it.contains(it.mapFromScene(scene_pos)):
                    return (idx, it)
            return None

        if event.button() == Qt.LeftButton:
            self._drawing = True
            scene_pos = self.mapToScene(event.position().toPoint())
            # If clicked on an existing note, start dragging it instead of drawing a new one
            hit = _note_hit(scene_pos)
            if hit is not None:
                idx, item = hit
                self._drawing = False
                self._dragging_note_item = item
                self._dragging_note_index = idx
                # Compute grab offset and note length in frames
                start_frame, pitch = self._scene_pos_to_frame_pitch(scene_pos)
                note = self._notes[idx]
                self._dragging_note_length = max(1, int(note.end_frame - note.start_frame))
                self._dragging_note_grab_frame_offset = max(0, int(start_frame - note.start_frame))
                return
            self._drag_start_scene = scene_pos
            start_frame, pitch = self._scene_pos_to_frame_pitch(scene_pos)
            # Create a temporary rubberband rect at least 1 pixel wide
            rect = self._frame_pitch_to_rect(start_frame, start_frame + 1, pitch)
            self._rubberband_item = QGraphicsRectItem(rect)
            self._rubberband_item.setBrush(QColor(0, 255, 0, 80))
            self._rubberband_item.setPen(QColor(0, 200, 0, 180))
            self._scene.addItem(self._rubberband_item)
        elif event.button() == Qt.MiddleButton or event.button() == Qt.RightButton:
            # Start dragging the playhead if near it; otherwise jump playhead to click and start drag
            scene_pos = self.mapToScene(event.position().toPoint())
            if event.button() == Qt.RightButton:
                # Right click on a note deletes it
                hit = _note_hit(scene_pos)
                if hit is not None:
                    idx, item = hit
                    # Remove the graphics item and data
                    try:
                        self._scene.removeItem(item)
                    except Exception:
                        pass
                    self._note_items.pop(idx)
                    self._notes.pop(idx)
                    return
            if self._playhead_item is not None:
                x_line = self._playhead_item.line().x1()
                if abs(scene_pos.x() - x_line) <= 6:
                    self._dragging_playhead = True
                else:
                    self._dragging_playhead = True
                    self._set_playhead_by_scene_pos(scene_pos)
                    self.marker_moved_seconds.emit(self._playhead_seconds)
            else:
                self._dragging_playhead = True
                self._set_playhead_by_scene_pos(scene_pos)
                self.marker_moved_seconds.emit(self._playhead_seconds)
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._drawing and self._rubberband_item is not None and self._drag_start_scene is not None:
            current_scene = self.mapToScene(event.position().toPoint())
            start_frame, pitch = self._scene_pos_to_frame_pitch(self._drag_start_scene)
            end_frame, _ = self._scene_pos_to_frame_pitch(current_scene)
            if end_frame <= start_frame:
                end_frame = start_frame + 1
            rect = self._frame_pitch_to_rect(start_frame, end_frame, pitch)
            self._rubberband_item.setRect(rect)
        elif self._dragging_note_item is not None and self._dragging_note_index is not None:
            # Move existing note with the cursor, snapping to frame/pitch grid
            current_scene = self.mapToScene(event.position().toPoint())
            frame_at_cursor, pitch_at_cursor = self._scene_pos_to_frame_pitch(current_scene)
            start_frame = int(frame_at_cursor - self._dragging_note_grab_frame_offset)
            # Clamp
            start_frame = max(0, min(start_frame, max(0, self._n_frames - self._dragging_note_length)))
            end_frame = int(start_frame + self._dragging_note_length)
            pitch = int(pitch_at_cursor)
            # Clamp pitch into valid range
            if self._cqt is not None:
                min_pitch = int(self._f_min_midi)
                max_pitch = int(self._f_min_midi + self._n_bins - 1)
                pitch = max(min_pitch, min(max_pitch, pitch))
            rect = self._frame_pitch_to_rect(start_frame, end_frame, pitch)
            self._dragging_note_item.setRect(rect)
        elif self._dragging_playhead:
            current_scene = self.mapToScene(event.position().toPoint())
            self._set_playhead_by_scene_pos(current_scene)
            self.marker_moved_seconds.emit(self._playhead_seconds)
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if self._drawing and event.button() == Qt.LeftButton and self._rubberband_item is not None and self._drag_start_scene is not None:
            self._drawing = False
            end_scene = self.mapToScene(event.position().toPoint())
            start_frame, pitch = self._scene_pos_to_frame_pitch(self._drag_start_scene)
            end_frame, _ = self._scene_pos_to_frame_pitch(end_scene)
            if end_frame <= start_frame:
                end_frame = start_frame + 1
            # Finalize note
            self._note_items.append(self._rubberband_item)
            self._notes.append(DrawnNote(pitch=pitch, start_frame=start_frame, end_frame=end_frame, velocity=80))
            try:
                self.note_created.emit(pitch, start_frame, end_frame, 80)
            except Exception:
                pass
            self._rubberband_item = None
            self._drag_start_scene = None
        elif event.button() == Qt.LeftButton and self._dragging_note_item is not None and self._dragging_note_index is not None:
            # Commit dragged note position into data
            item = self._dragging_note_item
            idx = self._dragging_note_index
            r: QRectF = item.rect()
            # Convert rect back to start/end frame and pitch using inverse of _frame_pitch_to_rect
            # Horizontal mapping: x in scene minus left gutter, divided by x_scale
            start_frame = int(round(max(0.0, (r.left() - self._left_gutter) / float(self._x_scale))))
            end_frame = int(round(max(0.0, (r.right() - self._left_gutter) / float(self._x_scale))))
            if end_frame <= start_frame:
                end_frame = start_frame + 1
            # Vertical mapping: map y to bin, then to midi
            if self._cqt is not None and self._n_bins > 0:
                bin_index = int(round(r.top() / float(self._y_scale)))
                bin_index = max(0, min(self._n_bins - 1, bin_index))
                pitch = int(self._midi_at_bin(bin_index))
            else:
                pitch = self._notes[idx].pitch
            # Update model
            self._notes[idx] = DrawnNote(pitch=pitch, start_frame=start_frame, end_frame=end_frame, velocity=self._notes[idx].velocity)
            # Clear dragging state
            self._dragging_note_item = None
            self._dragging_note_index = None
            self._dragging_note_grab_frame_offset = 0
            self._dragging_note_length = 0
        elif self._dragging_playhead and (event.button() in (Qt.MiddleButton, Qt.RightButton)):
            self._dragging_playhead = False
        else:
            super().mouseReleaseEvent(event)

    # Undo last created note
    def undo_last_note(self):
        if not self._note_items or not self._notes:
            return
        last_item = self._note_items.pop()
        try:
            self._scene.removeItem(last_item)
        except Exception:
            pass
        self._notes.pop()

    def _set_playhead_by_scene_pos(self, scene_pos: QPointF):
        # Convert scene x to seconds, clamp to available frames.
        # Use floating-point frame precision to avoid zoom-dependent rounding errors.
        denom = float(max(1e-12, float(self._x_scale)))
        frame_f = (scene_pos.x() - float(self._left_gutter)) / denom
        # Clamp
        frame_f = float(np.clip(frame_f, 0.0, float(max(0, self._n_frames))))
        secs_per_frame = self._hop_length / float(self._sample_rate)
        self._playhead_seconds = frame_f * secs_per_frame
        self._ensure_playhead()
        self._update_playhead_geometry()

    # Axes rendering (time on x-axis in seconds, MIDI notes on y-axis)
    def drawForeground(self, painter: QPainter, rect: QRectF):  # type: ignore[override]
        super().drawForeground(painter, rect)
        if self._cqt is None or self._pix_item is None:
            return
        scene_rect = self._scene.sceneRect()
        if scene_rect.isEmpty():
            return

        # Styling
        pen = QPen(QColor(240, 240, 240, 200))
        minor_pen = QPen(QColor(200, 200, 200, 120))
        painter.setPen(pen)
        font = painter.font()
        font.setPointSize(10)
        painter.setFont(font)

        # Compute time ticks along x
        secs_per_frame = self._hop_length / float(self._sample_rate)
        total_secs = self._n_frames * secs_per_frame
        # Requirement: place time markers every 1 second
        step = 1.0

        # Baseline at bottom
        bottom_y = scene_rect.bottom()
        # Limit time axis to span under the spectrogram area (to the right of gutter)
        spec_left = float(self._left_gutter)
        spec_right = float(self._left_gutter + self._n_frames * self._x_scale)
        painter.drawLine(spec_left, bottom_y, spec_right, bottom_y)

        # Draw ticks and labels
        tick_height = 8
        # Determine visible x-range to avoid drawing offscreen ticks for very long audio
        view_rect = self.mapToScene(self.viewport().rect()).boundingRect()
        vis_left = max(self._left_gutter, view_rect.left())
        vis_right = min(self._left_gutter + self._n_frames * self._x_scale, view_rect.right())
        # Convert to seconds range
        start_frame_vis = max(0.0, (vis_left - self._left_gutter) / max(1e-9, self._x_scale))
        end_frame_vis = max(0.0, (vis_right - self._left_gutter) / max(1e-9, self._x_scale))
        start_sec = max(0.0, start_frame_vis * secs_per_frame)
        end_sec = min(total_secs, end_frame_vis * secs_per_frame)
        # Snap to whole seconds
        t = float(int(start_sec))
        if t < start_sec:
            t += 1.0
        while t <= end_sec + 1e-6:
            frame = t / secs_per_frame
            x = self._left_gutter + frame * self._x_scale
            # Major tick
            painter.drawLine(x, bottom_y, x, bottom_y - tick_height)
            # Label
            label = self._format_time_label(t)
            painter.drawText(QPointF(x + 2, bottom_y - tick_height - 2), label)
            t += step

        # Y-axis labels: keep them pinned to the viewport's left edge so they never scroll off-screen
        # Compute label y-positions from scene, then render in view coordinates
        painter.save()
        painter.resetTransform()  # switch to viewport/widget coordinates
        view_width = self.viewport().width()
        view_height = self.viewport().height()
        axis_x = 0  # left edge of the viewport

        # Clear the left label strip to prevent any stale drawings when panning/scrolling.
        # Use at least the gutter width as a persistent label area.
        try:
            bg_color = self.viewport().palette().window().color()
            strip_width = max(48, self._left_gutter)
            painter.fillRect(0, 0, strip_width, view_height, bg_color)
        except Exception:
            # Fallback: do nothing if palette is unavailable in some contexts
            pass

        # Draw a simple vertical axis line on the left of the viewport
        painter.setPen(pen)
        painter.drawLine(axis_x, 0, axis_x, view_height)

        # Label every octave (12 semitones) honoring bins_per_octave range
        midi_min = int(self._f_min_midi)
        midi_max = int(np.floor(self._f_min_midi + (self._n_bins * 12.0 / float(max(1, self._bins_per_octave)))) )
        for midi_pitch in range(midi_min, midi_max + 1):
            # Scene y-position for this pitch (baseline at bin bottom)
            rect_for_pitch = self._frame_pitch_to_rect(0, 1, midi_pitch)
            scene_y = rect_for_pitch.y() + rect_for_pitch.height()
            # Map to view coordinates to keep labels visible regardless of pan/zoom
            view_pt = self.mapFromScene(QPointF(0, scene_y))
            vy = float(view_pt.y())
            if -20 <= vy <= view_height + 20:  # cull if far off-screen
                painter.setPen(minor_pen)
                painter.drawLine(axis_x, vy, axis_x + 10, vy)
                if midi_pitch % 12 == 0:
                    painter.setPen(pen)
                    painter.drawText(QPointF(axis_x + 12, vy - 2), (self._midi_to_name(midi_pitch)))
        painter.restore()

    # Helper methods to rebuild image/pixmap with current scales
    def _rebuild_pixmap(self):
        if self._cqt is None:
            return
        img = self._to_qimage(self._cqt)
        pix = QPixmap.fromImage(img)
        # Preserve existing notes
        notes_copy = list(self._notes)
        # Clear scene and add pixmap
        self._scene.clear()
        self._note_items.clear()
        self._pix_item = self._scene.addPixmap(pix)
        # Place the spectrogram image to the right of the left gutter
        self._pix_item.setOffset(self._left_gutter, 0)
        self._pix_item.setScale(1.0)
        # Scene rect spans the left gutter plus the image area
        self._scene.setSceneRect(0, 0, self._left_gutter + pix.width(), pix.height())
        # Re-add note rectangles for preserved notes
        self._notes = notes_copy
        for n in self._notes:
            rect = self._frame_pitch_to_rect(n.start_frame, n.end_frame, n.pitch)
            item = QGraphicsRectItem(rect)
            item.setBrush(QColor(0, 255, 0, 80))
            item.setPen(QColor(0, 200, 0, 180))
            self._scene.addItem(item)
            self._note_items.append(item)
        # Re-add playhead if needed
        if self._playhead_item is not None:
            self._playhead_item = None
        self._ensure_playhead()
        self._update_playhead_geometry()

    def _update_y_scale_to_fit_height(self):
        if self._cqt is None or self._n_bins <= 0:
            return
        # Determine target pixel height to fill viewport without scrollbars (reserve ~20px for axes/labels)
        avail = max(1, self.viewport().height() - 24)
        new_scale = max(1, int(round(avail / float(self._n_bins))))
        if new_scale != self._y_scale:
            self._y_scale = new_scale

    @staticmethod
    def _midi_to_name(m: int) -> str:
        names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        octave = (m // 12) - 1
        return f"{names[m % 12]}{octave} ({m})"

    @staticmethod
    def _format_time_label(t: float) -> str:
        if t < 60:
            return f"{t:0.2f}s"
        m = int(t // 60)
        s = t - 60 * m
        return f"{m}:{s:04.1f}"
