from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QSpinBox,
    QDoubleSpinBox,
    QLineEdit,
    QMessageBox,
)
import librosa


class CQTSettingsDialog(QDialog):
    def __init__(
        self,
        *,
        hop_length: int,
        n_bins: int,
        bins_per_octave: int,
        f_min_note: str,
        power_scaling: float | None,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("CQT Settings")

        form = QFormLayout(self)

        self.hop_sb = QSpinBox(self)
        self.hop_sb.setRange(16, 8192)
        self.hop_sb.setSingleStep(16)
        self.hop_sb.setValue(int(hop_length))
        form.addRow("Hop length (samples)", self.hop_sb)

        self.nbins_sb = QSpinBox(self)
        self.nbins_sb.setRange(12, 240)
        self.nbins_sb.setSingleStep(12)
        self.nbins_sb.setValue(int(n_bins))
        form.addRow("Total bins (rows)", self.nbins_sb)

        self.bpo_sb = QSpinBox(self)
        self.bpo_sb.setRange(6, 48)
        self.bpo_sb.setSingleStep(1)
        self.bpo_sb.setValue(int(bins_per_octave))
        form.addRow("Bins per octave", self.bpo_sb)

        # Minimum note as a note name (e.g., C0, A4, F#3)
        self.fmin_le = QLineEdit(self)
        self.fmin_le.setPlaceholderText("e.g., C0, A4, F#3, Db4")
        self.fmin_le.setText(str(f_min_note))
        form.addRow("Lowest note (f_min)", self.fmin_le)

        self.power_sb = QDoubleSpinBox(self)
        self.power_sb.setRange(0.01, 5.0)
        self.power_sb.setDecimals(2)
        self.power_sb.setSingleStep(0.05)
        # Allow None by using 0 to indicate disabled
        self.power_sb.setSpecialValueText("Disabled")
        self.power_sb.setMinimum(0.0)
        self.power_sb.setValue(float(power_scaling) if power_scaling is not None else 0.0)
        form.addRow("Power scaling (γ)", self.power_sb)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def values(self) -> dict:
        power_val = self.power_sb.value()
        return {
            "hop_length": int(self.hop_sb.value()),
            "n_bins": int(self.nbins_sb.value()),
            "bins_per_octave": int(self.bpo_sb.value()),
            "f_min_note": self.fmin_le.text().strip() or "C0",
            "power_scaling": None if power_val == 0.0 else float(power_val),
        }

    # Validate note name before accepting
    def accept(self) -> None:  # type: ignore[override]
        note = self.fmin_le.text().strip() or "C0"
        try:
            # Validate conversion; this raises for bad input
            _ = librosa.note_to_midi(note)
        except Exception:
            QMessageBox.warning(self, "Invalid note", f"'{note}' is not a valid note name. Try formats like C0, A4, F#3, or Db4.")
            self.fmin_le.setFocus()
            return
        super().accept()
