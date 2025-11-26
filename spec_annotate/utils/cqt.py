import librosa
import numpy as np

def generate_spectrogram(
    waveform,
    hop_length: int = 128,
    sample_rate: int = 22050,
    n_bins: int = 240,
    bins_per_octave: int = 36,
    f_min: float = librosa.note_to_hz("C2"),
    power_scaling: float | None = None,
):
    """
    Generate a normalized CQT spectrogram in the range [0, 1].

    Parameters
    - waveform: array-like. 1D mono or multi-channel time series. Supports numpy arrays or lists.
    - hop_length: hop length for CQT in samples.
    - sample_rate: sampling rate of the audio.
    - n_bins: total number of CQT bins (rows). If f_min corresponds to MIDI 0 and bins_per_octave=12,
      bins map directly to MIDI pitches [0, n_bins-1].
    - bins_per_octave: bins per octave.
    - f_min: minimum frequency for the lowest CQT bin.
    - power_scaling: optional gamma to apply as cqt ** gamma for display contrast.

    Returns
    - 2D numpy array of shape (n_bins, n_frames), normalized to [0, 1].
    """

    # Ensure numpy float32 array, convert to mono if needed
    x = np.asarray(waveform, dtype=np.float32)
    if x.ndim == 2:
        # assume shape (channels, samples) or (samples, channels)
        if x.shape[0] < x.shape[1]:
            # likely (channels, samples)
            mono = x.mean(axis=0)
        else:
            # likely (samples, channels)
            mono = x.mean(axis=1)
    elif x.ndim == 1:
        mono = x
    else:
        raise ValueError("waveform must be 1D or 2D array-like")

    cqt = librosa.hybrid_cqt(
        mono,
        hop_length=hop_length,
        sr=sample_rate,
        n_bins=n_bins,
        bins_per_octave=bins_per_octave,
        fmin=f_min,
    )
    cqt = np.abs(cqt)

    # Log amplitude then min-max normalize to [0, 1]
    cqt_db = librosa.amplitude_to_db(cqt, ref=np.max)
    cqt_min = cqt_db.min()
    cqt_max = cqt_db.max()
    if cqt_max == cqt_min:
        norm = np.zeros_like(cqt_db, dtype=np.float32)
    else:
        norm = (cqt_db - cqt_min) / (cqt_max - cqt_min)

    if power_scaling is not None:
        norm = norm ** float(power_scaling)

    return norm.astype(np.float32)
