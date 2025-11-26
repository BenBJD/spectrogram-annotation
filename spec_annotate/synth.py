from __future__ import annotations

from typing import Dict, Optional
import numpy as np
import sounddevice as sd
from PySide6.QtCore import QObject


class SynthEngine(QObject):
    """
    Polyphonic sine-wave synth for auditioning notes, optimized for low latency
    using python-sounddevice and NumPy vectorization.

    - Mono, 16-bit PCM, callback-based rendering.

    Usage:
        synth = SynthEngine(parent)
        vid = synth.note_on(freq_hz=440.0, velocity=64)
        # ... later ...
        synth.note_off(vid)
        # Remember to call synth.stop() when done
    """

    def __init__(self, parent=None, sample_rate: int = 44100,
        max_voices: int = 32):
        super().__init__(parent)

        # Configuration for SoundDevice
        self.sample_rate = int(sample_rate)
        self.max_voices = int(max_voices)
        self.channels = 1
        self.dtype = np.int16  # Output data type for sounddevice stream

        # Critical for low latency: A small blocksize (e.g., 128 to 512 frames)
        # 256 frames at 44.1kHz is ~5.8ms latency, 512 is ~11.6ms
        self.blocksize = 512

        # Voice state
        self._next_voice_id = 1
        self._voices: Dict[int, dict] = {}
        self.master_gain = 0.2  # 0..1

        # Envelope params (seconds)
        self.attack_sec = 0.01
        self.release_sec = 0.03

        # SoundDevice Stream Setup (non-blocking, callback-based)
        # The callback function self._render_chunk will be called whenever the
        # audio buffer needs data, providing the lowest possible latency.
        self._stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=self.dtype,
            blocksize=self.blocksize,
            callback=self._render_chunk,
            finished_callback=self._stream_finished
        )
        self._stream.start()

    def _stream_finished(self):
        """Called by sounddevice when the stream stops (e.g., in self.stop())."""
        print("SynthEngine audio stream stopped.")

    # Public API -----------------------------------------------------------
    def note_on(self, freq_hz: float, velocity: int) -> Optional[int]:
        if len(self._voices) >= self.max_voices:
            return None
        vid = self._next_voice_id
        self._next_voice_id += 1
        # Map MIDI velocity (0..127) to gain (0..1) with mild curve
        v = max(0, min(127, int(velocity))) / 127.0
        gain = v ** 0.8

        # Pre-calculate envelope increments (per sample)
        attack_samples = max(1, int(self.attack_sec * self.sample_rate))
        release_samples = max(1, int(self.release_sec * self.sample_rate))
        attack_inc = 1.0 / float(attack_samples)
        release_inc = 1.0 / float(release_samples)

        voice = {
            'freq': float(freq_hz),
            'phase': 0.0,  # radians
            'gain': float(gain),
            'state': 'attack',  # attack -> sustain -> release -> done
            'env': 0.0,  # Current envelope level (0..1)
            'attack_inc': attack_inc,
            'release_inc': release_inc,
        }
        self._voices[vid] = voice
        return vid

    def note_off(self, voice_id: int) -> None:
        v = self._voices.get(int(voice_id))
        if v is not None and v['state'] != 'release':
            v['state'] = 'release'

    def set_voice_freq(self, voice_id: int, freq_hz: float) -> None:
        """
        Retune an active voice to a new frequency.
        """
        v = self._voices.get(int(voice_id))
        if v is not None:
            try:
                v['freq'] = float(freq_hz)
            except Exception:
                pass

    def all_notes_off(self) -> None:
        for v in self._voices.values():
            v['state'] = 'release'

    def stop(self) -> None:
        """Stops the sound device stream and clears voices."""
        try:
            # Stop and close the sounddevice stream
            if self._stream.running:
                self._stream.stop()
            self._stream.close()
        except Exception as e:
            # Handle potential PortAudio errors during close
            print(f"Error stopping stream: {e}")
        finally:
            self._voices.clear()

    # Rendering ------------------------------------------------------------
    def _render_chunk(self, outdata: np.ndarray, frames: int, time, status):
        """
        SoundDevice callback function. Fills the 'outdata' buffer directly.

        :param outdata: The NumPy array to fill with PCM data.
        :param frames: The number of frames (samples) to render (equal to self.blocksize).
        :param status: Stream status flags (e.g., 'underrun' warning).
        """
        try:
            if status:
                # Log any potential buffer underruns or other warnings
                print(f"Audio stream status warning: {status}")

            n_frames = frames

            # Internal audio generation buffer (float32, full scale -1.0 to 1.0)
            out = np.zeros(n_frames, dtype=np.float32)
            two_pi_over_sr = 2.0 * np.pi / float(self.sample_rate)

            dead_voices = []

            # --- Voice Processing Loop (Optimized for speed) ---
            for vid, v in self._voices.items():
                f = v['freq']
                # Phase increment per sample
                w = two_pi_over_sr * f

                # --- Envelope Handling (Linear step per sample, state machine) ---

                # Get current state variables
                phase = v['phase']
                env = v['env']
                gain = v['gain']
                state = v['state']

                if state == 'done':
                    # Already marked for removal, skip rendering
                    dead_voices.append(vid)
                    continue

                # Pre-calculate envelope increments
                env_inc = 0.0
                if state == 'attack':
                    env_inc = v['attack_inc']
                elif state == 'release':
                    env_inc = -v['release_inc']
                # sustain: env_inc remains 0.0

                # --- Vectorized Sine Wave Generation ---
                # 1. Create a phase vector for the whole chunk
                # np.arange(n_frames) gives [0, 1, 2, ..., n_frames-1]
                phase_vector = phase + w * np.arange(n_frames)

                # 2. Generate the sine wave
                sine_wave = np.sin(phase_vector)

                # --- Apply Envelope, State Change, and Accumulate ---

                # In a real-time system, we must ensure the envelope transition
                # happens correctly, even if it spans across the current chunk.

                # Simplified check: calculate the envelope after n_frames
                env_end = env + env_inc * n_frames

                if env_inc > 0.0:  # Attack
                    if env_end >= 1.0:
                        # Envelope maxed out mid-chunk
                        # Calculate the sample index where env hits 1.0
                        transition_idx = int((1.0 - env) / env_inc)

                        # Apply linear attack gain until transition_idx
                        chunk_env_attack = env + env_inc * np.arange(
                            transition_idx)
                        out[:transition_idx] += sine_wave[:transition_idx] * (
                                self.master_gain * gain * chunk_env_attack)

                        # Apply full sustain gain for the rest
                        out[transition_idx:] += sine_wave[transition_idx:] * (
                                self.master_gain * gain * 1.0)

                        v['env'] = 1.0
                        v['state'] = 'sustain'
                    else:
                        # Full attack phase across the chunk
                        chunk_env_attack = env + env_inc * np.arange(n_frames)
                        out += sine_wave * (
                                self.master_gain * gain * chunk_env_attack)
                        v['env'] = env_end

                elif env_inc < 0.0:  # Release
                    if env_end <= 0.0:
                        # Envelope finished mid-chunk
                        transition_idx = int((env) / abs(env_inc))

                        # Apply linear release gain until transition_idx
                        chunk_env_release = env + env_inc * np.arange(
                            transition_idx)
                        out[:transition_idx] += sine_wave[:transition_idx] * (
                                self.master_gain * gain * chunk_env_release)

                        # The rest of the chunk is silence (added 0 above)
                        v['env'] = 0.0
                        v['state'] = 'done'
                        dead_voices.append(vid)
                    else:
                        # Full release phase across the chunk
                        chunk_env_release = env + env_inc * np.arange(n_frames)
                        out += sine_wave * (
                                self.master_gain * gain * chunk_env_release)
                        v['env'] = env_end

                else:  # Sustain (env_inc == 0.0)
                    out += sine_wave * (self.master_gain * gain * env)
                    # State and env are unchanged

                # 4. Update Final Phase for next chunk
                # Keep phase normalized
                v['phase'] = np.mod(phase_vector[-1], 2 * np.pi)

            # --- Cleanup and Write to Output ---

            # Remove finished voices
            for vid in dead_voices:
                self._voices.pop(vid, None)

            # Mixdown, clip to float range [-1.0, 1.0]
            pcm_float = np.clip(out, -1.0, 1.0)

            # Convert to target dtype (np.int16) and fill outdata
            # Max value for 16-bit PCM is 32767
            pcm_i16 = (pcm_float * 32767.0).astype(self.dtype)

            # SoundDevice expects a 2D array (N, channels) for mono it's (N, 1)
            outdata[:] = pcm_i16.reshape(-1, self.channels)

        except Exception as e:
            # Crucial: print errors but ensure the callback finishes quickly
            # to avoid stream dropouts.
            print(f"Error in audio rendering callback: {e}")
            # Fill buffer with silence in case of error
            outdata.fill(0)