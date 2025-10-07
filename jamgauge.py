#!/usr/bin/env python3
"""
JamGauge – Enhanced Local Audio Feedback Tool
Improvements: Better error handling, progress feedback, additional metrics
"""
from __future__ import annotations
import argparse
import json
import math
import os
import sys
import warnings
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import librosa
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt

# Suppress librosa warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Optional loudness
try:
    import pyloudnorm as pyln
    HAVE_LOUDNESS = True
except Exception:
    HAVE_LOUDNESS = False

# Krumhansl & Kessler key profiles
KRUMHANSL_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KRUMHANSL_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
NOTE_NAMES = np.array(["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"])


def estimate_key_from_chroma(chroma: np.ndarray) -> Tuple[str, float]:
    """Return (key_string, confidence [0..1]) using Krumhansl template correlation."""
    chroma_mean = chroma.mean(axis=1)
    if np.allclose(chroma_mean.sum(), 0):
        return ("Unknown", 0.0)
    chroma_norm = chroma_mean / (np.linalg.norm(chroma_mean) + 1e-8)
    
    maj_scores = []
    min_scores = []
    for i in range(12):
        maj_template = np.roll(KRUMHANSL_MAJOR, i)
        min_template = np.roll(KRUMHANSL_MINOR, i)
        maj_scores.append(np.dot(chroma_norm, maj_template / np.linalg.norm(maj_template)))
        min_scores.append(np.dot(chroma_norm, min_template / np.linalg.norm(min_template)))
    
    maj_idx = int(np.argmax(maj_scores))
    min_idx = int(np.argmax(min_scores))
    maj_score = float(np.max(maj_scores))
    min_score = float(np.max(min_scores))
    
    if maj_score >= min_score:
        key = f"{NOTE_NAMES[maj_idx]} major"
        conf = maj_score
    else:
        key = f"{NOTE_NAMES[min_idx]} minor"
        conf = min_score
    
    # Normalize confidence
    conf = float(np.clip((conf + 1) / 2, 0, 1))
    return key, conf


def swing_ratio_from_beats(beat_times: np.ndarray, sr: int, onset_env: np.ndarray, 
                          hop_length: int, tempo: float) -> float:
    """Estimate swing ratio from beat timing deviations."""
    if len(beat_times) < 3 or tempo <= 0:
        return float("nan")
    
    sr_hz = sr / hop_length
    oenv_times = np.arange(len(onset_env)) / sr_hz
    energies_first = []
    energies_second = []
    
    for i in range(len(beat_times) - 1):
        b0, b1 = beat_times[i], beat_times[i + 1]
        dur = b1 - b0
        t1 = b0 + dur * (1/3)
        t2 = b0 + dur * (2/3)
        e1 = np.interp(t1, oenv_times, onset_env)
        e2 = np.interp(t2, oenv_times, onset_env)
        energies_first.append(max(e1, 1e-9))
        energies_second.append(max(e2, 1e-9))
    
    if not energies_first:
        return float("nan")
    
    ratio = float(np.median(np.array(energies_first) / np.array(energies_second)))
    return float(np.clip(ratio, 1.0, 2.0))


def spectral_bands(mel_s: np.ndarray, sr: int, n_mels: int) -> Dict[str, float]:
    """Return energy share (%) for low/mid/high mel bins."""
    total = mel_s.sum() + 1e-9
    thirds = n_mels // 3
    low = mel_s[:thirds].sum() / total
    mid = mel_s[thirds:2*thirds].sum() / total
    high = mel_s[2*thirds:].sum() / total
    return {
        "low_pct": float(low * 100),
        "mid_pct": float(mid * 100),
        "high_pct": float(high * 100)
    }


def crest_factor(x: np.ndarray) -> float:
    """Calculate crest factor (peak to RMS ratio) in dB."""
    rms = np.sqrt(np.mean(np.square(x)) + 1e-12)
    peak = np.max(np.abs(x)) + 1e-12
    return float(20 * np.log10(peak / (rms + 1e-12)))


def check_audio_validity(y: np.ndarray, sr: int) -> Tuple[bool, str]:
    """Validate audio is not silent or corrupted."""
    if len(y) == 0:
        return False, "Audio file is empty"
    
    rms = np.sqrt(np.mean(y**2))
    if rms < 1e-6:
        return False, "Audio appears to be silent"
    
    if np.isnan(y).any() or np.isinf(y).any():
        return False, "Audio contains invalid values (NaN or Inf)"
    
    min_duration = 1.0  # seconds
    if len(y) / sr < min_duration:
        return False, f"Audio too short (< {min_duration}s)"
    
    return True, "OK"


@dataclass
class JamGaugeReport:
    file: str
    duration_sec: float
    samplerate: int
    channels: int
    tempo_bpm: float
    beat_count: int
    swing_ratio: float
    key: str
    key_confidence: float
    rms_db: float
    crest_db: float
    loudness_lufs: float | None
    spectral_centroid_hz_mean: float
    spectral_centroid_hz_std: float
    spectral_balance: Dict[str, float]
    harmonic_percussive_ratio: float
    section_times_sec: List[float]
    section_labels: List[str]
    # New fields
    tempo_confidence: float
    zero_crossing_rate_mean: float


def analyze(path: str, sr: int = 22050, start: float = 0.0, end: float = -1.0,
           plots: bool = True, n_sections: int = 8, verbose: bool = True) -> JamGaugeReport:
    """Analyze audio file and generate comprehensive report."""
    
    if verbose:
        print(f"Loading audio: {os.path.basename(path)}...", file=sys.stderr)
    
    try:
        y, orig_sr = librosa.load(
            path, sr=sr, mono=True,
            offset=start if start else 0.0,
            duration=None if end < 0 else max(0.0, end - max(0.0, start))
        )
    except Exception as e:
        raise ValueError(f"Failed to load audio file: {e}")
    
    # Validate audio
    valid, msg = check_audio_validity(y, sr)
    if not valid:
        raise ValueError(f"Invalid audio: {msg}")
    
    duration_sec = float(len(y) / sr)
    if verbose:
        print(f"Duration: {duration_sec:.2f}s", file=sys.stderr)
    
    # Basic stats
    rms = float(20 * np.log10(np.sqrt(np.mean(y**2) + 1e-12)))
    crest = crest_factor(y)
    
    # Loudness (optional)
    loudness = None
    if HAVE_LOUDNESS:
        if verbose:
            print("Computing integrated loudness...", file=sys.stderr)
        try:
            meter = pyln.Meter(sr)
            loudness = float(meter.integrated_loudness(y.astype(np.float32)))
        except Exception as e:
            if verbose:
                print(f"Warning: Loudness computation failed: {e}", file=sys.stderr)
    
    # Onset envelope & tempo
    if verbose:
        print("Detecting tempo and beats...", file=sys.stderr)
    hop_length = 512
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, hop_length=hop_length)
    beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=hop_length)
    
    # Estimate tempo confidence from beat consistency
    if len(beat_times) > 2:
        beat_intervals = np.diff(beat_times)
        tempo_conf = float(1.0 - (np.std(beat_intervals) / (np.mean(beat_intervals) + 1e-6)))
        tempo_conf = np.clip(tempo_conf, 0, 1)
    else:
        tempo_conf = 0.0
    
    # Swing
    swing = swing_ratio_from_beats(beat_times, sr, onset_env, hop_length, tempo)
    
    # Spectral features
    if verbose:
        print("Analyzing spectral features...", file=sys.stderr)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    centroid_mean = float(np.mean(centroid))
    centroid_std = float(np.std(centroid))
    
    # Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    zcr_mean = float(np.mean(zcr))
    
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=96, hop_length=hop_length)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    balance = spectral_bands(mel, sr, mel.shape[0])
    
    # HPSS
    if verbose:
        print("Separating harmonic and percussive components...", file=sys.stderr)
    harm, perc = librosa.effects.hpss(y)
    hpr = float((np.sum(np.abs(harm)) + 1e-9) / (np.sum(np.abs(perc)) + 1e-9))
    
    # Key via CQT chroma
    if verbose:
        print("Estimating key...", file=sys.stderr)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    key, key_conf = estimate_key_from_chroma(chroma)
    
    # Sections via novelty + agglomerative segmentation
    if verbose:
        print(f"Detecting ~{n_sections} structural sections...", file=sys.stderr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    feat = librosa.util.normalize(np.vstack([mfcc, chroma]))
    R = librosa.segment.recurrence_matrix(feat, width=3, mode='affinity', sym=True)
    path_enhance = librosa.segment.path_enhance(R, 5)
    seg = librosa.segment.agglomerative(path_enhance, k=max(2, int(n_sections)))
    bound_frames = librosa.segment.boundaries(seg)
    bound_times = librosa.frames_to_time(bound_frames, sr=sr)
    section_times = [float(t) for t in bound_times]
    labels = [f"S{i+1}" for i in range(len(section_times))]
    
    # Generate outputs
    base = os.path.splitext(path)[0]
    
    # Plots
    if plots:
        if verbose:
            print("Generating visualizations...", file=sys.stderr)
        
        # Waveform
        plt.figure(figsize=(12, 3))
        librosa.display.waveshow(y, sr=sr)
        for t in section_times:
            plt.axvline(t, linestyle='--', alpha=0.3, color='red')
        plt.title(f"Waveform – {os.path.basename(path)}")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.tight_layout()
        plt.savefig(base + "_jmg_waveform.png", dpi=150)
        plt.close()
        
        # Spectrogram
        plt.figure(figsize=(12, 4))
        librosa.display.specshow(mel_db, x_axis='time', sr=sr, hop_length=hop_length, y_axis='mel')
        for t in section_times:
            plt.axvline(t, linestyle='--', alpha=0.5, color='white')
        plt.colorbar(format='%+2.0f dB')
        plt.title("Log-Mel Spectrogram")
        plt.tight_layout()
        plt.savefig(base + "_jmg_spectrogram.png", dpi=150)
        plt.close()
    
    # CSVs
    if verbose:
        print("Writing CSV files...", file=sys.stderr)
    np.savetxt(base + "_jmg_beats.csv", beat_times, fmt='%.3f',
               header='beat_time_seconds', comments='')
    with open(base + "_jmg_sections.csv", 'w') as f:
        f.write('section_index,section_time_sec,label\n')
        for i, t in enumerate(section_times):
            f.write(f"{i},{t:.3f},{labels[i]}\n")
    
    report = JamGaugeReport(
        file=os.path.basename(path),
        duration_sec=duration_sec,
        samplerate=sr,
        channels=1,
        tempo_bpm=float(tempo),
        beat_count=int(len(beat_times)),
        swing_ratio=float(swing) if not math.isnan(swing) else float('nan'),
        key=key,
        key_confidence=float(key_conf),
        rms_db=float(rms),
        crest_db=float(crest),
        loudness_lufs=float(loudness) if loudness is not None else None,
        spectral_centroid_hz_mean=centroid_mean,
        spectral_centroid_hz_std=centroid_std,
        spectral_balance=balance,
        harmonic_percussive_ratio=float(hpr),
        section_times_sec=section_times,
        section_labels=labels,
        tempo_confidence=float(tempo_conf),
        zero_crossing_rate_mean=float(zcr_mean),
    )
    
    # Write JSON report
    if verbose:
        print("Writing reports...", file=sys.stderr)
    with open(base + "_jmg_report.json", 'w') as f:
        json.dump(asdict(report), f, indent=2)
    
    # Write text report
    with open(base + "_jmg_report.txt", 'w') as f:
        def line(k, v):
            f.write(f"{k}: {v}\n")
        
        f.write("=" * 60 + "\n")
        f.write("JamGauge Audio Analysis Report\n")
        f.write("=" * 60 + "\n\n")
        
        line("File", report.file)
        line("Duration (s)", f"{report.duration_sec:.2f}")
        line("Sample rate", report.samplerate)
        f.write("\n--- Timing & Groove ---\n")
        line("Tempo (BPM)", f"{report.tempo_bpm:.1f}")
        line("Tempo confidence", f"{report.tempo_confidence:.2f}")
        line("Beats detected", report.beat_count)
        line("Swing ratio", f"{report.swing_ratio:.2f} (1=straight, 2=shuffle)")
        f.write("\n--- Harmony ---\n")
        line("Key (estimated)", f"{report.key}")
        line("Key confidence", f"{report.key_confidence:.2f}")
        f.write("\n--- Dynamics & Loudness ---\n")
        line("RMS dBFS", f"{report.rms_db:.1f}")
        if report.loudness_lufs is not None:
            line("Integrated LUFS", f"{report.loudness_lufs:.1f}")
        line("Crest factor (dB)", f"{report.crest_db:.1f}")
        f.write("\n--- Frequency Content ---\n")
        line("Spectral centroid (mean)", f"{report.spectral_centroid_hz_mean:.0f} Hz")
        line("Spectral centroid (std)", f"{report.spectral_centroid_hz_std:.0f} Hz")
        line("Zero crossing rate", f"{report.zero_crossing_rate_mean:.4f}")
        f.write(f"Spectral balance: Low {report.spectral_balance['low_pct']:.1f}% | "
               f"Mid {report.spectral_balance['mid_pct']:.1f}% | "
               f"High {report.spectral_balance['high_pct']:.1f}%\n")
        f.write("\n--- Mix Characteristics ---\n")
        line("Harmonic/Percussive ratio", f"{report.harmonic_percussive_ratio:.2f}")
        f.write("\n--- Structure ---\n")
        f.write(f"Section markers ({len(section_times)} sections):\n")
        for i, t in enumerate(report.section_times_sec):
            f.write(f"  {labels[i]}: {t:.2f}s\n")
    
    if verbose:
        print(f"\nAnalysis complete! Files written to: {os.path.dirname(path) or '.'}", file=sys.stderr)
    
    return report


def main():
    ap = argparse.ArgumentParser(
        description="JamGauge – Enhanced Local Audio Feedback Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument('--input', required=True, help='Path to audio file')
    ap.add_argument('--sr', type=int, default=22050, help='Analysis sample rate (default: 22050)')
    ap.add_argument('--start', type=float, default=0.0, help='Start time in seconds')
    ap.add_argument('--end', type=float, default=-1.0, help='End time in seconds (-1 for EOF)')
    ap.add_argument('--sections', type=int, default=8, help='Target number of sections to detect')
    ap.add_argument('--no-plots', action='store_true', help='Disable PNG plot generation')
    ap.add_argument('--quiet', action='store_true', help='Suppress progress messages')
    args = ap.parse_args()
    
    path = args.input
    if not os.path.isfile(path):
        print(f"Error: File not found: {path}", file=sys.stderr)
        sys.exit(1)
    
    try:
        report = analyze(
            path,
            sr=args.sr,
            start=args.start,
            end=args.end,
            plots=not args.no_plots,
            n_sections=args.sections,
            verbose=not args.quiet,
        )
        
        # Console summary (JSON)
        if not args.quiet:
            print("\n" + "=" * 60, file=sys.stderr)
            print("JSON Summary:", file=sys.stderr)
            print("=" * 60, file=sys.stderr)
        print(json.dumps(asdict(report), indent=2))
        
    except Exception as e:
        print(f"\nError during analysis: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
