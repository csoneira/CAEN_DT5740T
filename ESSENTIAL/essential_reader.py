#!/usr/bin/env python3
"""
Minimal DT5740 reader: computes trigger time (s) and integral for selected channels only.
UPDATED: Supports dynamic binary header parsing for CAEN x740 family (6-word header).
         + Includes status printout every 1000 events.

Configuration:
  - essential_config.json : data paths, output directory, interesting channels, sample format.
  - input_dt5740.txt      : acquisition parameters (baseline samples, integration windows, etc.).
"""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

MAX_CHANNELS = 32
# CAEN x740 family (and standard WaveDump binary) uses 6 words for header
HEADER_WORDS = 6 
TTT_MAX = 2**31
TTT_RES_NS = 16.0  # ns


class SequentialReader:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.lines = path.read_text().splitlines()
        self.index = 0

    def next_line(self, allow_blank: bool = False) -> str:
        while self.index < len(self.lines):
            raw = self.lines[self.index]
            self.index += 1
            cleaned = raw.split("!", 1)[0]
            if allow_blank:
                return cleaned.rstrip()
            cleaned = cleaned.strip()
            if cleaned:
                return cleaned
        raise ValueError(f"Unexpected EOF while parsing {self.path}")

    def next_tokens(self) -> List[str]:
        return self.next_line().split()


def read_value_block(reader: SequentialReader, count: int) -> List[float]:
    values: List[float] = []
    while len(values) < count:
        tokens = reader.next_tokens()
        for token in tokens:
            values.append(float(token))
            if len(values) == count:
                break
    return values


class InputConfig:
    def __init__(self, path: Path) -> None:
        reader = SequentialReader(path)
        self.suffix = reader.next_line(allow_blank=True).strip()
        self.samples = int(float(reader.next_line()))
        self.blsamples = int(float(reader.next_line()))
        self.srate = float(reader.next_line())
        self.step = 1.0 / self.srate
        self.thres = float(reader.next_line())
        self.ichmax = int(float(reader.next_line()))
        self.maxpulses = int(float(reader.next_line()))
        self.minlen = float(reader.next_line())
        self.tbinsize = float(reader.next_line())
        block_size = 8
        block_count = max(1, math.ceil(self.ichmax / block_size))

        def alloc_int(default: int = 0) -> np.ndarray:
            arr = np.full(MAX_CHANNELS, default, dtype=np.int32)
            return arr

        self.doWrPul = alloc_int()
        self.polarity = np.ones(MAX_CHANNELS, dtype=np.int32)
        self.inti = alloc_int(0)
        self.intf = alloc_int(0)

        for block in range(block_count):
            block_indices = range(block * block_size, (block + 1) * block_size)
            tokens = reader.next_tokens()
            for offset, channel in enumerate(block_indices):
                if channel < MAX_CHANNELS:
                    self.doWrPul[channel] = int(float(tokens[offset]))

            tokens = reader.next_tokens()
            for offset, channel in enumerate(block_indices):
                if channel < MAX_CHANNELS:
                    self.polarity[channel] = int(float(tokens[offset]))

            for offset, channel in enumerate(block_indices):
                tokens = reader.next_tokens()
                if channel < MAX_CHANNELS:
                    self.inti[channel] = int(float(tokens[0]))
                    self.intf[channel] = int(float(tokens[1]))

            for _ in block_indices:
                reader.next_tokens()  # skip energy window lines (unused here)


class WaveReader:
    def __init__(
        self,
        path: Path,
        expected_samples: int,
        sample_dtype: np.dtype,
        header_words: int = HEADER_WORDS,
    ) -> None:
        self.path = path
        self.expected_samples = expected_samples
        self.sample_dtype = sample_dtype
        self.header_dtype = np.dtype("<u4")
        self.header_words = header_words
        self.header_size_bytes = self.header_words * self.header_dtype.itemsize
        self.file = path.open("rb")

    def read_event(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        # 1. Read Header (6 words = 24 bytes)
        header_bytes = self.file.read(self.header_size_bytes)
        if len(header_bytes) < self.header_size_bytes:
            return None # End of file
        
        header = np.frombuffer(header_bytes, dtype=self.header_dtype, count=self.header_words)

        # 2. Determine Waveform Size from Header
        # header[0] contains the Total Event Size in Bytes (Header + Data)
        total_event_size_bytes = header[0]
        data_size_bytes = total_event_size_bytes - self.header_size_bytes

        # Sanity check
        if data_size_bytes < 0:
            print(f"[ERROR] Invalid event size in header: {total_event_size_bytes}")
            return None

        # 3. Read Waveform Data
        sample_bytes = self.file.read(data_size_bytes)
        if len(sample_bytes) < data_size_bytes:
            return None # Incomplete event file
        
        # 4. Convert bytes to samples
        pulse = np.frombuffer(sample_bytes, dtype=self.sample_dtype)
        
        # 5. Handle Record Length Mismatch
        if len(pulse) != self.expected_samples:
            if len(pulse) > self.expected_samples:
                pulse = pulse[:self.expected_samples]
            # If less, we continue, and the main loop handles padding if needed

        pulse = pulse.astype(np.float32)
        return header, pulse

    def close(self) -> None:
        self.file.close()


def resolve_paths(base: Path, entries: List[str]) -> List[Path]:
    resolved = []
    for entry in entries:
        path = Path(entry)
        if not path.is_absolute():
            path = (base / path).resolve()
        resolved.append(path)
    return resolved


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Minimal DT5740 integral extractor.")
    parser.add_argument("--config", default=str(script_dir / "essential_config.json"))
    parser.add_argument("--input", default=str(script_dir / "input_dt5740.txt"))
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    with config_path.open() as fh:
        cfg = json.load(fh)
    base = config_path.parent

    data_paths = resolve_paths(base, cfg.get("data_paths", []))
    channels = cfg.get("channels", [])
    suffix = cfg.get("suffix", "")
    sample_format = cfg.get("sample_format", "int16").lower()
    max_events_cfg = cfg.get("max_events", 0)

    input_cfg = InputConfig(Path(args.input).resolve())
    print("")
    print("############################################################")
    print(f"# RECORD_LENGTH in input_dt5740.txt = {input_cfg.samples}")
    print("# IMPORTANT: ensure WaveDump's RECORD_LENGTH matches this value")
    print("############################################################")
    print("")
    output_dir = resolve_paths(base, [cfg.get("output_dir", "ESSENTIAL_OUTPUTS")])[0]
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "integrals.csv"

    dtype = np.dtype("<i2") if sample_format == "int16" else np.dtype("<f4")

    def wave_filename(ch: int) -> str:
        if suffix:
            return f"wave_{ch}{suffix}.dat"
        return f"wave_{ch}.dat"

    readers: Dict[int, WaveReader] = {}
    for ch in channels:
        fname = wave_filename(ch)
        file_path = None
        for data_dir in data_paths:
            candidate = data_dir / fname
            if candidate.exists():
                file_path = candidate
                break
        if file_path is None:
            print(f"[WARN] Missing file for channel {ch}: {fname}")
            continue
        readers[ch] = WaveReader(file_path, input_cfg.samples, dtype)
        print(f"Using {file_path} for channel {ch}")

    if not readers:
        print("No valid wave files found. Exiting.")
        return

    with results_path.open("w") as out:
        out.write("ipulse,start_time_s,channel,fine_time_ns,integral\n")
        trgsample = int(math.floor(0.2 * input_cfg.samples))
        pretime = {ch: 0 for ch in readers}
        timecycle = {ch: 0 for ch in readers}
        event_index = 0
        while True:
            if max_events_cfg and event_index >= max_events_cfg:
                break
            event_index += 1
            complete = True
            event_data: Dict[int, Tuple[float, float, float]] = {}

            for ch, reader in readers.items():
                payload = reader.read_event()
                if payload is None:
                    complete = False
                    break
                header, pulse = payload

                if len(pulse) < input_cfg.blsamples:
                    print(f"[WARN] Event {event_index} Ch {ch}: Pulse too short.")
                    complete = False
                    break

                bl = float(np.mean(pulse[: input_cfg.blsamples]))
                polarity = input_cfg.polarity[ch]
                peak = float(np.max(pulse)) if polarity >= 1 else float(np.min(pulse))
                thresval = input_cfg.thres * (peak - bl)

                compare = (lambda value: (value - bl) >= thresval) if polarity >= 1 else (lambda value: (value - bl) <= thresval)
                thres_sample = 1
                for idx, value in enumerate(pulse, start=1):
                    if compare(value):
                        thres_sample = idx
                        break

                shift = thres_sample - trgsample
                newpulse = np.empty_like(pulse)
                if len(newpulse) != input_cfg.samples:
                    newpulse = np.zeros(input_cfg.samples, dtype=np.float32)

                for idx in range(1, input_cfg.samples + 1):
                    src = idx + shift
                    if 1 <= src <= len(pulse):
                        newpulse[idx - 1] = pulse[src - 1] - bl
                    else:
                        newpulse[idx - 1] = 0.0

                raw_ttt = int(header[5])
                
                if pretime[ch] > raw_ttt:
                    timecycle[ch] += 1
                pretime[ch] = raw_ttt
                start_time = (raw_ttt + timecycle[ch] * TTT_MAX) * TTT_RES_NS / 1e9
                fine_time = thres_sample * input_cfg.step

                lo = max(0, min(input_cfg.samples - 1, input_cfg.inti[ch] - 1))
                hi = max(0, min(input_cfg.samples - 1, input_cfg.intf[ch] - 1))
                if hi < lo:
                    lo, hi = hi, lo
                integral = float(polarity * np.sum(newpulse[lo : hi + 1]))
                event_data[ch] = (start_time, fine_time, integral)

                # ========================================================
                # NEW: Print Header and Pulse Info every 1000 events
                # ========================================================
                if event_index % 1000 == 0:
                    print(f"[MONITOR] Event #{event_index} | Channel {ch}")
                    print(f"  HEADER -> EvtCnt: {header[4]} | TTT: {header[5]} | Size(bytes): {header[0]}")
                    print(f"  PULSE  -> BaseLn: {bl:.1f} | Peak: {peak:.1f} | Integral: {integral:.1f}")
                    print(f"  TIME   -> {start_time:.6f} s")
                    print("-" * 50)
                # ========================================================

            if not complete:
                break

            for ch in channels:
                if ch not in event_data:
                    continue
                start_time, fine_time, integral = event_data[ch]
                out.write(f"{event_index},{start_time:.9f},{ch},{fine_time:.3f},{integral:.6f}\n")

    for reader in readers.values():
        reader.close()

    print(f"Wrote integrals to {results_path}")


if __name__ == "__main__":
    main()