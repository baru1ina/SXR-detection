import os
import numpy as np
from typing import List
import shtRipper

from .shot import Shot
from config.path import ALL_CHANNELS


class SHTLoader:
    def __init__(self, source_dir: str):
        self.source_dir = source_dir

    def load_shot(self, filename: str,
                  channels: List[str] = None) -> Shot:

        if channels is None:
            channels = ALL_CHANNELS

        full_path = os.path.join(self.source_dir, filename)

        print(f"Loading {filename} from {full_path}")

        if not os.path.exists(full_path):
            raise FileNotFoundError(f"{full_path} not found")

        raw_data = shtRipper.ripper.read(full_path, channels)

        time_ref = None
        signal_list = []
        loaded_channels = []

        for ch in channels:
            if ch not in raw_data:
                print(f"Warning: channel {ch} not found in file")
                continue

            x = np.array(raw_data[ch]["x"])
            y = np.array(raw_data[ch]["y"])

            if time_ref is None:
                time_ref = x
            else:
                if not np.allclose(time_ref, x):
                    raise ValueError(
                        f"Time grid mismatch in channel {ch}"
                    )

            signal_list.append(y)
            loaded_channels.append(ch)

        if len(signal_list) == 0:
            raise ValueError("No channels loaded")

        signals = np.stack(signal_list, axis=1)

        shot_id = filename.replace(".SHT", "")

        return Shot(
            time=time_ref,
            signals=signals,
            channel_names=loaded_channels,
            shot_id=shot_id,
            metadata={"file": filename}
        )

    def list_files(self):
        return [
            f for f in os.listdir(self.source_dir)
            if f.endswith(".SHT")
        ]
