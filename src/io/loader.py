import os
import numpy as np
from pathlib import Path
from typing import List, Optional, Sequence

import shtRipper

from .shot import ChannelSignal, MultiRateShot, Shot
from config.path import ALL_CHANNELS


class SHTLoader:
    def __init__(self, source_dir: str, logger):
        self.source_dir = source_dir
        self.logger = logger

    def _full_path(self, filename: str) -> str:
        full_path = os.path.join(self.source_dir, filename)
        if not os.path.isfile(full_path):
            raise FileNotFoundError(f"{full_path} not found")
        return full_path

    @staticmethod
    def _channel_from_raw(name: str, raw_channel: dict) -> ChannelSignal:
        if not isinstance(raw_channel, dict):
            raise ValueError(f"Channel {name!r}: expected a mapping from shtRipper")
        if "x" not in raw_channel or "y" not in raw_channel:
            raise ValueError(f"Channel {name!r}: shtRipper data must contain 'x' and 'y'")

        metadata = {
            key: value
            for key, value in raw_channel.items()
            if key not in {"x", "y"}
        }
        return ChannelSignal(
            name=name,
            time=np.asarray(raw_channel["x"], dtype=float),
            values=np.asarray(raw_channel["y"], dtype=float),
            metadata=metadata,
        )

    def list_available_channels(self, filename: str) -> List[str]:
        """возращает все ключи списком, т.е. названия всех доступных данных из файла разряда"""

        full_path = self._full_path(filename)
        self.logger.info(f"Inspecting channels in {filename} from {full_path}")
        raw_data = shtRipper.ripper.read(full_path)
        if not isinstance(raw_data, dict):
            raise ValueError("shtRipper.read() must return a channel mapping")
        return list(raw_data)

    def load_channel(self, filename: str, channel: str) -> ChannelSignal:
        shot = self.load_multirate_shot(filename, channels=[channel])
        return shot.get_channel(channel)

    def load_multirate_shot(
        self,
        filename: str,
        channels: Optional[Sequence[str]] = None,
    ) -> MultiRateShot:
        """загрузка данных разряда, пока без приведения к общей временной сетке"""

        full_path = self._full_path(filename)
        requested = None if channels is None else list(dict.fromkeys(channels))

        self.logger.info(f"Loading {filename} from {full_path}")
        if requested is None:
            raw_data = shtRipper.ripper.read(full_path)
            channel_order = list(raw_data)
        else:
            raw_data = shtRipper.ripper.read(full_path, requested)
            channel_order = requested

        if not isinstance(raw_data, dict):
            raise ValueError("shtRipper.read() must return a channel mapping")

        loaded = {}
        for name in channel_order:
            if name not in raw_data:
                self.logger.warning(f"Warning: channel {name} not found in file")
                continue
            loaded[name] = self._channel_from_raw(name, raw_data[name])

        if not loaded:
            raise ValueError("No channels loaded")

        return MultiRateShot(
            channels=loaded,
            shot_id=Path(filename).stem,
            metadata={"file": filename},
        )

    def load_shot(
        self,
        filename: str,
        channels: Optional[List[str]] = None,
    ) -> Shot:

        if channels is None:
            channels = ALL_CHANNELS

        multirate = self.load_multirate_shot(filename, channels=channels)
        return self.common_grid_view(multirate)

    @staticmethod
    def common_grid_view(
        multirate: MultiRateShot,
        channels: Optional[Sequence[str]] = None,
    ) -> Shot:
        """Build the legacy matrix representation from same-grid channels."""

        if channels is None:
            loaded_channels = multirate.channel_names
        else:
            loaded_channels = [
                channel for channel in channels if multirate.has_channel(channel)
            ]
        if not loaded_channels:
            raise ValueError("No channels selected for common-grid view")

        time_ref = multirate.get_channel(loaded_channels[0]).time
        signal_list = []

        for ch in loaded_channels:
            channel = multirate.get_channel(ch)
            if channel.time.shape != time_ref.shape or not np.allclose(time_ref, channel.time):
                raise ValueError(f"Time grid mismatch in channel {ch}")
            signal_list.append(channel.values)

        signals = np.stack(signal_list, axis=1)

        return Shot(
            time=time_ref,
            signals=signals,
            channel_names=loaded_channels,
            shot_id=multirate.shot_id,
            metadata=dict(multirate.metadata),
        )

    def list_files(self):
        return [
            f for f in os.listdir(self.source_dir)
            if os.path.isfile(os.path.join(self.source_dir, f))
            and f.casefold().endswith(".sht")
        ]
