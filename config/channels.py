from dataclasses import dataclass


@dataclass(frozen=True)
class DiagnosticChannel:
    name: str
    key: str
    group: str
    alignment_method: str = "linear"
    max_gap_factor: float = 2.5

    def __post_init__(self):
        if not self.name:
            raise ValueError("Diagnostic channel name must not be empty")
        if not self.key or not self.key.replace("_", "").isalnum():
            raise ValueError(
                f"Diagnostic channel {self.name!r} must have an alphanumeric key"
            )
        if not self.group:
            raise ValueError(f"Diagnostic channel {self.name!r} must have a group")
        if self.alignment_method not in {"linear", "nearest", "previous"}:
            raise ValueError(
                f"Unsupported alignment method {self.alignment_method!r} "
                f"for channel {self.name!r}"
            )
        if self.max_gap_factor <= 0:
            raise ValueError("max_gap_factor must be positive")


@dataclass(frozen=True)
class DiagnosticChannelProfile:
    name: str
    channels: tuple[DiagnosticChannel, ...]

    def __post_init__(self):
        names = self.channel_names
        if len(names) != len(set(names)):
            raise ValueError(f"Duplicate channel names in profile {self.name!r}")
        keys = self.channel_keys
        if len(keys) != len(set(keys)):
            raise ValueError(f"Duplicate channel keys in profile {self.name!r}")

    @property
    def channel_names(self) -> list[str]:
        return [channel.name for channel in self.channels]

    @property
    def channels_by_name(self) -> dict[str, DiagnosticChannel]:
        return {channel.name: channel for channel in self.channels}

    @property
    def channel_keys(self) -> list[str]:
        return [channel.key for channel in self.channels]

    @property
    def groups(self) -> list[str]:
        return list(dict.fromkeys(channel.group for channel in self.channels))


MULTICHANNEL_FEATURE_PROFILE = DiagnosticChannelProfile(
    name="plasma_diagnostics_v1",
    channels=(
        DiagnosticChannel(
            "Ip новый (Пр1ВК) (инт.16)",
            "ip_new",
            "plasma_current",
        ),
        DiagnosticChannel(
            "Ip внутр.(Пр2ВК) (инт.18)",
            "ip_internal",
            "plasma_current",
        ),
        DiagnosticChannel("D-alfa нижний купол", "dalpha_lower_dome", "d_alpha"),
        DiagnosticChannel("D-alfa верхний купол", "dalpha_upper_dome", "d_alpha"),
        DiagnosticChannel("D-alfa  хорда R=50 cm", "dalpha_r50", "d_alpha"),
        DiagnosticChannel("D-alfa  хорда R=42 cm", "dalpha_r42", "d_alpha"),
        DiagnosticChannel("D-alpha (на столб)", "dalpha_column", "d_alpha"),
        DiagnosticChannel("МГД наружный       ", "mhd_outer", "mhd"),
        DiagnosticChannel("МГД быстрый зонд рад.", "mhd_fast_radial", "mhd"),
        DiagnosticChannel("МГД быстрый зонд тор.", "mhd_fast_toroidal", "mhd"),
        DiagnosticChannel("МГД быстрый зонд верт.", "mhd_fast_vertical", "mhd"),
        DiagnosticChannel("Plasma shift", "plasma_shift", "plasma_position"),
        DiagnosticChannel(
            "Emission electrode voltage",
            "emission_electrode_voltage",
            "voltage",
        ),
        DiagnosticChannel(
            "Напряжение пучка новый инжектор",
            "beam_voltage",
            "voltage",
        ),
        DiagnosticChannel(
            "Up (внутреннее 175 петля)",
            "up_inner",
            "voltage",
        ),
        DiagnosticChannel(
            "Up (внешнее,183 петля, 171 канал)",
            "up_outer",
            "voltage",
        ),
        DiagnosticChannel("U PF3", "u_pf3", "voltage"),
        DiagnosticChannel("Радиометр 8 mm", "radiometer_8mm", "radiometer"),
        DiagnosticChannel(
            "Диамагнитный сигнал (новый инт.)",
            "diamagnetic",
            "diamagnetic",
        ),
    ),
)
