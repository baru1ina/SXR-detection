source_dir="data/raw"

SXR_CHANNELS = [
    "SXR 15 мкм",
    "SXR 27 мкм",
    "SXR 50 mkm",
    "SXR 80 mkm",
    "SXR 127 мкм"
]

IP_CHANNEL = "Ip внутр.(Пр2ВК) (инт.18)"

ALL_CHANNELS = SXR_CHANNELS + [IP_CHANNEL]

PATH_TO_RES_AUTOCORR = 'data/process/autocorr'
PATH_TO_RES_TEMP = 'data/process/temp_res'