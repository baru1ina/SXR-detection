# SXR-detection

## Описание
Программа для автоматического детектирования срывов пилообразных колебаний по данным SXR-диагностики на токамак «Глобус-М2»

## Установка

```bash
git clone <https://github.com/baru1ina/SXR-detection.git>
cd SXR-detection
pip install -r requirements.txt
```

## Запуск 
Пути к папке с файлами ```.SHT```  можно задать в файе ```config/path.py```

```python
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
```

Программу можно запустить как через терминал, так и в среде программирования. 

### Запуск через терминал

Чтобы запустить детектирование через терминал можно воспользоваться одной из следующих команд:
```bash
python main.py --mode detect --file sht46358.SHT sht39627.SHT --ch "SXR 80 mkm" "SXR 15 мкм" --wt_threshold 1 25
python main.py --mode detect --file sht46358.SHT sht39627.SHT --ch "SXR 80 mkm" "SXR 15 мкм"
python main.py --mode detect --file sht46358.SHT --ch "SXR 80 mkm"
python main.py --mode detect --file sht46358.SHT sht39627.SHT
python main.py --mode detect --file sht46358.SHT sht39627.SHT --ch "SXR 80 mkm"
```

Обязательным параметром для запуска через консоль является имя файла (набор имен файлов) ```.sht```

Несколько файлов указываются через пробел (см. пример)

Остальные параметры (толщина фольги  ```--ch``` и пороговое значение для отсеивания аномальных значений в энергетическом профиле вайвлет-преоразования ```--wt_threshold```) необязательны и, если не задаются пользователем через консоль, полагаются равными значениям по умолчанию ```ch="SXR 50 mkm"``` и ```wt_threshold = 1.0```

Если в последовательности имен файлов больше, чем значений в последовательности необязательных параметров, пропуски значений для толщины фольги и порога будут выставлены по умолчанию, например:
```bash
python main.py --mode detect --file a.SHT b.SHT c.SHT --ch "SXR 80 mkm" "SXR 15 мкм" --wt_threshold 2.0
```
Для файла a.SHT будут установлены параметры ```--ch="SXR 80 mkm"; --wt_threshold=2.0```

Для файла b.SHT будут установлены параметры ```--ch="SXR 15 мкм"; --wt_threshold=1.0```

Для файла c.SHT будут установлены параметры ```--ch="SXR 50 mkm"; --wt_threshold=1.0```

### Запуск через PyCharm

Чтобы запустить программу в среде программирования, необходимо задать две переменные в файле ```main.py```

```python
 test_files_channels = [
            ("sht46358.SHT", "SXR 50 mkm"),
            ("sht39627.SHT", "SXR 15 мкм"),
            ("sht45898.SHT", "SXR 127 мкм"),
        ]

        wt_thresholds = [1.5, 1, 2]

        main(mode="detect", files_channels=test_files_channels, log_to_file=False)
```

### Логирование

Парметр ```log_to_file=False``` следует изменить на значение ```True```, если необходим вывод логов в файл (по расположению ```.\logs```). При работе в терминале вывод логов осуществляется по умолчанию. 

Графики сигнала с помеченными срывами сохраняются в ```data/plots```.
