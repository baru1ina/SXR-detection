import shtRipper

print("Все атрибуты модуля shtRipper:")
print(dir(shtRipper))

print("\nАтрибуты shtRipper.ripper:")
print(dir(shtRipper.ripper))

import inspect
print("\nФункции в модуле shtRipper:")
for name, obj in inspect.getmembers(shtRipper):
    if inspect.isfunction(obj):
        print(f"Функция: {name}")
    elif inspect.isclass(obj):
        print(f"Класс: {name}")

print("\nФункции в shtRipper.ripper:")
for name, obj in inspect.getmembers(shtRipper.ripper):
    if inspect.isfunction(obj):
        print(f"Функция: {name}")

print("Документация модуля:", shtRipper.__doc__)
print("Документация ripper:", shtRipper.ripper.__doc__)

help(shtRipper)
help(shtRipper.ripper)
