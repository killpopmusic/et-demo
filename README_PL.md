# Aplikacja demonstracyjna śledzenia wzroku
Projekt demonstruje działanie algorytmu śledzenia wzroku, bazującego jedynie na współrzędnych punktów charakterystycznych twarzy. Potwierdzono działanie na 3 typach urządzeń:

- Gałąź **main** zakłada urządzenie z GPU Nvidia, ponieważ używa CUDA do przyspieszenia wnioskowania modelu.
- Gałąź **laptop** jest dostosowana do systemów bez dedykowanego GPU.
- Aplikacja została przetestowana na Raspberry Pi 4b. Aby ją uruchomić, należy postępować zgodnie z instrukcjami umieszczonymi w pliku README na gałęzi **rpi**.

Demo składa się z 2 trybów:
- *Tryb ewaluacji (Evaluation mode)*: stworzony do testowania wydajności i dokładności modelu, zawiera pomiar metryk;
- *Tryb galerii (Gallery mode)*: umożliwia użytkownikom przeglądanie zdjęć umieszczonych w folderze **Gallery**. Można swobodnie dodawać i przeglądać własne zdjęcia.

## Wymagania
- Ubuntu 22.04 lub nowsze
- Kamera internetowa

## Konfiguracja
Projekt wykorzystuje system zarządzania zależnościami poetry. Aby zainstalować:
```
curl -sSL https://install.python-poetry.org | python -
```
Aby zainstalować zależności i utworzyć środowisko:

```
poetry install
```
Środowisko należy uruchomić używając:

```
poetry shell
```

Aby uruchomić aplikację:

```
poetry run python3  src/main.py
```
