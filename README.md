# ME193

Python scripts for controlling LEGO Education hardware using the
[legoeducation](https://github.com/LEGO/LEGOEducation) library.

## Setup

```powershell
:: 1. Create a virtual environment named "my_env"
python -m venv my_env

:: 2. Activate the virtual environment
my_env\Scripts\activate.bat

:: 3. Upgrade pip and install the package
python -m pip install --upgrade pip
pip install legoeducation
```

## Files

- [single_motor.py](single_motor.py) - connects to a Single Motor, runs it at low
  speed, speeds up after one full rotation, then stops and disconnects.

## Usage

With the virtual environment activated and a Single Motor connected via its
Connection Card, update `CARD_COLOR` and `CARD_SERIAL` in `single_motor.py` to
match your card, then run:

```powershell
python single_motor.py
```
