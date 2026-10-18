"""Songs as (frequency Hz, duration ms) notes. 0 Hz is a rest.

The UNO Q app keeps a copy of SAD in ArduinoApps/sumo-ultrasonic/python/main.py;
keep the two in sync.
"""

# "Wah wah wah waaah": a sad descending trombone.
SAD = [
    (392, 450), (0, 80),     # G4
    (370, 450), (0, 80),     # F#4
    (349, 450), (0, 80),     # F4
    (330, 300), (311, 300), (330, 300), (311, 300), (330, 900),  # E4 wobble
]

SONGS = {"sad": SAD}
