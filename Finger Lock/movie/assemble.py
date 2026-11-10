"""Join the scene clips into the full movie, with and without the music.

  python movie/assemble.py                       -> out/finger_lock_explainer_silent.mp4 (no audio)
                                                    out/finger_lock_explainer_music.mp4  (with background music)
  python movie/add_voice.py my_voice.wav         -> out/finger_lock_explainer_final.mp4  (voice over the music)
"""
import os
import subprocess
import sys

import imageio_ffmpeg

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
FF = imageio_ffmpeg.get_ffmpeg_exe()
N = 13


def run(args):
    r = subprocess.run([FF, "-y", "-loglevel", "error"] + args, capture_output=True, text=True)
    if r.returncode:
        print(r.stderr)
        sys.exit(1)


lst = os.path.join(OUT, "concat.txt")
with open(lst, "w") as f:
    for i in range(N):
        f.write(f"file '{os.path.join(OUT, f's{i}.mp4').replace(chr(92), '/')}'\n")
silent = os.path.join(OUT, "finger_lock_explainer_silent.mp4")
run(["-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", silent])
music = os.path.join(OUT, "music.wav")
if os.path.exists(music):
    run(["-i", silent, "-i", music, "-c:v", "copy", "-c:a", "aac", "-b:a", "160k",
         "-map", "0:v", "-map", "1:a", "-shortest", os.path.join(OUT, "finger_lock_explainer_music.mp4")])
print("done:", silent)
