"""Mix your voice-over (wav/mp3/m4a) over the quiet music and the silent video.

  python movie/add_voice.py my_voice.wav [--music 0.15] [--offset 0.0]

--music  background music volume under the voice (default 0.15, voice stays at 1.0)
--offset seconds of silence before the voice starts (default 0)
Writes movie/out/finger_lock_explainer_final.mp4 (video length is kept).
"""
import argparse
import os
import subprocess
import sys

import imageio_ffmpeg

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
ap = argparse.ArgumentParser()
ap.add_argument("voice")
ap.add_argument("--music", type=float, default=0.15)
ap.add_argument("--offset", type=float, default=0.0)
args = ap.parse_args()

delay = int(args.offset * 1000)
flt = (f"[1:a]adelay={delay}|{delay},volume=1.0,apad[v];"
       f"[2:a]volume={args.music}[m];"
       "[v][m]amix=inputs=2:duration=longest:normalize=0[a]")
cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
       "-i", os.path.join(OUT, "finger_lock_explainer_silent.mp4"), "-i", args.voice,
       "-i", os.path.join(OUT, "music.wav"), "-filter_complex", flt,
       "-map", "0:v", "-map", "[a]", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest",
       os.path.join(OUT, "finger_lock_explainer_final.mp4")]
r = subprocess.run(cmd, capture_output=True, text=True)
if r.returncode:
    print(r.stderr)
    sys.exit(1)
print("saved", os.path.join(OUT, "finger_lock_explainer_final.mp4"))
