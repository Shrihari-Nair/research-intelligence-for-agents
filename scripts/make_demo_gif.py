"""Build docs/demo/demo.gif from the frames written by frontend/record_demo.mjs (Pillow only; no ffmpeg needed)."""
from __future__ import annotations

from pathlib import Path

from PIL import Image

FRAMES, OUT = Path("docs/demo/frames"), Path("docs/demo/demo.gif")
WIDTH, MS_PER_FRAME = 960, 2600


def main() -> None:
    files = sorted(FRAMES.glob("*.png"))
    if not files:
        raise SystemExit(f"no frames in {FRAMES}; run `cd frontend && node record_demo.mjs` first")
    imgs = []
    for f in files:
        im = Image.open(f).convert("RGB")
        imgs.append(im.resize((WIDTH, round(im.height * WIDTH / im.width)), Image.LANCZOS).quantize(colors=128, method=Image.Quantize.MEDIANCUT))
    imgs[0].save(OUT, save_all=True, append_images=imgs[1:], duration=MS_PER_FRAME, loop=0, optimize=True)
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB, {len(imgs)} frames)")


if __name__ == "__main__":
    main()
