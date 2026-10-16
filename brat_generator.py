#!/usr/bin/env python3
"""
BRAT GENERATOR  -  upload a song, get a Brat-style lyric video for TikTok.

What it does automatically from ONE music file:
  * transcribes the lyrics with word-level timestamps (faster-whisper)
  * detects the tempo and every beat (librosa)
  * finds the hook (the most repeated lyric line) and highlights it
  * can auto-pick the most energetic section of the song as your clip
  * renders a 9:16 MP4 with blurry lowercase "brat" text, synced to the music

Install (Python 3.9+):
    pip install gradio faster-whisper librosa soundfile pillow numpy imageio-ffmpeg

Run:
    python brat_generator.py
Then open http://127.0.0.1:7860 (it opens automatically).

Notes:
  * The first run downloads the Whisper model (small ~ 460 MB), one time only.
  * Lyrics from a full mix are never perfect. After generating, fix any words in
    the "Synced lyrics" box and press "Re-render" (no re-transcribing needed).
    Line format:   start - end | text      (put a star before the text to mark a hook)
"""

import colorsys
import difflib
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from functools import lru_cache
from pathlib import Path

try:
    import numpy as np
    from PIL import Image, ImageDraw, ImageFilter, ImageFont
except ImportError:
    sys.exit("Missing packages. Run:\n  pip install gradio faster-whisper librosa "
             "soundfile pillow numpy imageio-ffmpeg")

OUT_DIR = Path("brat_outputs")
OUT_DIR.mkdir(exist_ok=True)

SQUEEZE = 0.84  # horizontal squash that gives the narrow "Arial Narrow" brat look

# --------------------------------------------------------------------------- #
# Colors
# --------------------------------------------------------------------------- #
BRAT_GREEN = "#8ACE00"

SOLID_PRESETS = {
    "Brat Green": "#8ACE00",
    "Hot Pink": "#FF2D95",
    "Electric Blue": "#1E5BFF",
    "Sunset Orange": "#FF7A00",
    "Purple": "#8B3DFF",
    "Lemon Yellow": "#FFE600",
    "Aqua": "#00E5D0",
    "Red": "#FF1F1F",
    "White": "#FFFFFF",
    "Black": "#000000",
    "Custom (use picker)": None,
}

TEXT_PRESETS = {
    "Black": "#000000",
    "White": "#FFFFFF",
    "Hot Pink": "#FF2D95",
    "Electric Blue": "#1E5BFF",
    "Lemon Yellow": "#FFE600",
    "Brat Green": "#8ACE00",
    "Purple": "#8B3DFF",
    "Orange": "#FF7A00",
    "Custom (use picker)": None,
}

PALETTES = {
    "Brat Pop": ["#8ACE00", "#FF2D95", "#FFE600", "#00E5D0", "#8B3DFF", "#FF7A00"],
    "Neon Party": ["#FF00E5", "#00F0FF", "#B6FF00", "#FF3B3B", "#7A5CFF", "#FFEA00"],
    "Pastel Dream": ["#FFB3D9", "#B8F1D0", "#FFF3B0", "#B9D6FF", "#E0C3FF", "#FFD3B6"],
    "Sunset": ["#FF5E5B", "#FF9F1C", "#FFD23F", "#EE4266", "#C147E9", "#FF7AA2"],
    "Rainbow": ["#FF3B3B", "#FF9F1C", "#FFE600", "#2EE66B", "#18C8FF", "#6A5CFF", "#FF4FD8"],
}

BG_MODES = [
    "Brat Green",
    "Solid Color",
    "Rainbow Gradient",
    "Multicolor Gradient",
    "Multicolor (changes on beat)",
    "Assorted (new color each line)",
]

TEXT_MODES = ["Single color", "Rainbow per word", "Rotate colors per line"]


def parse_color(c, default=(0, 0, 0)):
    """Accepts '#rrggbb', '#rgb' or 'rgb(a)(r, g, b, ...)' and returns an (r, g, b) tuple."""
    if not c:
        return default
    if isinstance(c, (tuple, list)):
        return tuple(int(v) for v in c[:3])
    c = str(c).strip()
    if c.startswith("#"):
        h = c[1:]
        if len(h) == 3:
            h = "".join(ch * 2 for ch in h)
        try:
            return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
        except ValueError:
            return default
    nums = re.findall(r"[\d.]+", c)
    if c.lower().startswith("rgb") and len(nums) >= 3:
        return tuple(int(float(n)) for n in nums[:3])
    return default


# --------------------------------------------------------------------------- #
# Audio analysis: beats, hook section, lyrics
# --------------------------------------------------------------------------- #
def _progress(cb, frac, desc):
    if cb is not None:
        cb(frac, desc=desc)


def find_hook_start(y, sr, clip_len, beat_times):
    """Start time of the loudest/most energetic window (usually the chorus), snapped to a beat."""
    import librosa
    hop = 512
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    win = max(1, int(clip_len * sr / hop))
    if len(rms) <= win:
        return 0.0
    csum = np.concatenate([[0.0], np.cumsum(rms)])
    means = (csum[win:] - csum[:-win]) / win
    best = int(np.argmax(means))
    t = best * hop / sr
    if len(beat_times):
        t = float(beat_times[int(np.argmin(np.abs(beat_times - t)))])
    return t


def group_lines(words, max_words=6, gap=0.65):
    lines, cur = [], []
    for w in words:
        if cur:
            too_long = len(cur) >= max_words
            pause = w["s"] - cur[-1]["e"] > gap
            punct = len(cur) >= 3 and re.search(r"[.!?,]$", cur[-1]["t"])
            if too_long or pause or punct:
                lines.append(cur)
                cur = []
        cur.append(w)
    if cur:
        lines.append(cur)
    out = []
    for ws in lines:
        out.append({
            "start": ws[0]["s"],
            "end": ws[-1]["e"],
            "words": ws,
            "text": " ".join(x["t"] for x in ws),
            "hook": False,
        })
    return out


def _norm(s):
    return re.sub(r"[^a-z0-9' ]", "", s.lower()).strip()


def mark_hooks(lines):
    """Flag the most repeated (fuzzy-matched) line cluster as the hook. Returns the hook phrase."""
    cand = [i for i, l in enumerate(lines)
            if len(_norm(l["text"]).split()) >= 2 and len(_norm(l["text"])) >= 6]
    best_i, best_group = None, []
    for i in cand:
        group = [j for j in cand
                 if difflib.SequenceMatcher(None, _norm(lines[i]["text"]),
                                            _norm(lines[j]["text"])).ratio() >= 0.75]
        if len(group) > len(best_group) or (
                len(group) == len(best_group) and best_i is not None
                and len(lines[i]["text"]) > len(lines[best_i]["text"])):
            best_i, best_group = i, group
    if best_i is None or len(best_group) < 2:
        return ""
    for j in best_group:
        lines[j]["hook"] = True
    return lines[best_i]["text"]


def transcribe(wav_path, model_size, language):
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise RuntimeError("faster-whisper is not installed. Run: pip install faster-whisper")
    model = WhisperModel(model_size, device="auto", compute_type="auto")
    segments, _info = model.transcribe(
        wav_path,
        language=None if language == "auto" else language,
        word_timestamps=True,
        vad_filter=True,
        condition_on_previous_text=False,
        beam_size=5,
    )
    words = []
    for seg in segments:
        for w in (seg.words or []):
            t = w.word.strip()
            if t:
                words.append({"t": t, "s": float(w.start), "e": float(w.end)})
    return words


def analyze_audio(audio_path, clip_len, auto_hook, start, model_size, language, progress=None):
    import librosa
    import soundfile as sf

    _progress(progress, 0.03, "Loading audio...")
    y, sr = librosa.load(audio_path, sr=22050, mono=True)
    total = len(y) / sr
    clip_len = float(min(clip_len, total))

    _progress(progress, 0.10, "Detecting tempo & beats...")
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(np.atleast_1d(tempo)[0])
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    if auto_hook:
        _progress(progress, 0.18, "Finding the most energetic section (hook)...")
        start = find_hook_start(y, sr, clip_len, beat_times)
    start = float(max(0.0, min(start, total - clip_len)))
    end = start + clip_len

    clip = y[int(start * sr):int(end * sr)]
    tmp_wav = os.path.join(tempfile.mkdtemp(), "clip.wav")
    sf.write(tmp_wav, clip, sr)

    _progress(progress, 0.25, f"Transcribing lyrics with Whisper '{model_size}' "
                              "(first run downloads the model)...")
    words = transcribe(tmp_wav, model_size, language)
    lines = group_lines(words)
    hook = mark_hooks(lines)

    beats = [float(b - start) for b in beat_times if start <= b <= end]

    return {
        "audio_path": audio_path,
        "clip_start": start,
        "clip_dur": clip_len,
        "tempo": tempo,
        "beats": beats,
        "lines": lines,
        "hook": hook,
    }


# --------------------------------------------------------------------------- #
# Editable lyric text  <->  line objects
# --------------------------------------------------------------------------- #
LINE_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*[-\u2013]\s*(\d+(?:\.\d+)?)\s*\|\s*(.*?)\s*$")


STAR = "\u2605"


def lines_to_text(lines):
    out = []
    for l in lines:
        mark = STAR + " " if l["hook"] else ""
        out.append(f"{l['start']:.2f} - {l['end']:.2f} | {mark}{l['text']}")
    return "\n".join(out)


def parse_lyrics(text, original_lines):
    out = []
    for raw in (text or "").splitlines():
        m = LINE_RE.match(raw)
        if not m:
            continue
        s, e, body = float(m.group(1)), float(m.group(2)), m.group(3)
        hook = body.startswith("\u2605")
        body = body.lstrip("\u2605").strip()
        if not body:
            continue
        if e <= s:
            e = s + 1.0
        words = None
        for o in original_lines:
            if (abs(o["start"] - s) < 0.02 and abs(o["end"] - e) < 0.02
                    and _norm(o["text"]) == _norm(body)):
                words = o["words"]
                break
        if words is None:  # edited line: spread word timings by word length
            toks = body.split()
            lens = [max(1, len(t)) for t in toks]
            tot = float(sum(lens))
            words, cur = [], s
            for t, ln in zip(toks, lens):
                d = (e - s) * ln / tot
                words.append({"t": t, "s": cur, "e": cur + d})
                cur += d
        out.append({"start": s, "end": e, "words": words, "text": body, "hook": hook})
    out.sort(key=lambda l: l["start"])
    return out


# --------------------------------------------------------------------------- #
# Text rendering
# --------------------------------------------------------------------------- #
FONT_CANDIDATES = [
    "Arial Narrow.ttf", "arialn.ttf", "ARIALN.TTF",
    "/System/Library/Fonts/Supplemental/Arial Narrow.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Regular.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSansNarrow-Regular.ttf",
    "Arial.ttf", "arial.ttf", "/Library/Fonts/Arial.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "DejaVuSans.ttf",
]


@lru_cache(maxsize=128)
def get_font(size):
    for c in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(c, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def build_layout(words, W, H, scale):
    """Word-wrap + auto-fit the full line so text never shifts as words appear."""
    avail_w = W * 0.86 / SQUEEZE
    avail_h = H * 0.60
    size = max(30, int(W * 0.19 * scale))
    while True:
        font = get_font(size)
        space = font.getlength(" ")
        asc, desc = font.getmetrics()
        lh = int((asc + desc) * 0.98)
        rows, cur, cur_w = [], [], 0.0
        for i, w in enumerate(words):
            ww = font.getlength(w)
            add = ww if not cur else space + ww
            if cur and cur_w + add > avail_w:
                rows.append((cur, cur_w))
                cur, cur_w, add = [], 0.0, ww
            cur.append((i, w, ww))
            cur_w += add
        if cur:
            rows.append((cur, cur_w))
        total_h = lh * len(rows)
        fits = all(rw <= avail_w for _, rw in rows) and total_h <= avail_h
        if fits or size <= 28:
            break
        size = int(size * 0.92)

    block_w = int(max(rw for _, rw in rows)) + 1
    pos = []
    for ri, (items, rw) in enumerate(rows):
        x = (block_w - rw) / 2
        y = ri * lh
        for i, w, ww in items:
            pos.append((i, w, x, y))
            x += ww + space
    return {"size": size, "w": block_w, "h": total_h, "pos": pos}


def render_text_layer(layout, n_visible, word_colors, blur):
    pad = int(blur * 3) + 8
    w, h = layout["w"] + 2 * pad, layout["h"] + 2 * pad
    sw = max(1, int(w * SQUEEZE))
    font = get_font(layout["size"])
    groups = {}
    for i, t, x, y in layout["pos"]:
        if i < n_visible:
            groups.setdefault(word_colors[i], []).append((t, x, y))
    out = Image.new("RGBA", (sw, h), (0, 0, 0, 0))
    for col, items in groups.items():
        mask = Image.new("L", (w, h), 0)
        d = ImageDraw.Draw(mask)
        for t, x, y in items:
            d.text((x + pad, y + pad), t, font=font, fill=255)
        mask = mask.resize((sw, h), Image.LANCZOS)
        if blur > 0:
            mask = mask.filter(ImageFilter.GaussianBlur(blur))
        layer = Image.new("RGBA", (sw, h), tuple(col) + (255,))
        layer.putalpha(mask)
        out = Image.alpha_composite(out, layer)
    return out


# --------------------------------------------------------------------------- #
# Backgrounds
# --------------------------------------------------------------------------- #
def _lerp_palette(p, pal):
    n = len(pal)
    stops = np.linspace(0, 1, n + 1)
    ext = np.vstack([pal, pal[:1]])
    return np.stack([np.interp(p, stops, ext[:, c]) for c in range(3)], axis=-1)


def make_background(opts, line_starts, beats):
    """Returns f(t, flash) -> PIL RGB image. `flash` (0..1) lightens the colors on each beat."""
    W, H = opts["W"], opts["H"]
    mode = opts["bg_mode"]
    pal = np.array(opts["palette"], dtype=np.float32)
    beats = np.asarray(beats, dtype=np.float64)

    def lighten(rgb, f):
        rgb = np.asarray(rgb, dtype=np.float32)
        return tuple(int(v) for v in (rgb + (255 - rgb) * f))

    if mode in ("Brat Green", "Solid Color"):
        base = parse_color(BRAT_GREEN) if mode == "Brat Green" else opts["solid"]
        return lambda t, f: Image.new("RGB", (W, H), lighten(base, f))

    if mode in ("Rainbow Gradient", "Multicolor Gradient"):
        if mode == "Rainbow Gradient":
            colors = np.array([parse_color(c) for c in PALETTES["Rainbow"]], dtype=np.float32)
        else:
            colors = pal
        lw, lh = max(2, W // 8), max(2, H // 8)
        xs, ys = np.linspace(0, 1, lw), np.linspace(0, 1, lh)
        grid = xs[None, :] * 0.4 + ys[:, None] * 0.6

        def grad(t, f):
            p = (grid + t * 0.07) % 1.0
            arr = _lerp_palette(p, colors)
            arr = arr + (255 - arr) * f
            img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")
            return img.resize((W, H), Image.BILINEAR)
        return grad

    if mode == "Multicolor (changes on beat)":
        every = max(1, int(opts["beats_per_color"]))

        def beat_color(t, f):
            idx = int(np.searchsorted(beats, t, side="right")) - 1 if len(beats) else -1
            ci = max(idx, 0) // every
            return Image.new("RGB", (W, H), lighten(pal[ci % len(pal)], f))
        return beat_color

    # Assorted: a new color for every lyric line, with a quick cross-fade
    starts = np.asarray(line_starts, dtype=np.float64)

    def per_line(t, f):
        idx = int(np.searchsorted(starts, t, side="right")) - 1 if len(starts) else -1
        cur = pal[max(idx, 0) % len(pal)]
        if idx >= 1:
            k = (t - starts[idx]) / 0.18
            if k < 1.0:
                prev = pal[(idx - 1) % len(pal)]
                cur = prev * (1 - k) + cur * k
        return Image.new("RGB", (W, H), lighten(cur, f))
    return per_line


# --------------------------------------------------------------------------- #
# Video rendering
# --------------------------------------------------------------------------- #
def ffmpeg_exe():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        raise RuntimeError("ffmpeg not found. Run: pip install imageio-ffmpeg")


def line_word_colors(li, line, opts):
    n = len(line["words"])
    if opts["hook_on"] and line.get("hook"):
        return [opts["hook_rgb"]] * n
    mode = opts["text_mode"]
    if mode == "Rainbow per word":
        cols = []
        for j in range(n):
            r, g, b = colorsys.hsv_to_rgb((j * 0.12 + li * 0.07) % 1.0, 0.85, 1.0)
            cols.append((int(r * 255), int(g * 255), int(b * 255)))
        return cols
    if mode == "Rotate colors per line":
        pal = opts["palette"]
        return [tuple(int(v) for v in pal[li % len(pal)])] * n
    return [opts["text_rgb"]] * n


def render_video(audio_path, out_path, clip_start, clip_dur, beats, lines, opts, progress=None):
    W, H, fps = opts["W"], opts["H"], opts["fps"]
    n_frames = int(clip_dur * fps)
    beats_arr = np.asarray(beats, dtype=np.float64)

    lines = [l for l in lines if l["words"]]
    starts = [l["start"] for l in lines]
    show_from = [l["start"] - 0.08 for l in lines]
    show_to = []
    for i, l in enumerate(lines):
        t_end = l["end"] + 1.0
        if i + 1 < len(lines):
            t_end = min(t_end, lines[i + 1]["start"])
        show_to.append(t_end)

    bg_fn = make_background(opts, starts, beats_arr)
    layouts = {}
    cache = {"key": None, "layer": None}

    log_path = os.path.join(tempfile.mkdtemp(), "ffmpeg.log")
    cmd = [
        ffmpeg_exe(), "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
        "-ss", f"{clip_start:.3f}", "-t", f"{clip_dur:.3f}", "-i", audio_path,
        "-map", "0:v", "-map", "1:a?",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart",
        str(out_path),
    ]
    log = open(log_path, "wb")
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=log, stderr=log)

    cur = 0
    try:
        for fi in range(n_frames):
            t = fi / fps

            # beat pulse
            if len(beats_arr):
                bi = int(np.searchsorted(beats_arr, t, side="right")) - 1
                dt = t - beats_arr[bi] if bi >= 0 else 9.0
            else:
                dt = 9.0
            pulse = math.exp(-dt * 8.0)

            frame = bg_fn(t, opts["flash"] * pulse)

            # which lyric line is on screen?
            if lines:
                while cur + 1 < len(lines) and t >= show_from[cur + 1]:
                    cur += 1
                if show_from[cur] <= t < show_to[cur]:
                    line = lines[cur]
                    is_hook = bool(line.get("hook")) and opts["hook_on"]
                    if opts["word_reveal"]:
                        n_vis = sum(1 for w in line["words"] if w["s"] <= t + 0.02)
                    else:
                        n_vis = len(line["words"])
                    if n_vis > 0:
                        key = (cur, n_vis)
                        if cache["key"] != key:
                            if cur not in layouts:
                                words = [w["t"].lower() if opts["lowercase"] else w["t"]
                                         for w in line["words"]]
                                sc = opts["size_scale"] * (1.08 if is_hook else 1.0)
                                layouts[cur] = build_layout(words, W, H, sc)
                            cols = line_word_colors(cur, line, opts)
                            cache["layer"] = render_text_layer(
                                layouts[cur], n_vis, cols, opts["blur"] * W / 1080.0)
                            cache["key"] = key
                        layer = cache["layer"]
                        scale = 1.0 + opts["pulse"] * pulse * (1.7 if is_hook else 1.0)
                        if abs(scale - 1.0) > 0.002:
                            layer = layer.resize((int(layer.width * scale),
                                                  int(layer.height * scale)), Image.BILINEAR)
                        frame.paste(layer, ((W - layer.width) // 2, (H - layer.height) // 2), layer)

            try:
                proc.stdin.write(frame.tobytes())
            except BrokenPipeError:
                break
            if fi % 15 == 0:
                _progress(progress, 0.55 + 0.43 * fi / max(1, n_frames),
                          f"Rendering video... {fi}/{n_frames} frames")
    finally:
        try:
            proc.stdin.close()
        except Exception:
            pass
        code = proc.wait()
        log.close()

    if code != 0:
        with open(log_path, "rb") as f:
            tail = f.read()[-1500:].decode("utf-8", "ignore")
        raise RuntimeError(f"ffmpeg failed:\n{tail}")
    return str(out_path)


# --------------------------------------------------------------------------- #
# Gradio UI
# --------------------------------------------------------------------------- #
def build_ui():
    import gradio as gr

    with gr.Blocks(title="Brat Generator", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# \U0001F49A brat generator\n"
            "Upload a song \u2192 get synced lyrics, beats, hooks and a ready-to-post TikTok lyric video."
        )
        state = gr.State(None)

        with gr.Row():
            with gr.Column(scale=1):
                audio = gr.Audio(label="Upload your music file", type="filepath",
                                 sources=["upload"])

                with gr.Accordion("Clip & lyrics detection", open=True):
                    clip_len = gr.Slider(10, 90, value=30, step=1, label="Clip length (seconds)")
                    auto_hook = gr.Checkbox(value=True,
                                            label="Auto-pick the hook (most energetic section)")
                    start_t = gr.Slider(0, 600, value=0, step=1, interactive=False,
                                        label="Manual start time (seconds)")
                    model = gr.Dropdown(["tiny", "base", "small", "medium"], value="small",
                                        label="Whisper model (bigger = more accurate, slower)")
                    lang = gr.Dropdown(["auto", "en", "tl", "es", "ko", "ja", "fr", "pt", "id"],
                                       value="auto", label="Lyrics language")

                with gr.Accordion("Background", open=True):
                    bg_mode = gr.Dropdown(BG_MODES, value="Brat Green", label="Background style")
                    solid_preset = gr.Dropdown(list(SOLID_PRESETS), value="Brat Green",
                                               label="Quick solid colors (for 'Solid Color')")
                    solid_color = gr.ColorPicker(value=BRAT_GREEN, label="Solid background color")
                    palette_name = gr.Dropdown(list(PALETTES), value="Brat Pop",
                                               label="Palette (multicolor / assorted modes, "
                                                     "and 'Rotate colors per line' text)")
                    beats_per_color = gr.Slider(1, 8, value=2, step=1,
                                                label="Beats per color change (beat mode)")

                with gr.Accordion("Text", open=True):
                    text_mode = gr.Dropdown(TEXT_MODES, value="Single color", label="Text color mode")
                    text_preset = gr.Dropdown(list(TEXT_PRESETS), value="Black",
                                              label="Quick text colors")
                    text_color = gr.ColorPicker(value="#000000", label="Lyrics text color")
                    hook_on = gr.Checkbox(value=True, label="Highlight hook lines (bigger + own color)")
                    hook_color = gr.ColorPicker(value="#FFFFFF", label="Hook text color")
                    size_scale = gr.Slider(0.5, 1.4, value=1.0, step=0.05, label="Text size")
                    blur = gr.Slider(0, 8, value=2.5, step=0.5, label="Brat blur")
                    lowercase = gr.Checkbox(value=True, label="lowercase (brat style)")
                    word_reveal = gr.Checkbox(value=True, label="Reveal words as they are sung")

                with gr.Accordion("Beat & output", open=False):
                    pulse = gr.Slider(0, 0.2, value=0.06, step=0.01, label="Text beat pulse")
                    flash = gr.Slider(0, 0.5, value=0.12, step=0.01, label="Background beat flash")
                    resolution = gr.Dropdown(["1080x1920 (TikTok HD)", "720x1280 (faster)"],
                                             value="720x1280 (faster)", label="Resolution")

                gen_btn = gr.Button("\u2728 Generate brat video", variant="primary")

            with gr.Column(scale=1):
                video = gr.Video(label="Your brat video", autoplay=True)
                info = gr.Markdown()
                lyrics_box = gr.Textbox(
                    label="Synced lyrics (edit then Re-render).  Format: start - end | text   "
                          "(\u2605 = hook line)",
                    lines=12, interactive=True)
                rerender_btn = gr.Button("\U0001F3AC Re-render with edited lyrics / new style")

        # ---- small UI helpers -------------------------------------------------
        auto_hook.change(lambda v: gr.update(interactive=not v), auto_hook, start_t)
        solid_preset.change(
            lambda n: SOLID_PRESETS.get(n) if SOLID_PRESETS.get(n) else gr.update(),
            solid_preset, solid_color)
        text_preset.change(
            lambda n: TEXT_PRESETS.get(n) if TEXT_PRESETS.get(n) else gr.update(),
            text_preset, text_color)

        style_inputs = {bg_mode, solid_color, palette_name, beats_per_color, text_mode,
                        text_color, hook_on, hook_color, size_scale, blur, lowercase,
                        word_reveal, pulse, flash, resolution}

        def collect_opts(d):
            W, H = (1080, 1920) if d[resolution].startswith("1080") else (720, 1280)
            palette = [parse_color(c) for c in PALETTES[d[palette_name]]]
            return {
                "W": W, "H": H, "fps": 30,
                "bg_mode": d[bg_mode],
                "solid": parse_color(d[solid_color], parse_color(BRAT_GREEN)),
                "palette": palette,
                "beats_per_color": d[beats_per_color],
                "text_mode": d[text_mode],
                "text_rgb": parse_color(d[text_color], (0, 0, 0)),
                "hook_on": bool(d[hook_on]),
                "hook_rgb": parse_color(d[hook_color], (255, 255, 255)),
                "size_scale": float(d[size_scale]),
                "blur": float(d[blur]),
                "lowercase": bool(d[lowercase]),
                "word_reveal": bool(d[word_reveal]),
                "pulse": float(d[pulse]),
                "flash": float(d[flash]),
            }

        def make_info(a):
            hook = f"\u201c{a['hook']}\u201d" if a["hook"] else "none detected (no repeated line)"
            msg = (f"**Tempo:** {a['tempo']:.0f} BPM \u00b7 **Beats:** {len(a['beats'])} \u00b7 "
                   f"**Clip:** {a['clip_start']:.1f}s \u2013 {a['clip_start'] + a['clip_dur']:.1f}s"
                   f" \u00b7 **Hook:** {hook}")
            if not a["lines"]:
                msg += ("\n\n\u26A0\uFE0F No lyrics were detected (instrumental, or try a bigger "
                        "Whisper model). You can type lines in the box using the "
                        "`start - end | text` format and press Re-render.")
            return msg

        def do_render(a, d, progress):
            opts = collect_opts(d)
            out = OUT_DIR / f"brat_{int(time.time())}.mp4"
            try:
                return render_video(a["audio_path"], out, a["clip_start"], a["clip_dur"],
                                    a["beats"], a["lines"], opts, progress)
            except RuntimeError as e:
                raise gr.Error(str(e))

        def on_generate(d, progress=gr.Progress()):
            path = d[audio]
            if not path:
                raise gr.Error("Upload a music file first.")
            try:
                a = analyze_audio(path, d[clip_len], d[auto_hook], d[start_t],
                                  d[model], d[lang], progress)
            except RuntimeError as e:
                raise gr.Error(str(e))
            _progress(progress, 0.52, "Starting render...")
            out = do_render(a, d, progress)
            return out, make_info(a), lines_to_text(a["lines"]), a

        def on_rerender(d, progress=gr.Progress()):
            a = d[state]
            if not a:
                raise gr.Error("Press 'Generate brat video' first.")
            if a["audio_path"] != d[audio]:
                raise gr.Error("The audio changed. Press 'Generate brat video' again.")
            a["lines"] = parse_lyrics(d[lyrics_box], a["lines"])
            a["hook"] = next((l["text"] for l in a["lines"] if l["hook"]), "")
            out = do_render(a, d, progress)
            return out, make_info(a), a

        all_gen = style_inputs | {audio, clip_len, auto_hook, start_t, model, lang}
        gen_btn.click(on_generate, inputs=all_gen, outputs=[video, info, lyrics_box, state])
        rerender_btn.click(on_rerender, inputs=style_inputs | {audio, lyrics_box, state},
                           outputs=[video, info, state])
    return demo


def main():
    missing = []
    for mod, pip in [("gradio", "gradio"), ("librosa", "librosa"), ("soundfile", "soundfile"),
                     ("faster_whisper", "faster-whisper")]:
        try:
            __import__(mod)
        except ImportError:
            missing.append(pip)
    if missing:
        sys.exit("Missing packages. Run:\n  pip install " + " ".join(missing))
    demo = build_ui()
    demo.queue().launch(inbrowser=True, allowed_paths=[str(OUT_DIR.resolve())])


if __name__ == "__main__":
    main()
