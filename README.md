# 🎬 Video & Lyric Tools

Two small Python tools for making social-media content from your own files:

| Tool | What it does |
|------|--------------|
| [`video_to_gif.py`](#-video-to-gif) | Converts any video into an animated GIF (command line, with a file picker if you don't pass a path) |
| [`brat_generator.py`](#-brat-generator) | Turns a song into a Brat-style synced lyric video for TikTok (web UI that runs locally) |

Everything runs on your own computer. Nothing is uploaded anywhere.

---

## 📁 Project structure

```
.
├── video_to_gif.py
├── brat_generator.py
├── README.md
├── venv/             # your virtual environment (not committed)
└── brat_outputs/     # rendered brat videos (created automatically)
```

## ✅ Requirements

- Python **3.10 or newer**
- About 1 GB free disk space (the Whisper model is downloaded on first use of the Brat Generator)

## ⚙️ Setup

Create a fresh folder and virtual environment, then install only what you need.

**Windows**
```bash
python -m venv venv
venv\Scripts\activate
```

**Mac / Linux**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Install packages**

```bash
# Video to GIF only
pip install opencv-python pillow

# Brat Generator only
pip install gradio faster-whisper librosa soundfile pillow numpy imageio-ffmpeg

# Both tools
pip install opencv-python pillow gradio faster-whisper librosa soundfile numpy imageio-ffmpeg
```

> The Brat Generator does **not** need OpenCV. If you only use that tool, skip `opencv-python`.

---

## 🖼️ Video to GIF

Converts a video into an animated GIF. You can pass a path, or run it with no arguments and pick a file in a window. If no window is available, it asks you to paste a path.

### Usage

```bash
# Opens a file picker
python video_to_gif.py

# Pass a path directly (quotes are fine for paths with spaces)
python video_to_gif.py "C:\Videos\clip.mp4"

# Custom settings
python video_to_gif.py clip.mp4 --start 5 --duration 4 --fps 12 --width 480 -o out.gif
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `video` | file picker | Path to the video |
| `-o`, `--output` | `<video name>.gif` | Output GIF path |
| `--start` | `0` | Start time in seconds |
| `--duration` | until the end | Length of the GIF in seconds |
| `--fps` | `10` | GIF frames per second |
| `--width` | `480` | Output width in pixels (`0` keeps the original size) |
| `--colors` | `256` | Palette size, 2 to 256 |
| `--loop` | `0` | Number of loops (`0` = forever) |

### Example: one minute, original size, 14 fps, looping forever

```bash
python video_to_gif.py "your_video.mp4" --start 5 --duration 60 --fps 14 --width 0 --loop 0
```

### Tips

- GIFs get big fast. For long videos, use a short `--duration`, a lower `--fps` and a smaller `--width`.
- The script holds every frame in memory while building the GIF. A full-size, one-minute GIF can use a lot of RAM and produce a file of hundreds of MB.
- A 3 to 5 second clip at 10 fps and 480 px wide is usually only a few MB.

---

## 💚 Brat Generator

Upload one music file and get a vertical (9:16) lyric video in the Brat style: lowercase, blurry, narrow text on a bold background.

### What it does automatically

- **Synced lyrics:** transcribes the song with word-level timestamps (faster-whisper). Words appear on screen as they are sung.
- **Beat detection:** finds the tempo and every beat (librosa). The text pulses and the background flashes on each beat.
- **Hook detection:** finds the most repeated lyric line and shows it larger and in its own color. This is based on repetition, not AI-written hooks.
- **Auto-clip:** can pick the most energetic section of the song (usually the chorus) as your clip.
- **Export:** renders an H.264 MP4 with AAC audio, ready for TikTok.

### Run

```bash
python brat_generator.py
```

The app opens in your browser at `http://127.0.0.1:7860`.

### How to use it

1. Upload a music file (mp3, wav, m4a, flac, etc.).
2. Choose your clip length and background and text colors.
3. Click **Generate brat video**.
4. Check the **Synced lyrics** box and fix any wrong words.
5. Click **Re-render** to apply your edits or try a new style. This does not re-transcribe, so it is fast.

Rendered videos are saved in `brat_outputs/`.

### Background options

| Style | Description |
|-------|-------------|
| Brat Green | The classic `#8ACE00` |
| Solid Color | 10 presets or any color from the picker |
| Rainbow Gradient | Animated flowing rainbow |
| Multicolor Gradient | Animated gradient using your chosen palette |
| Multicolor (changes on beat) | Solid color that changes every few beats |
| Assorted (new color each line) | A new color with a quick fade for each lyric line |

**Palettes** (for the multicolor and assorted modes): Brat Pop, Neon Party, Pastel Dream, Sunset, Rainbow.

### Text options

- **Text color modes:** single color (presets or picker), rainbow per word, or a rotating color per line.
- **Hook color:** its own picker, plus a toggle to turn hook highlighting off.
- **Style controls:** text size, blur amount, lowercase on/off, word-by-word reveal on/off.
- **Beat controls:** text pulse strength and background flash strength.

### Editing lyrics

Lyrics from a full song mix are never perfect, so you can edit them in the **Synced lyrics** box. Each line looks like this:

```
12.40 - 15.10 | i'm so brat and i'm so cool
38.20 - 41.00 | ★ club classics all night long
```

- Format is `start - end | text` (times in seconds, relative to the clip).
- Put a `★` before the text to mark a line as a hook.
- If you change the words in a line, the app spreads the word timing evenly across that line.
- If no lyrics are detected, you can type your own lines in this format and click **Re-render**.

### Settings

| Setting | Notes |
|---------|-------|
| Clip length | 10 to 90 seconds |
| Auto-pick hook | Uses the most energetic section. Turn it off to set a manual start time |
| Whisper model | `tiny`, `base`, `small` (default), `medium`. Bigger is more accurate but slower |
| Language | Auto-detect, or choose one (English, Tagalog, Spanish, Korean, Japanese, French, Portuguese, Indonesian) |
| Resolution | 720×1280 (faster, default) or 1080×1920 (TikTok HD) |

---

## 🛠️ Troubleshooting

**`ModuleNotFoundError` or "Missing packages"**
Make sure your virtual environment is activated, then re-run the `pip install` line for the tool you are using.

**First Brat Generator run is slow or seems stuck**
It is downloading the Whisper model. This only happens once per model size.

**Lyrics are wrong or missing**
Try a bigger Whisper model (`medium`), set the language manually, or edit the lyrics in the box and re-render. Heavy music over vocals makes transcription harder.

**`ffmpeg not found` or `ffmpeg failed`**
Run `pip install imageio-ffmpeg`, or install FFmpeg on your system and make sure it is on your PATH.

**Can't open an audio file**
Convert it to mp3 or wav and try again.

**The browser doesn't open for the Brat Generator**
Open `http://127.0.0.1:7860` manually. If the port is busy, close any other running Gradio apps.

**Video to GIF runs out of memory or the file is huge**
Lower `--fps`, set a smaller `--width`, shorten `--duration`, or reduce `--colors`.

**No file picker window appears (Video to GIF)**
The script falls back to asking you to paste the path. You can also pass the path directly as an argument.

---

## 📝 Notes

- Only use music and videos you have the rights to use. Check TikTok's rules for music before posting.
- Suggested `.gitignore`:
  ```
  venv/
  brat_outputs/
  __pycache__/
  *.gif
  *.mp4
  ```
