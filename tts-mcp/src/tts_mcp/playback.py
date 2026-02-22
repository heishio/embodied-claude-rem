"""Audio playback logic (engine-agnostic)."""

from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import time
import urllib.request
import wave
from datetime import datetime
from pathlib import Path
from typing import Iterator
from urllib.parse import quote

# カメラスピーカー再生時の末尾バッファリング途切れ対策（秒）
_SILENCE_PADDING_SECONDS = 0.3


def _append_silence_to_wav(wav_bytes: bytes, seconds: float) -> bytes:
    """WAVデータの末尾に無音を追加する。"""
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as r:
            params = r.getparams()
            frames = r.readframes(params.nframes)

        silence_frames = int(params.framerate * seconds) * params.nchannels * params.sampwidth
        silence = b"\x00" * silence_frames

        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setparams(params)
            w.writeframes(frames + silence)
        return buf.getvalue()
    except Exception:
        return wav_bytes


def save_audio(audio_bytes: bytes, audio_format: str, save_dir: str) -> str:
    """Save audio bytes to disk. Returns file path."""
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    file_path = os.path.join(save_dir, f"tts_{timestamp}.{audio_format}")
    with open(file_path, "wb") as f:
        f.write(audio_bytes)
    return file_path


# ---------------------------------------------------------------------------
# mpv streaming
# ---------------------------------------------------------------------------

def _build_mpv_env(
    pulse_sink: str | None, pulse_server: str | None,
) -> dict[str, str] | None:
    if not pulse_sink and not pulse_server:
        return None
    env = os.environ.copy()
    if pulse_sink:
        env["PULSE_SINK"] = pulse_sink
    if pulse_server:
        env["PULSE_SERVER"] = pulse_server
    return env


def _start_mpv(
    pulse_sink: str | None = None, pulse_server: str | None = None,
) -> subprocess.Popen:
    mpv = shutil.which("mpv")
    if not mpv:
        raise FileNotFoundError("mpv not found")
    env = _build_mpv_env(pulse_sink, pulse_server)
    return subprocess.Popen(
        [mpv, "--no-cache", "--no-terminal", "--", "fd://0"],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
    )


def stream_with_mpv(
    audio_stream: Iterator[bytes],
    pulse_sink: str | None = None,
    pulse_server: str | None = None,
) -> tuple[bytes, str]:
    """Stream audio chunks to mpv. Returns (collected_bytes, status)."""
    process = _start_mpv(pulse_sink, pulse_server)
    chunks: list[bytes] = []
    try:
        for chunk in audio_stream:
            chunks.append(chunk)
            process.stdin.write(chunk)
            process.stdin.flush()
    finally:
        process.stdin.close()
        process.wait()
    return b"".join(chunks), "streamed via mpv"


def stream_sentences_with_mpv(
    sentence_streams: list[tuple[str, Iterator[bytes]]],
    pulse_sink: str | None = None,
    pulse_server: str | None = None,
) -> tuple[bytes, str]:
    """Stream multiple sentence audio streams sequentially via mpv.

    Args:
        sentence_streams: List of (sentence_text, audio_iterator) tuples.

    Returns (collected_bytes, status).
    """
    all_chunks: list[bytes] = []
    for _sentence, audio_stream in sentence_streams:
        process = _start_mpv(pulse_sink, pulse_server)
        try:
            for chunk in audio_stream:
                all_chunks.append(chunk)
                process.stdin.write(chunk)
                process.stdin.flush()
        finally:
            process.stdin.close()
            process.wait()
    count = len(sentence_streams)
    return b"".join(all_chunks), f"streamed via mpv ({count} sentences)"


def can_stream() -> bool:
    """Check if mpv is available for streaming."""
    return shutil.which("mpv") is not None


# ---------------------------------------------------------------------------
# Local playback (file-based)
# ---------------------------------------------------------------------------

def _play_with_paplay(
    file_path: str, pulse_sink: str | None, pulse_server: str | None,
) -> tuple[bool, str]:
    paplay = shutil.which("paplay")
    if not paplay:
        return False, "paplay not available"

    wav_path = file_path
    if not file_path.lower().endswith((".wav", ".wave")):
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            return False, "paplay needs WAV (ffmpeg missing)"
        wav_path = str(Path(file_path).with_suffix(".wav"))
        result = subprocess.run(
            [ffmpeg, "-y", "-i", file_path, wav_path],
            check=False, capture_output=True, text=True,
        )
        if result.returncode != 0:
            error = result.stderr.strip() or result.stdout.strip()
            return False, f"paplay conversion failed: {error}"

    env = os.environ.copy()
    if pulse_sink:
        env["PULSE_SINK"] = pulse_sink
    if pulse_server:
        env["PULSE_SERVER"] = pulse_server
    result = subprocess.run(
        [paplay, wav_path], check=False, capture_output=True, text=True, env=env,
    )
    if result.returncode == 0:
        notes: list[str] = []
        if pulse_sink:
            notes.append(f"PULSE_SINK={pulse_sink}")
        if pulse_server:
            notes.append(f"PULSE_SERVER={pulse_server}")
        suffix = f" ({', '.join(notes)})" if notes else ""
        return True, f"played via paplay{suffix}"
    error = result.stderr.strip() or result.stdout.strip()
    return False, f"paplay failed: {error}"


def play_audio(
    audio_bytes: bytes,
    file_path: str,
    playback: str,
    pulse_sink: str | None,
    pulse_server: str | None,
) -> str:
    """Play audio locally using the best available player."""
    playback = (playback or "auto").strip().lower()
    last_error: str | None = None

    if playback in {"auto", "afplay"}:
        afplay = shutil.which("afplay")
        if afplay:
            result = subprocess.run(
                [afplay, file_path], check=False, capture_output=True, text=True,
            )
            if result.returncode == 0:
                return "played via afplay"
            error = result.stderr.strip() or result.stdout.strip()
            last_error = f"afplay failed: {error}"
            if playback == "afplay":
                return last_error
        else:
            last_error = "afplay not available"
            if playback == "afplay":
                return last_error

    if playback in {"auto", "paplay"}:
        ok, message = _play_with_paplay(file_path, pulse_sink, pulse_server)
        if ok:
            return message
        last_error = message
        if playback == "paplay":
            return message

    if playback in {"auto", "elevenlabs"}:
        try:
            from elevenlabs.play import play

            old_sink = os.environ.get("PULSE_SINK")
            old_server = os.environ.get("PULSE_SERVER")
            if pulse_sink:
                os.environ["PULSE_SINK"] = pulse_sink
            if pulse_server:
                os.environ["PULSE_SERVER"] = pulse_server
            try:
                play(audio_bytes)
            finally:
                if pulse_sink:
                    if old_sink is None:
                        os.environ.pop("PULSE_SINK", None)
                    else:
                        os.environ["PULSE_SINK"] = old_sink
                if pulse_server:
                    if old_server is None:
                        os.environ.pop("PULSE_SERVER", None)
                    else:
                        os.environ["PULSE_SERVER"] = old_server
            notes: list[str] = []
            if pulse_sink:
                notes.append(f"PULSE_SINK={pulse_sink}")
            if pulse_server:
                notes.append(f"PULSE_SERVER={pulse_server}")
            suffix = f" ({', '.join(notes)})" if notes else ""
            return f"played via elevenlabs{suffix}"
        except Exception as exc:  # noqa: BLE001
            last_error = f"elevenlabs play failed: {exc}"
            if playback == "elevenlabs":
                return last_error

    if playback in {"auto", "ffplay"}:
        ffplay = shutil.which("ffplay")
        if not ffplay:
            return f"playback skipped (no ffplay, last error: {last_error})"
        result = subprocess.run(
            [ffplay, "-nodisp", "-autoexit", "-loglevel", "error", file_path],
            check=False, capture_output=True, text=True,
        )
        if result.returncode == 0:
            return "played via ffplay"
        error = result.stderr.strip() or result.stdout.strip()
        return f"playback failed via ffplay: {error}"

    return f"playback skipped (unknown playback setting: {playback})"


# ---------------------------------------------------------------------------
# Camera speaker via go2rtc
# ---------------------------------------------------------------------------

def _send_audio_to_stream(
    abs_path: str,
    go2rtc_url: str,
    stream_name: str,
) -> tuple[bool, str]:
    """Send audio to a specific go2rtc stream. Returns (success, message)."""
    src = f"ffmpeg:{abs_path}#audio=pcma#input=file"
    url = (
        f"{go2rtc_url}/api/streams"
        f"?dst={quote(stream_name, safe='')}"
        f"&src={quote(src, safe='')}"
    )

    req = urllib.request.Request(url, method="POST", data=b"")
    with urllib.request.urlopen(req, timeout=10) as resp:
        body = json.loads(resp.read())

    has_sender = False
    for consumer in body.get("consumers", []):
        if consumer.get("senders"):
            has_sender = True
            break

    if not has_sender:
        return False, "no audio sender established"

    ffmpeg_producer_id = None
    for p in body.get("producers", []):
        if p.get("format_name") == "wav" or "ffmpeg" in p.get("source", ""):
            ffmpeg_producer_id = p.get("id")
            break

    if ffmpeg_producer_id:
        for _ in range(60):
            time.sleep(0.5)
            try:
                status_url = f"{go2rtc_url}/api/streams"
                with urllib.request.urlopen(status_url, timeout=5) as r:
                    streams = json.loads(r.read())
                stream = streams.get(stream_name, {})
                still_playing = any(
                    p.get("id") == ffmpeg_producer_id
                    for p in stream.get("producers", [])
                )
                if not still_playing:
                    break
            except Exception:
                break

    return True, f"played via go2rtc → {stream_name}"


def play_with_go2rtc(
    file_path: str,
    go2rtc_url: str,
    go2rtc_stream: str,
    go2rtc_ffmpeg: str,
) -> tuple[bool, str]:
    """Play audio through camera speaker via go2rtc backchannel."""
    try:
        abs_path = os.path.abspath(file_path)

        # WAVファイルなら末尾に無音パディングを追加（バッファリング途切れ対策）
        if abs_path.lower().endswith(".wav"):
            with open(abs_path, "rb") as f:
                original = f.read()
            padded = _append_silence_to_wav(original, _SILENCE_PADDING_SECONDS)
            if padded is not original:
                with open(abs_path, "wb") as f:
                    f.write(padded)

        # Try backchannel stream first (tapo:// only, bypasses RTSP backchannel)
        bc_stream = f"{go2rtc_stream}_bc"
        try:
            ok, msg = _send_audio_to_stream(abs_path, go2rtc_url, bc_stream)
            if ok:
                return True, msg
        except Exception:
            pass

        # Fall back to main stream (RTSP backchannel)
        ok, msg = _send_audio_to_stream(abs_path, go2rtc_url, go2rtc_stream)
        if ok:
            return True, msg
        return False, f"go2rtc: {msg} (camera may not support backchannel)"
    except Exception as exc:
        return False, f"go2rtc failed: {exc}"
