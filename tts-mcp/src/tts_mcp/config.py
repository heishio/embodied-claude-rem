"""Configuration for TTS MCP Server."""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _parse_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _detect_pulse_server() -> str | None:
    explicit = os.getenv("ELEVENLABS_PULSE_SERVER") or os.getenv("PULSE_SERVER")
    if explicit:
        return explicit
    wslg_socket = "/mnt/wslg/PulseServer"
    if os.path.exists(wslg_socket):
        return f"unix:{wslg_socket}"
    return None


@dataclass(frozen=True)
class ElevenLabsConfig:
    """ElevenLabs-specific configuration."""

    api_key: str
    voice_id: str
    model_id: str
    output_format: str

    @classmethod
    def from_env(cls) -> "ElevenLabsConfig | None":
        """Create config from environment variables. Returns None if not configured."""
        api_key = os.getenv("ELEVENLABS_API_KEY", "")
        if not api_key:
            return None
        return cls(
            api_key=api_key,
            voice_id=os.getenv("ELEVENLABS_VOICE_ID", "uYp2UUDeS74htH10iY2e"),
            model_id=os.getenv("ELEVENLABS_MODEL_ID", "eleven_v3"),
            output_format=os.getenv("ELEVENLABS_OUTPUT_FORMAT", "mp3_44100_128"),
        )


@dataclass(frozen=True)
class VoicevoxConfig:
    """VOICEVOX-specific configuration."""

    url: str
    speaker: int

    @classmethod
    def from_env(cls) -> "VoicevoxConfig | None":
        """Create config from environment variables. Returns None if not configured."""
        url = os.getenv("VOICEVOX_URL", "")
        if not url:
            return None
        return cls(
            url=url.rstrip("/"),
            speaker=int(os.getenv("VOICEVOX_SPEAKER", "3")),
        )


@dataclass(frozen=True)
class SBV2Config:
    """Style-Bert-VITS2-specific configuration."""

    url: str
    model_id: int
    model_name: str | None
    speaker_id: int
    style: str
    style_weight: float
    length: float
    language: str

    @classmethod
    def from_env(cls) -> "SBV2Config | None":
        """Create config from environment variables. Returns None if not configured."""
        url = os.getenv("SBV2_API_URL", "") or os.getenv("SBV2_URL", "")
        if not url:
            return None
        return cls(
            url=url.rstrip("/"),
            model_id=int(os.getenv("SBV2_MODEL_ID", "0")),
            model_name=os.getenv("SBV2_MODEL_NAME") or None,
            speaker_id=int(os.getenv("SBV2_SPEAKER_ID", "0")),
            style=os.getenv("SBV2_STYLE", "Neutral"),
            style_weight=float(os.getenv("SBV2_STYLE_WEIGHT", "5.0")),
            length=float(os.getenv("SBV2_LENGTH", "1.0")),
            language=os.getenv("SBV2_LANGUAGE", "JP"),
        )


@dataclass(frozen=True)
class PlaybackConfig:
    """Playback and go2rtc configuration (shared across engines)."""

    play_audio: bool
    save_dir: str
    playback: str
    pulse_sink: str | None
    pulse_server: str | None
    go2rtc_url: str | None
    go2rtc_stream: str
    go2rtc_ffmpeg: str
    go2rtc_bin: str | None
    go2rtc_config: str | None
    go2rtc_auto_start: bool
    go2rtc_camera_host: str | None
    go2rtc_camera_username: str | None
    go2rtc_camera_password: str | None

    @classmethod
    def from_env(cls) -> "PlaybackConfig":
        """Create config from environment variables."""
        return cls(
            play_audio=_parse_bool(
                os.getenv("TTS_PLAY_AUDIO") or os.getenv("ELEVENLABS_PLAY_AUDIO"), True,
            ),
            save_dir=os.getenv("TTS_SAVE_DIR")
            or os.getenv("ELEVENLABS_SAVE_DIR", "/tmp/tts-mcp"),
            playback=os.getenv("TTS_PLAYBACK")
            or os.getenv("ELEVENLABS_PLAYBACK", "auto"),
            pulse_sink=os.getenv("ELEVENLABS_PULSE_SINK") or None,
            pulse_server=_detect_pulse_server(),
            go2rtc_url=os.getenv("GO2RTC_URL") or None,
            go2rtc_stream=os.getenv("GO2RTC_STREAM", "tapo_cam"),
            go2rtc_ffmpeg=os.getenv("GO2RTC_FFMPEG", "ffmpeg"),
            go2rtc_bin=os.getenv("GO2RTC_BIN") or None,
            go2rtc_config=os.getenv("GO2RTC_CONFIG") or None,
            go2rtc_auto_start=_parse_bool(os.getenv("GO2RTC_AUTO_START"), True),
            go2rtc_camera_host=(
                os.getenv("GO2RTC_CAMERA_HOST")
                or os.getenv("TAPO_CAMERA_HOST")
                or None
            ),
            go2rtc_camera_username=(
                os.getenv("GO2RTC_CAMERA_USERNAME")
                or os.getenv("TAPO_USERNAME")
                or None
            ),
            go2rtc_camera_password=(
                os.getenv("GO2RTC_CAMERA_PASSWORD")
                or os.getenv("TAPO_PASSWORD")
                or None
            ),
        )


@dataclass(frozen=True)
class TTSConfig:
    """Top-level TTS configuration."""

    default_engine: str | None
    elevenlabs: ElevenLabsConfig | None
    voicevox: VoicevoxConfig | None
    sbv2: SBV2Config | None
    playback: PlaybackConfig

    @classmethod
    def from_env(cls) -> "TTSConfig":
        """Create config from environment variables."""
        return cls(
            default_engine=os.getenv("TTS_DEFAULT_ENGINE") or None,
            elevenlabs=ElevenLabsConfig.from_env(),
            voicevox=VoicevoxConfig.from_env(),
            sbv2=SBV2Config.from_env(),
            playback=PlaybackConfig.from_env(),
        )

    def resolve_engine(self, requested: str | None = None) -> str:
        """Resolve which engine to use.

        Priority:
        1. Explicit request (from tool call)
        2. TTS_DEFAULT_ENGINE env var
        3. Auto-detect (elevenlabs first for backward compat, then voicevox, then sbv2)
        """
        if requested:
            return requested
        if self.default_engine:
            return self.default_engine
        if self.elevenlabs:
            return "elevenlabs"
        if self.voicevox:
            return "voicevox"
        if self.sbv2:
            return "sbv2"
        raise ValueError(
            "No TTS engine configured. Set ELEVENLABS_API_KEY, VOICEVOX_URL, or SBV2_URL."
        )


@dataclass(frozen=True)
class ServerConfig:
    """MCP Server configuration."""

    name: str = "tts"
    version: str = "0.2.0"

    @classmethod
    def from_env(cls) -> "ServerConfig":
        """Create config from environment variables."""
        return cls(
            name=os.getenv("MCP_SERVER_NAME", "tts"),
            version=os.getenv("MCP_SERVER_VERSION", "0.2.0"),
        )
