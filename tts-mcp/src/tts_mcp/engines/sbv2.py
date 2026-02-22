"""Style-Bert-VITS2 TTS engine."""

from __future__ import annotations

import logging
import urllib.parse
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)


class SBV2Engine:
    """Style-Bert-VITS2 TTS engine (local HTTP API)."""

    def __init__(
        self,
        url: str = "http://localhost:5000",
        model_id: int = 0,
        model_name: str | None = None,
        speaker_id: int = 0,
        style: str = "Neutral",
        style_weight: float = 5.0,
        length: float = 1.0,
        language: str = "JP",
    ) -> None:
        self._url = url.rstrip("/")
        self._model_id = model_id
        self._model_name = model_name
        self._speaker_id = speaker_id
        self._style = style
        self._style_weight = style_weight
        self._length = length
        self._language = language

    @property
    def engine_name(self) -> str:
        return "sbv2"

    def is_available(self) -> bool:
        """Check if SBV2 server is running."""
        try:
            req = urllib.request.Request(f"{self._url}/models/info", method="GET")
            with urllib.request.urlopen(req, timeout=3) as resp:
                resp.read()
            return True
        except Exception:
            return False

    def synthesize(self, text: str, **kwargs: Any) -> tuple[bytes, str]:
        """Synthesize text using Style-Bert-VITS2 API.

        Kwargs:
            model_name: Override model name (takes priority over model_id).
            model_id: Override model ID.
            speaker_id: Override speaker ID.
            style: Override style name.
            style_weight: Override style weight.
            length: Override speech rate (1.0 = normal).
            language: Override language (JP, EN, ZH).

        Returns:
            Tuple of (wav_bytes, 'wav').
        """
        params: dict[str, Any] = {"text": text}

        model_name = kwargs.get("model_name", self._model_name)
        if model_name:
            params["model_name"] = model_name
        else:
            params["model_id"] = kwargs.get("model_id", self._model_id)

        params["speaker_id"] = kwargs.get("speaker_id", self._speaker_id)
        params["style"] = kwargs.get("style", self._style)
        params["style_weight"] = kwargs.get("style_weight", self._style_weight)
        params["length"] = kwargs.get("length", self._length)
        params["language"] = kwargs.get("language", self._language)

        query = urllib.parse.urlencode(params)
        req = urllib.request.Request(
            f"{self._url}/voice?{query}",
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            wav_bytes = resp.read()

        return wav_bytes, "wav"
