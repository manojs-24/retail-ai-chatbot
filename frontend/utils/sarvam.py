"""
Sarvam AI Integration Utilities
================================
Reusable helpers for Speech-to-Text, Text-to-Speech, and language detection
using the Sarvam AI API.

Patterns reused from Assessment/Assessment-2/app.py:
  - api-subscription-key header for STT
  - multipart/form-data file upload for STT
  - language_code field from STT response
  - Base64 audio decoding for TTS

Call flow in both chatbots:
  1.  stt_from_audio()        – audio bytes → (transcript, language_code)
  2.  detect_language()       – typed text  → language_code  (via Sarvam translate/detect)
  3.  translate_response()    – English response + target lang → translated text
  4.  tts_to_audio()          – text + language_code → wav bytes
"""

from __future__ import annotations

import base64
import logging
import os

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sarvam API endpoints (same keys as Assessment-2/app.py)
# ---------------------------------------------------------------------------
_STT_URL       = "https://api.sarvam.ai/speech-to-text"
_TRANSLATE_URL = "https://api.sarvam.ai/translate"
_TTS_URL       = "https://api.sarvam.ai/text-to-speech"

# Sarvam language codes → human-readable names
LANGUAGE_NAMES: dict[str, str] = {
    "en-IN": "English",
    "hi-IN": "Hindi",
    "ta-IN": "Tamil",
    "te-IN": "Telugu",
    "kn-IN": "Kannada",
    "ml-IN": "Malayalam",
    "mr-IN": "Marathi",
    "gu-IN": "Gujarati",
    "bn-IN": "Bengali",
    "pa-IN": "Punjabi",
    "od-IN": "Odia",
}

# Languages supported by the Sarvam TTS endpoint
_TTS_SUPPORTED = set(LANGUAGE_NAMES.keys())

# Default fallback
_DEFAULT_LANG = "en-IN"


def _get_key() -> str:
    """Return the Sarvam API subscription key from the environment."""
    key = os.environ.get("SARVAM_API_KEY", "").strip()
    if not key or key == "your_sarvam_api_key_here":
        raise EnvironmentError(
            "SARVAM_API_KEY is not set or still has the placeholder value. "
            "Add your real key to retail_ai/.env: SARVAM_API_KEY=<your_key>"
        )
    return key


def _auth_headers() -> dict[str, str]:
    """Return the subscription-key header used by all Sarvam endpoints."""
    return {"api-subscription-key": _get_key()}


# ---------------------------------------------------------------------------
# 1. Speech-to-Text
# ---------------------------------------------------------------------------

def stt_from_audio(audio_file) -> tuple[str, str]:
    """
    Convert a recorded audio object to text using Sarvam STT.

    Args:
        audio_file: The Streamlit ``st.audio_input`` object
                    (has ``.name``, ``.read()``, ``.type`` attributes).

    Returns:
        Tuple of (transcript: str, language_code: str).
        On failure returns ("", _DEFAULT_LANG).
    """
    try:
        files = {
            "file": (audio_file.name, audio_file, audio_file.type)
        }
        response = requests.post(
            _STT_URL,
            headers=_auth_headers(),
            files=files,
            timeout=30,
        )
        if response.status_code == 200:
            data = response.json()
            transcript    = data.get("transcript", "").strip()
            language_code = data.get("language_code", _DEFAULT_LANG)
            logger.info(
                "STT success — lang=%s transcript_len=%d",
                language_code, len(transcript),
            )
            return transcript, language_code
        else:
            logger.warning("STT API error %d: %s", response.status_code, response.text[:200])
            return "", _DEFAULT_LANG
    except Exception as exc:  # noqa: BLE001
        logger.warning("STT request failed: %s", exc)
        return "", _DEFAULT_LANG


# ---------------------------------------------------------------------------
# 2. Translate query → English + detect source language (single API call)
# ---------------------------------------------------------------------------

def translate_to_english(text: str) -> tuple[str, str]:
    """
    Translate *text* into English and detect its source language in one call.

    The Sarvam /translate endpoint with ``source_language_code="auto"``
    returns both the English translation AND ``source_language_code``.
    We reuse this so callers get the English text to pass to the backend
    AND the original language code to translate the response back later.

    Args:
        text: Input text in any supported language.

    Returns:
        Tuple of (english_text: str, source_language_code: str).
        Falls back to (original text, "en-IN") on any error so the
        pipeline always receives something usable.
    """
    if not text or not text.strip():
        return text, _DEFAULT_LANG
    try:
        key = _get_key()  # raises EnvironmentError if key is missing/placeholder
    except EnvironmentError as exc:
        logger.error("translate_to_english — %s", exc)
        # Re-raise so the UI catches it and shows an actionable error
        raise

    try:
        payload = {
            "input":                text[:500],
            "source_language_code": "auto",
            "target_language_code": "en-IN",
            "speaker_gender":       "Male",
            "mode":                 "formal",
            "model":                "mayura:v1",
            "enable_preprocessing": False,
        }
        response = requests.post(
            _TRANSLATE_URL,
            headers={"api-subscription-key": key, "Content-Type": "application/json"},
            json=payload,
            timeout=15,
        )
        logger.info(
            "translate_to_english RAW — status=%d body=%s",
            response.status_code,
            response.text[:400],
        )
        if response.status_code == 200:
            data         = response.json()
            english_text = data.get("translated_text", text) or text
            lang_code    = data.get("source_language_code", _DEFAULT_LANG) or _DEFAULT_LANG
            logger.info(
                "translate_to_english OK — detected=%s en=%r",
                lang_code, english_text[:80],
            )
            return english_text, lang_code
        # Non-200: log full body so we can see exactly what Sarvam said
        logger.error(
            "translate_to_english API %d: %s", response.status_code, response.text[:400]
        )
        return text, _DEFAULT_LANG
    except EnvironmentError:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("translate_to_english request failed: %s", exc)
        return text, _DEFAULT_LANG


def detect_language(text: str) -> str:
    """
    Detect the language of *text*.
    Thin wrapper around :func:`translate_to_english` — discards the translation.
    """
    _, lang_code = translate_to_english(text)
    return lang_code


# ---------------------------------------------------------------------------
# 3. Translate English response → target language
# ---------------------------------------------------------------------------

def translate_response(text: str, target_language_code: str) -> str:
    """
    Translate *text* (expected to be English) into *target_language_code*.

    Args:
        text:                 English text to translate.
        target_language_code: Sarvam language code, e.g. ``"hi-IN"``.

    Returns:
        Translated text string. Falls back to the original *text* on error.
    """
    if target_language_code == _DEFAULT_LANG or not target_language_code:
        return text
    if target_language_code not in LANGUAGE_NAMES:
        logger.warning("Unsupported language %s — returning English", target_language_code)
        return text
    try:
        payload = {
            "input":                text[:5000],
            "source_language_code": "en-IN",
            "target_language_code": target_language_code,
            "speaker_gender":       "Male",
            "mode":                 "formal",
            "model":                "mayura:v1",
            "enable_preprocessing": False,
        }
        response = requests.post(
            _TRANSLATE_URL,
            headers={**_auth_headers(), "Content-Type": "application/json"},
            json=payload,
            timeout=20,
        )
        if response.status_code == 200:
            translated = response.json().get("translated_text", text)
            logger.info(
                "Translated response → %s (len=%d)",
                target_language_code, len(translated),
            )
            return translated or text
        logger.warning("Translate API error %d: %s", response.status_code, response.text[:200])
        return text
    except Exception as exc:  # noqa: BLE001
        logger.warning("Translation failed: %s", exc)
        return text


# ---------------------------------------------------------------------------
# 4. Text-to-Speech
# ---------------------------------------------------------------------------

def tts_to_audio(text: str, language_code: str) -> bytes | None:
    """
    Convert *text* to speech in *language_code* using Sarvam TTS.

    Args:
        text:          Text to synthesise (max ~500 chars per call).
        language_code: Sarvam language code, e.g. ``"hi-IN"``.

    Returns:
        WAV audio bytes, or ``None`` on failure.
    """
    if language_code not in _TTS_SUPPORTED:
        language_code = _DEFAULT_LANG
    try:
        payload = {
            "inputs":               [text[:500]],  # TTS endpoint requires a list
            "target_language_code": language_code,
            "speaker":              "meera",
            "pitch":                0,
            "pace":                 1.0,
            "loudness":             1.5,
            "speech_sample_rate":   8000,
            "enable_preprocessing": True,
            "model":                "bulbul:v1",
        }
        response = requests.post(
            _TTS_URL,
            headers={**_auth_headers(), "Content-Type": "application/json"},
            json=payload,
            timeout=30,
        )
        if response.status_code == 200:
            data = response.json()
            # Sarvam TTS returns either "audios" (list) or "audio" (string)
            audio_b64 = None
            if "audios" in data and data["audios"]:
                audio_b64 = data["audios"][0]
            elif "audio" in data and data["audio"]:
                audio_b64 = data["audio"]
            if audio_b64:
                audio_bytes = base64.b64decode(audio_b64)
                logger.info(
                    "TTS success — lang=%s bytes=%d", language_code, len(audio_bytes)
                )
                return audio_bytes
            logger.warning("TTS response had no audio data: %s", list(data.keys()))
        else:
            logger.warning("TTS API error %d: %s", response.status_code, response.text[:200])
        return None
    except Exception as exc:  # noqa: BLE001
        logger.warning("TTS request failed: %s", exc)
        return None
