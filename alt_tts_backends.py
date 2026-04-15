"""
Alternative TTS backends: Piper (ONNX) and MMS (VITS).

Provides on-device synthesis for languages not covered by Kokoro.
Each backend lazy-loads models on first use and caches them in memory.
"""

import logging
import numpy as np
from typing import Dict, Optional, Tuple, Any

logger = logging.getLogger('speak')


class FallbackTTSBackend:
    """Base class for alternative TTS synthesizers."""

    def synthesize(self, text: str) -> Tuple[np.ndarray, int]:
        """Return (float32_waveform, sample_rate) for the given text."""
        raise NotImplementedError

    @property
    def sample_rate(self) -> int:
        raise NotImplementedError

    @property
    def language_name(self) -> str:
        raise NotImplementedError

    def __repr__(self) -> str:
        cls = self.__class__.__name__
        lang = getattr(self, 'lang_code', '?')
        return f"<{cls} lang={lang}>"


class MMSTTSBackend(FallbackTTSBackend):
    """
    Meta MMS-TTS backend using VITS architecture.

    Covers low-resource languages (Quechua, Guarani, Aymara, etc.)
    with ~30 MB per language model at 16 kHz output.
    """

    SUPPORTED_LANGUAGES: Dict[str, Dict[str, Any]] = {
        'qu': {'model': 'facebook/mms-tts-que', 'name': 'Quechua', 'sr': 16000},
        'gn': {'model': 'facebook/mms-tts-grn', 'name': 'Guarani', 'sr': 16000},
        'ay': {'model': 'facebook/mms-tts-ayr', 'name': 'Aymara', 'sr': 16000},
        'sw': {'model': 'facebook/mms-tts-swh', 'name': 'Swahili', 'sr': 16000},
        'rw': {'model': 'facebook/mms-tts-kin', 'name': 'Kinyarwanda', 'sr': 16000},
        'ar': {'model': 'facebook/mms-tts-arb', 'name': 'Arabic', 'sr': 16000},
    }

    def __init__(self, lang_code: str):
        if lang_code not in self.SUPPORTED_LANGUAGES:
            raise ValueError(f"MMS does not support language '{lang_code}'.")

        self.lang_code = lang_code
        self.config = self.SUPPORTED_LANGUAGES[lang_code]
        self._model = None
        self._tokenizer = None

    def _ensure_loaded(self) -> None:
        """Load model and tokenizer on first use."""
        if self._model is not None:
            return

        try:
            from transformers import VitsModel, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                f"MMS requires 'transformers': pip install transformers. ({e})"
            )

        model_name = self.config['model']
        logger.debug(f"Loading MMS model: {model_name}")

        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = VitsModel.from_pretrained(model_name)
        self._model.eval()

    def synthesize(self, text: str) -> Tuple[np.ndarray, int]:
        """Synthesize text to a float32 waveform."""
        import torch
        self._ensure_loaded()

        inputs = self._tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            output = self._model(**inputs)

        waveform = output.waveform.squeeze().numpy().astype(np.float32)
        return waveform, self.config['sr']

    @property
    def sample_rate(self) -> int:
        return self.config['sr']

    @property
    def language_name(self) -> str:
        return self.config['name']

    @classmethod
    def supported_languages(cls) -> list:
        """Return list of supported language codes."""
        return list(cls.SUPPORTED_LANGUAGES.keys())


class PiperBackend(FallbackTTSBackend):
    """
    Piper TTS backend using ONNX-optimized VITS models.

    Covers Tier 2 languages (Arabic, Spanish, French, etc.)
    with ~60 MB per medium-quality voice at 22.05 kHz output.
    """

    SUPPORTED_LANGUAGES: Dict[str, Dict[str, Any]] = {
        'ar': {'model': 'ar_JO-kareem-medium', 'name': 'Arabic', 'sr': 22050},
        'es': {'model': 'es_ES-davefx-medium', 'name': 'Spanish', 'sr': 22050},
        'fr': {'model': 'fr_FR-siwis-medium', 'name': 'French', 'sr': 22050},
        'pt': {'model': 'pt_BR-faber-medium', 'name': 'Portuguese (BR)', 'sr': 22050},
        'hi': {'model': 'hi_IN-priyamvada-medium', 'name': 'Hindi', 'sr': 22050},
        'sw': {'model': 'sw_CD-lanfrica-medium', 'name': 'Swahili', 'sr': 22050},
        'zh': {'model': 'zh_CN-huayan-medium', 'name': 'Chinese (Mandarin)', 'sr': 22050},
    }

    def __init__(self, lang_code: str):
        if lang_code not in self.SUPPORTED_LANGUAGES:
            raise ValueError(f"Piper does not support language '{lang_code}'.")

        self.lang_code = lang_code
        self.config = self.SUPPORTED_LANGUAGES[lang_code]
        self._engine = None

    def _ensure_loaded(self) -> None:
        """Download model artifacts and load the Piper voice."""
        if self._engine is not None:
            return

        try:
            from piper import PiperVoice
        except ImportError as e:
            raise ImportError(
                f"Piper requires 'piper-tts': pip install piper-tts. ({e})"
            )

        model_name = self.config['model']
        logger.debug(f"Loading Piper model: {model_name}")
        onnx_file, json_file = self._resolve_model_artifacts(model_name)
        self._engine = PiperVoice.load(onnx_file, config_path=json_file)

    def _resolve_model_artifacts(self, model_name: str) -> Tuple[str, str]:
        """Download ONNX + JSON config from HuggingFace."""
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise ImportError(
                "Piper model download requires 'huggingface_hub': "
                "pip install huggingface_hub."
            )

        # model_name format: "ar_JO-kareem-medium"
        # path format:       "ar/ar_JO/kareem/medium/ar_JO-kareem-medium"
        parts = model_name.replace('-', '_').split('_')
        lang = parts[0]
        locale = f"{parts[0]}_{parts[1]}"
        voice = model_name.split('-')[1]
        quality = model_name.split('-')[2]

        base_path = f"{lang}/{locale}/{voice}/{quality}/{model_name}"
        repo_id = "rhasspy/piper-voices"

        onnx_path = hf_hub_download(repo_id=repo_id, filename=f"{base_path}.onnx")
        json_path = hf_hub_download(repo_id=repo_id, filename=f"{base_path}.onnx.json")
        return onnx_path, json_path

    def synthesize(self, text: str) -> Tuple[np.ndarray, int]:
        """Synthesize text to a float32 waveform."""
        self._ensure_loaded()

        raw_buffer = b"".join(
            chunk for chunk in self._engine.synthesize_stream_raw(text)
        )
        waveform = np.frombuffer(raw_buffer, dtype=np.int16).astype(np.float32) / 32768.0
        return waveform, self.config['sr']

    @property
    def sample_rate(self) -> int:
        return self.config['sr']

    @property
    def language_name(self) -> str:
        return self.config['name']

    @classmethod
    def supported_languages(cls) -> list:
        """Return list of supported language codes."""
        return list(cls.SUPPORTED_LANGUAGES.keys())


# Preferred backend order per language.
# 'primary' = Kokoro (or espeak fallback if Kokoro unavailable).
LANGUAGE_BACKEND_PREFERENCE: Dict[str, list] = {
    'en-us': ['primary'], 'en-gb': ['primary'],
    'es': ['primary', 'piper'], 'fr-fr': ['primary', 'piper'],
    'hi': ['primary', 'piper'], 'it': ['primary'],
    'pt-br': ['primary', 'piper'], 'ja': ['primary'],
    'zh': ['primary', 'piper'],
    'ar': ['piper', 'primary', 'mms'],
    'sw': ['piper', 'primary', 'mms'],
    'rw': ['mms', 'primary'], 'gn': ['mms', 'primary'],
    'qu': ['mms', 'primary'], 'ay': ['mms', 'primary'],
}


def get_tts_backend(
    lang_code: str,
    preferred_engine: Optional[str] = None
) -> Optional[FallbackTTSBackend]:
    """
    Pick the best available backend for the given language.

    Returns None when 'primary' (Kokoro/espeak) should be used.
    Tries each engine in preference order and skips unavailable ones.
    """
    if preferred_engine:
        preferences = [preferred_engine]
    else:
        preferences = LANGUAGE_BACKEND_PREFERENCE.get(lang_code, ['primary'])

    for engine in preferences:
        if engine == 'primary':
            return None

        if engine == 'mms' and lang_code in MMSTTSBackend.SUPPORTED_LANGUAGES:
            try:
                import torch  # noqa: F401
                from transformers import VitsModel, AutoTokenizer  # noqa: F401
                return MMSTTSBackend(lang_code)
            except (ImportError, Exception) as e:
                logger.warning(f"MMS unavailable for {lang_code}: {e}")
                continue

        if engine == 'piper' and lang_code in PiperBackend.SUPPORTED_LANGUAGES:
            try:
                from piper import PiperVoice  # noqa: F401
                return PiperBackend(lang_code)
            except (ImportError, Exception) as e:
                logger.warning(f"Piper unavailable for {lang_code}: {e}")
                continue

    return None
