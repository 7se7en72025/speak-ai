# Copyright (C) 2009, Aleksey Lim
# Copyright (C) 2019, Chihurumnaya Ibiam <ibiamchihurumnaya@sugarlabs.org>
# Copyright (C) 2026, Mebin J Thattil <mail@mebin.in>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

import time
import threading
import logging
from typing import Dict, List, Optional, Any, Callable
import numpy as np

import gi  # type: ignore
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib, GObject  # type: ignore

from sugar3.speech import GstSpeechPlayer  # type: ignore
from speech_utils.normalizer import normalize_text

import tts_cache
import alt_tts_backends

logger = logging.getLogger('speak')

try:
    from kokoro import KPipeline
    KOKORO_AVAILABLE = True
except ImportError:
    KOKORO_AVAILABLE = False
    logger.warning("Neural engines unavailable, falling back to espeak.")

PITCH_MIN, PITCH_MAX = 0, 200
RATE_MIN, RATE_MAX = 0, 200

# Voice prefixes describe Kokoro voice families. Keep these separate from the
# user-facing language codes that the UI and normalizer operate on.
VOICE_PREFIX_TO_LANG: Dict[str, str] = {
    'a': 'en',
    'b': 'en-gb',
    'e': 'es',
    'f': 'fr-fr',
    'h': 'hi',
    'i': 'it',
    'j': 'ja',
    'p': 'pt-br',
    'z': 'zh',
}

LANG_TO_KOKORO_CODE: Dict[str, str] = {
    'en': 'a',
    'en-us': 'a',
    'en-gb': 'b',
    'es': 'e',
    'fr': 'f',
    'fr-fr': 'f',
    'hi': 'h',
    'it': 'i',
    'ja': 'j',
    'pt': 'p',
    'pt-br': 'p',
    'zh': 'z',
    'ar': 'r',
    'sw': 's',
    'gn': 'g',
    'qu': 'q',
    'rw': 'w',
    'ay': 'y',
}


class Speech(GstSpeechPlayer):
    """
    Core speech controller integrating Kokoro, Piper, and MMS TTS backends
    with GStreamer audio pipelines and lip-sync event emission.
    """

    __gsignals__ = {
        'peak': (GObject.SIGNAL_RUN_FIRST, None, [GObject.TYPE_PYOBJECT]),
        'wave': (GObject.SIGNAL_RUN_FIRST, None, [GObject.TYPE_PYOBJECT]),
        'idle': (GObject.SIGNAL_RUN_FIRST, None, []),
    }

    def __init__(self) -> None:
        GstSpeechPlayer.__init__(self)
        self.pipeline: Optional[Gst.Pipeline] = None
        self.tts_cache = tts_cache.TTSCache()

        self.kokoro_pipeline = None
        self._kokoro_lang: str = 'a'
        self._kokoro_lock = threading.Lock()
        if KOKORO_AVAILABLE:
            t = threading.Thread(
                target=self._initialize_kokoro, daemon=True)
            t.start()

        self.kokoro_voices: List[str] = [
            'af_heart', 'af_alloy', 'af_aoede', 'af_bella', 'af_jessica', 'af_kore',
            'af_nicole', 'af_nova', 'af_river', 'af_sarah', 'af_sky', 'am_adam',
            'am_echo', 'am_eric', 'am_fenrir', 'am_liam', 'am_michael', 'am_onyx',
            'am_puck', 'am_santa', 'bf_alice', 'bf_emma', 'bf_isabella', 'bf_lily',
            'bm_daniel', 'bm_fable', 'bm_george', 'bm_lewis', 'jf_alpha',
            'jf_gongitsune', 'jf_nezumi', 'jf_tebukuro', 'jm_kumo', 'zf_xiaobei',
            'zf_xiaoni', 'zf_xiaoxiao', 'zf_xiaoyi', 'zm_yunjian', 'zm_yunxi',
            'zm_yunxia', 'zm_yunyang', 'ef_dora', 'em_alex', 'em_santa',
            'ff_siwis', 'hf_alpha', 'hf_beta', 'hm_omega', 'hm_psi',
            'if_sara', 'im_nicola', 'pf_dora', 'pm_alex', 'pm_santa'
        ]
        self.current_kokoro_voice: str = 'af_heart'
        self._requested_lang_code: str = 'en'
        self._cb_registry: Dict[str, Optional[int]] = {
            'peak': None, 'wave': None, 'idle': None
        }
        self.on_status_change: Optional[Callable] = None

    def _initialize_kokoro(self) -> None:
        """Load Kokoro model weights in a background thread."""
        with self._kokoro_lock:
            if self.kokoro_pipeline is None:
                self.kokoro_pipeline = KPipeline(
                    lang_code='a', repo_id='hexgrad/Kokoro-82M')
                self._kokoro_lang = 'a'

    def _normalize_lang_code(self, lang_code: Optional[str]) -> Optional[str]:
        if not lang_code:
            return None
        normalized = lang_code.strip().lower()
        if normalized == 'en':
            return 'en'
        if normalized == 'fr':
            return 'fr-fr'
        if normalized == 'pt':
            return 'pt-br'
        return normalized

    def _resolve_requested_lang(self, voice_name: str) -> str:
        """Determine the target language from the active persona or voice prefix."""
        import os
        import json

        explicit_lang = self._normalize_lang_code(
            getattr(self, '_requested_lang_code', None))
        if explicit_lang:
            return explicit_lang

        # 1. Look up the current persona from personas.json
        try:
            if voice_name:
                personas_path = os.path.join(
                    os.path.dirname(__file__), 'personas.json')
                with open(personas_path, 'r', encoding='utf-8') as f:
                    personas = json.load(f)
                for p_name, p_data in personas.items():
                    if p_data.get('voice') == voice_name and 'lang' in p_data:
                        return self._normalize_lang_code(p_data['lang']) or 'en'
        except Exception as e:
            logger.debug(f"Failed to load personas.json: {e}")

        # 2. Extract the Kokoro voice prefix
        try:
            if voice_name:
                prefix_char = voice_name[0]
                if prefix_char in VOICE_PREFIX_TO_LANG:
                    return VOICE_PREFIX_TO_LANG[prefix_char]
        except Exception:
            pass

        # 3. Safe default
        return "en"

    def _kokoro_code_for_lang(self, lang_code: str, voice_name: str) -> str:
        normalized = self._normalize_lang_code(lang_code) or 'en'
        if normalized in LANG_TO_KOKORO_CODE:
            return LANG_TO_KOKORO_CODE[normalized]
        if voice_name:
            return LANG_TO_KOKORO_CODE.get(
                VOICE_PREFIX_TO_LANG.get(voice_name[0], 'en'), 'a')
        return 'a'

    def _ensure_kokoro_lang(self, voice_name: str, lang_code: str) -> None:
        """Switch Kokoro G2P to the target language, reusing model weights."""
        required_lang = self._kokoro_code_for_lang(lang_code, voice_name)
        with self._kokoro_lock:
            if self.kokoro_pipeline is None or required_lang != self._kokoro_lang:
                logger.debug(
                    f"Switching Kokoro G2P: {self._kokoro_lang} -> {required_lang}")
                recycled_model = (
                    self.kokoro_pipeline.model if self.kokoro_pipeline else True)
                self.kokoro_pipeline = KPipeline(
                    lang_code=required_lang,
                    repo_id='hexgrad/Kokoro-82M',
                    model=recycled_model
                )
                self._kokoro_lang = required_lang

    def disconnect_all(self) -> None:
        """Detach all registered GTK signals."""
        for ev, hid in self._cb_registry.items():
            if hid is not None:
                self.disconnect(hid)
                self._cb_registry[ev] = None

    def connect_peak(self, cb: Callable) -> None:
        self._cb_registry['peak'] = self.connect('peak', cb)

    def connect_wave(self, cb: Callable) -> None:
        self._cb_registry['wave'] = self.connect('wave', cb)

    def connect_idle(self, cb: Callable) -> None:
        self._cb_registry['idle'] = self.connect('idle', cb)

    def set_kokoro_voice(self, voice_name: str) -> None:
        if not voice_name:
            return
        if voice_name in self.kokoro_voices:
            self.current_kokoro_voice = voice_name
        else:
            logger.warning(f"Unknown voice requested: {voice_name}")

    def get_available_kokoro_voices(self) -> List[str]:
        return self.kokoro_voices.copy()

    def get_default_kokoro_voices(self) -> List[str]:
        return ['af_heart', 'af_alloy', 'af_aoede']

    def get_addon_kokoro_voices(self) -> List[str]:
        defaults = set(self.get_default_kokoro_voices())
        return [v for v in self.kokoro_voices if v not in defaults]

    # ─── GStreamer Pipeline ──────────────────────────────────────────────

    def _build_pipeline_graph(self, use_appsrc: bool) -> str:
        """Build the GStreamer pipeline description string."""
        if use_appsrc:
            return (
                'appsrc name=source_input '
                '! audioconvert ! audioresample ! tee name=router '
                'router. ! queue ! autoaudiosink name=output_device '
                'router. ! queue ! audioconvert ! audioresample '
                '! audio/x-raw,format=S16LE,channels=1,rate=16000 '
                '! fakesink name=lipsync_sink'
            )
        return (
            'espeak name=espeak '
            '! capsfilter name=caps_constraint '
            '! tee name=router '
            'router. ! queue ! autoaudiosink name=output_device '
            'router. ! queue ! fakesink name=lipsync_sink'
        )

    def make_pipeline(self, use_appsrc: bool = False) -> None:
        """Create and configure the GStreamer pipeline."""
        if self.pipeline is not None:
            self.stop_sound_device()
            self.pipeline = None

        self.pipeline = Gst.parse_launch(
            self._build_pipeline_graph(use_appsrc))

        if not use_appsrc:
            caps = self.pipeline.get_by_name('caps_constraint')
            if caps:
                caps.set_property('caps', Gst.caps_from_string(
                    'audio/x-raw,channels=(int)1,depth=(int)16'))

        def _handoff_callback(element: Any, data: Gst.Buffer, pad: Any) -> bool:
            """Extract PCM frames for lip-sync signal emission."""
            size: int = data.get_size()
            if size == 0:
                return True

            duration = (
                data.duration
                if data.duration not in (0, Gst.CLOCK_TIME_NONE)
                else (size // 2) * Gst.SECOND // 16000
            )
            if duration <= 0:
                return True

            bpc = max(
                min(4096, size),
                size * 50000000 // max(duration, 1)
            ) // 2 * 2
            waves, peaks = [], []
            here = 0

            while here < size:
                try:
                    raw_bytes = data.extract_dup(here, bpc)
                    if not raw_bytes:
                        break
                    wave = np.frombuffer(raw_bytes, dtype='int16')
                    if wave.size == 0:
                        break
                    waves.append(wave)
                    peaks.append(np.max(np.abs(wave)))
                except Exception as e:
                    logger.debug(f"Buffer extraction stopped: {e}")
                    break
                here += bpc

            if not waves:
                return True

            interval_ms = max(25, int(duration / len(waves) / 1000000))

            def _dispatch_signals() -> bool:
                if waves:
                    self.emit("wave", waves.pop(0))
                    self.emit("peak", peaks.pop(0))
                    if waves:
                        GLib.timeout_add(interval_ms, _dispatch_signals)
                return False

            GLib.timeout_add(interval_ms, _dispatch_signals)
            return True

        lipsync_sink = self.pipeline.get_by_name('lipsync_sink')
        if lipsync_sink:
            lipsync_sink.props.signal_handoffs = True
            lipsync_sink.connect('handoff', _handoff_callback)

        def _bus_event(bus: Gst.Bus, message: Gst.Message) -> bool:
            if message.type in (Gst.MessageType.EOS, Gst.MessageType.ERROR):
                self.stop_sound_device()
            return True

        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect('message', _bus_event)

    def _push_buffer(self, appsrc: Gst.Element,
                     buffer_bytes: bytes, sr: int) -> bool:
        """Push raw audio bytes into the GStreamer appsrc element."""
        appsrc.set_property("caps", Gst.caps_from_string(
            f"audio/x-raw,format=F32LE,layout=interleaved,"
            f"rate={sr},channels=1"))
        buf = Gst.Buffer.new_wrapped(buffer_bytes)
        return appsrc.emit("push-buffer", buf) == Gst.FlowReturn.OK

    # ─── Backend Helpers ─────────────────────────────────────────────────

    def _backend_display_name(self, backend: Any) -> str:
        name = backend.__class__.__name__.replace('Backend', '')
        return name if name in ('MMS', 'Piper') else name

    def _espeak_voice_for_lang(self, status: Any, lang_code: str) -> str:
        normalized = self._normalize_lang_code(lang_code) or 'en'
        if normalized in ('en', 'en-us'):
            return getattr(status.voice, 'name', 'en')
        if normalized == 'en-gb':
            return 'en-gb'
        return normalized.split('-')[0]

    def _cache_voice_key(self, voice_name: str, lang: str,
                         backend: Optional[Any]) -> str:
        if backend is None:
            return voice_name
        return f"{self._backend_display_name(backend).lower()}:{lang}"

    # ─── Playback Methods ────────────────────────────────────────────────

    def _play_kokoro(self, text: str, voice: str,
                     lang: str, cache_key: str) -> None:
        try:
            appsrc = self.pipeline.get_by_name('source_input')
            if not appsrc:
                return

            with self._kokoro_lock:
                pipeline = self.kokoro_pipeline
            if pipeline is None:
                logger.error("Kokoro pipeline unavailable during playback.")
                return

            generator = pipeline(text, voice=voice)
            audio_array = []

            for _, _, chunk in generator:
                np_chunk = chunk.numpy()
                audio_array.append(np_chunk)
                if not self._push_buffer(appsrc, np_chunk.tobytes(), 24000):
                    break

            if audio_array:
                full_audio = np.concatenate(audio_array)
                self.tts_cache.put(
                    text, cache_key, lang, 1.0, full_audio, 24000)

            appsrc.emit("end-of-stream")
        except Exception as e:
            logger.error(f"Kokoro streaming error: {e}")

    def _play_alt_backend(self, text: str,
                          backend: alt_tts_backends.FallbackTTSBackend,
                          lang: str, cache_key: str) -> None:
        try:
            appsrc = self.pipeline.get_by_name('source_input')
            if not appsrc:
                return

            audio_np, sr = backend.synthesize(text)
            if not self._push_buffer(appsrc, audio_np.tobytes(), sr):
                return

            self.tts_cache.put(text, cache_key, lang, 1.0, audio_np, sr)
            appsrc.emit("end-of-stream")
        except Exception as e:
            logger.error(f"Alt backend error: {e}")
            # Try Kokoro as fallback
            fallback_voice = (
                getattr(self, 'current_kokoro_voice', None) or 'af_heart')
            if KOKORO_AVAILABLE:
                try:
                    self._ensure_kokoro_lang(fallback_voice, lang)
                    if getattr(self, 'on_status_change', None):
                        self.on_status_change(
                            ('Kokoro', lang), ('MISS', None))
                    self._play_kokoro(
                        text, fallback_voice, lang,
                        self._cache_voice_key(fallback_voice, lang, None))
                    return
                except Exception as kokoro_err:
                    logger.error(
                        f"Kokoro fallback also failed: {kokoro_err}")

    def _play_cached(self, audio_array: np.ndarray, sr: int) -> None:
        try:
            appsrc = self.pipeline.get_by_name('source_input')
            if appsrc:
                self._push_buffer(appsrc, audio_array.tobytes(), sr)
                appsrc.emit("end-of-stream")
        except Exception as e:
            logger.error(f"Cache playback error: {e}")

    def _play_espeak(self, status: Any, text: str,
                     lang_code: str) -> None:
        """Configure and trigger the espeak GStreamer element."""
        espeak_node = self.pipeline.get_by_name('espeak')
        if espeak_node:
            espeak_node.set_property('pitch', int(status.pitch) - 100)
            espeak_node.set_property('rate', int(status.rate) - 100)
            espeak_node.set_property(
                'voice', self._espeak_voice_for_lang(status, lang_code))
            espeak_node.set_property('track', 1)
            espeak_node.set_property('text', text)

    # ─── Main Entry Point ────────────────────────────────────────────────

    def speak(self, status: Any, text: str) -> None:
        """
        Route text to the best available TTS backend.
        Priority: Cache -> Alt backend -> Kokoro -> espeak-ng.
        """
        kokoro_voice = (
            getattr(self, 'current_kokoro_voice', None)
            or getattr(status.voice, 'name', None)
            or 'af_heart'
        )
        lang_code = self._resolve_requested_lang(kokoro_voice)
        norm_text = normalize_text(text, lang_code)
        backend = alt_tts_backends.get_tts_backend(lang_code)
        cache_key = self._cache_voice_key(kokoro_voice, lang_code, backend)

        t0 = time.perf_counter()
        cached_audio, sr = self.tts_cache.get(
            norm_text, cache_key, lang_code, 1.0)
        hit_ms = (time.perf_counter() - t0) * 1000
        expected_backend = (
            self._backend_display_name(backend) if backend is not None
            else ('Kokoro' if KOKORO_AVAILABLE else 'espeak-ng')
        )
        use_appsrc = (
            cached_audio is not None
            or backend is not None
            or KOKORO_AVAILABLE
        )

        self.stop_sound_device()
        self.make_pipeline(use_appsrc=use_appsrc)
        self.pipeline.set_state(Gst.State.PLAYING)

        # 1. Cache hit
        if cached_audio is not None:
            if getattr(self, 'on_status_change', None):
                self.on_status_change(
                    (expected_backend, lang_code),
                    ('HIT', f'{hit_ms:.2f}ms'))
            threading.Thread(
                target=self._play_cached,
                args=(cached_audio, sr), daemon=True).start()
            return

        # 2. Alt backend (Piper / MMS)
        if backend is not None:
            if getattr(self, 'on_status_change', None):
                self.on_status_change(
                    (self._backend_display_name(backend), lang_code),
                    ('MISS', None))
            threading.Thread(
                target=self._play_alt_backend,
                args=(norm_text, backend, lang_code, cache_key),
                daemon=True).start()
            return

        # 3. Kokoro
        if KOKORO_AVAILABLE:
            try:
                self._ensure_kokoro_lang(kokoro_voice, lang_code)
            except Exception as e:
                logger.error(
                    f"Kokoro setup failed, falling back to espeak: {e}")
                self.stop_sound_device()
                self.make_pipeline(use_appsrc=False)
                self.pipeline.set_state(Gst.State.PLAYING)
                if getattr(self, 'on_status_change', None):
                    self.on_status_change(
                        ('espeak-ng', lang_code), ('MISS', None))
                self._play_espeak(status, norm_text, lang_code)
                return

            if getattr(self, 'on_status_change', None):
                self.on_status_change(
                    ('Kokoro', lang_code), ('MISS', None))
            threading.Thread(
                target=self._play_kokoro,
                args=(norm_text, kokoro_voice, lang_code, cache_key),
                daemon=True).start()
        else:
            # 4. espeak-ng fallback
            if getattr(self, 'on_status_change', None):
                self.on_status_change(
                    ('espeak-ng', lang_code), ('MISS', None))
            self._play_espeak(status, norm_text, lang_code)

    def stop_sound_device(self) -> None:
        """Stop the active GStreamer pipeline."""
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)
        self.emit("idle")


_global_engine: Optional[Speech] = None


def get_speech() -> Speech:
    """Singleton factory for the Speech controller."""
    global _global_engine
    if _global_engine is None:
        _global_engine = Speech()
    return _global_engine
