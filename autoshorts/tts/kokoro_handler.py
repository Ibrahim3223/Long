# -*- coding: utf-8 -*-
"""
Kokoro TTS Handler - Ultra-realistic voice synthesis
✅ kokoro-onnx paketi ile basit ve çalışır implementasyon
✅ 26 voice seçeneği (af_heart, af_bella, am_michael, vb.)
✅ Otomatik model indirme
"""
import os
import logging
import tempfile
import requests
from pathlib import Path
from typing import Dict, Any, Tuple, List
import wave
import struct

logger = logging.getLogger(__name__)

KOKORO_VOICES = {
    # Female American
    "af_heart": "af_heart",
    "af_bella": "af_bella",
    "af_sarah": "af_sarah",
    "af_sky": "af_sky",
    "af_alloy": "af_alloy",
    "af_aoede": "af_aoede",
    "af_jessica": "af_jessica",
    "af_kore": "af_kore",
    "af_nicole": "af_nicole",
    "af_nova": "af_nova",
    "af_river": "af_river",
    # Male American  
    "am_adam": "am_adam",
    "am_michael": "am_michael",
    "am_echo": "am_echo",
    "am_eric": "am_eric",
    "am_fenrir": "am_fenrir",
    "am_liam": "am_liam",
    "am_onyx": "am_onyx",
    "am_puck": "am_puck",
    "am_santa": "am_santa",
    # British Female
    "bf_alice": "bf_alice",
    "bf_emma": "bf_emma",
    "bf_isabella": "bf_isabella",
    "bf_lily": "bf_lily",
    # British Male
    "bm_daniel": "bm_daniel",
    "bm_fable": "bm_fable",
    "bm_george": "bm_george",
    "bm_lewis": "bm_lewis"
}

DEFAULT_VOICE = "af_sarah"


class KokoroTTS:
    """Kokoro TTS using kokoro-onnx package."""
    
    MODEL_BASE_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
    SAMPLE_RATE = 24000
    
    PRECISION_FILES = {
        "fp32": "kokoro-v1.0.onnx",
        "fp16": "kokoro-v1.0.fp16.onnx",
        "int8": "kokoro-v1.0.int8.onnx"
    }
    
    def __init__(self, voice: str = DEFAULT_VOICE, precision: str = "fp32"):
        self.voice = voice if voice in KOKORO_VOICES else DEFAULT_VOICE
        self.precision = precision
        self.kokoro = None
        self.cache_dir = Path.home() / ".cache" / "kokoro"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"🎤 Kokoro TTS initialized: voice={self.voice}, precision={self.precision}")
    
    def _download_file(self, url: str, dest: Path):
        """Download file with progress."""
        if dest.exists():
            logger.info(f"✓ Model already cached: {dest.name}")
            return
            
        logger.info(f"📥 Downloading {dest.name}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        with open(dest, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    progress = (downloaded / total_size) * 100
                    if downloaded % (1024 * 1024) == 0:  # Log every MB
                        logger.info(f"  {progress:.1f}% - {downloaded // (1024*1024)}MB / {total_size // (1024*1024)}MB")
        
        logger.info(f"✅ Downloaded: {dest.name}")
    
    def _ensure_models(self):
        """Download models if not cached."""
        model_file = self.PRECISION_FILES.get(self.precision, self.PRECISION_FILES["fp32"])
        
        model_path = self.cache_dir / model_file
        voices_path = self.cache_dir / "voices-v1.0.bin"
        
        # Download model
        if not model_path.exists():
            model_url = f"{self.MODEL_BASE_URL}/{model_file}"
            self._download_file(model_url, model_path)
        
        # Download voices
        if not voices_path.exists():
            voices_url = f"{self.MODEL_BASE_URL}/voices-v1.0.bin"
            self._download_file(voices_url, voices_path)
        
        return str(model_path), str(voices_path)
    
    def _load_model(self):
        """Lazy load Kokoro model."""
        if self.kokoro is not None:
            return
        
        try:
            from kokoro_onnx import Kokoro
            
            model_path, voices_path = self._ensure_models()
            
            logger.info(f"🔄 Loading Kokoro model: {Path(model_path).name}")
            self.kokoro = Kokoro(model_path, voices_path)
            logger.info(f"✅ Kokoro model loaded")
            
        except ImportError as e:
            raise ImportError(
                "Kokoro TTS requires: pip install kokoro-onnx soundfile\n"
                f"Error: {e}"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load Kokoro model: {e}")
    
    def generate(self, text: str) -> Dict[str, Any]:
        """Generate speech from text."""
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        self._load_model()
        text = text.strip()
        
        logger.info(f"🎙️ Generating Kokoro TTS: voice={self.voice}, {len(text)} chars")
        
        # Generate audio with kokoro-onnx
        samples, sample_rate = self.kokoro.create(
            text,
            voice=self.voice,
            speed=1.0,
            lang="en-us"
        )
        
        # Convert to WAV bytes
        wav_bytes = self._array_to_wav(samples, sample_rate)
        duration = len(samples) / sample_rate
        
        logger.info(f"✅ Kokoro TTS generated: {duration:.2f}s")
        
        return {
            'audio': wav_bytes,
            'duration': duration,
            'word_timings': []
        }
    
    def synthesize(self, text: str, wav_out: str) -> Tuple[float, List[Tuple[str, float]]]:
        """Synthesize and save to file."""
        result = self.generate(text)
        
        with open(wav_out, 'wb') as f:
            f.write(result['audio'])
        
        return result['duration'], result['word_timings']
    
    def _array_to_wav(self, samples, sample_rate: int) -> bytes:
        """Convert numpy array to WAV bytes."""
        import numpy as np
        
        # Ensure samples are in correct format
        if samples.dtype != np.int16:
            # Normalize and convert to int16
            samples = np.clip(samples, -1.0, 1.0)
            samples = (samples * 32767).astype(np.int16)
        
        # Create WAV in memory
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            with wave.open(tmp_path, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(samples.tobytes())
            
            with open(tmp_path, 'rb') as f:
                wav_bytes = f.read()
            
            return wav_bytes
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    @classmethod
    def list_voices(cls) -> List[str]:
        """Get list of available voices."""
        return list(KOKORO_VOICES.keys())
    
    @classmethod
    def get_voice_info(cls, voice: str) -> Dict[str, str]:
        """Get voice information."""
        voice_info = {
            "af_heart": {"name": "Heart", "gender": "female", "accent": "american"},
            "af_bella": {"name": "Bella", "gender": "female", "accent": "american"},
            "af_sarah": {"name": "Sarah", "gender": "female", "accent": "american"},
            "af_sky": {"name": "Sky", "gender": "female", "accent": "american"},
            "am_adam": {"name": "Adam", "gender": "male", "accent": "american"},
            "am_michael": {"name": "Michael", "gender": "male", "accent": "american"},
            "bf_emma": {"name": "Emma", "gender": "female", "accent": "british"},
            "bf_isabella": {"name": "Isabella", "gender": "female", "accent": "british"},
            "bm_daniel": {"name": "Daniel", "gender": "male", "accent": "british"},
        }
        return voice_info.get(voice, {"name": "Unknown", "gender": "unknown", "accent": "unknown"})
