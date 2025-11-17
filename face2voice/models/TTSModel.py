import torch
import torch.nn as nn
from typing import Union, Optional
import soundfile as sf
import wave
from piper import PiperVoice, SynthesisConfig


class TTSModel(nn.Module): 
    def __init__(
        self,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        language: str = 'ru'
    ):
        super().__init__()
        
        self.device = device
        self.language = language
        
        self._load_tts_components()
    
    def _load_tts_components(self):
        try:
            if self.language == "ru":
                self.voice = PiperVoice.load(r"face2voice\checkpoints\pipertts\ru\ru_RU-dmitri-medium.onnx")
            elif self.language == "en":
                self.voice = PiperVoice.load(r"face2voice\checkpoints\pipertts\en\en_US-amy-medium.onnx")
            elif self.language == "es":
                self.voice = PiperVoice.load(r"face2voice\checkpoints\pipertts\es\es_ES-davefx-medium.onnx")
            elif self.language == "fr":
                self.voice = PiperVoice.load(r"face2voice\checkpoints\pipertts\fr\fr_FR-siwis-medium.onnx")
            elif self.language == "zh":
                self.voice = PiperVoice.load(r"face2voice\checkpoints\pipertts\zh\zh_CN-huayan-medium.onnx")
            
        except Exception as e:
            print(f"Error loading PiperTTS: {e}")
            raise
    
    def synthesize(
        self,
        text: str,
        output_path: Optional[str] = None,
        return_tensor: bool = False,
        volume=1,
        length_scale=1.0,
        noise_scale=1.0,
        noise_w_scale=1.0,
        normalize_audio=False,
    ) -> Union[str, torch.Tensor]:
        
        syn_config = SynthesisConfig(
            volume=volume,  # half as loud
            length_scale=length_scale,  # twice as slow
            noise_scale=noise_scale,  # more audio variation
            noise_w_scale=noise_w_scale,  # more speaking variation
            normalize_audio=normalize_audio, # use raw audio from voice
        )

        print(f"Synthesizing base speech")
        
        with wave.open(output_path, "wb") as wav_file:
            self.voice.synthesize_wav(text, wav_file, syn_config=syn_config)
        
        print(f"Base audio generated: {output_path}")
        # Return tensor if requested
        if return_tensor:
            audio, sr = sf.read(output_path)
            audio_tensor = torch.from_numpy(audio).float()
            return audio_tensor