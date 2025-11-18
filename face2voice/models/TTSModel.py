import torch
import torch.nn as nn
from typing import Union, Optional
import soundfile as sf
from TTS.api import TTS

class TTSModel(nn.Module): 
    def __init__(
        self,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        model_path: str = None
    ):
        super().__init__()
        
        self.device = device
        self.model_path = model_path
        
        self._load_tts_components()
    
    def _load_tts_components(self):
        try:
            self.tts_model = TTS(self.model_path)
            self.tts_model.to(self.device)
            
        except Exception as e:
            print(f"Error loading XTTS: {e}")
            raise
    
    def synthesize(
        self,
        text: str,
        output_path: Optional[str] = None,
        return_tensor: bool = False,
        language="ru"
    ) -> Union[str, torch.Tensor]:
        
        print(f"Synthesizing base speech")
        
        self.tts_model.tts_to_file(text=text,
                                   file_path=output_path,
                                   language=language)
        
        print(f"Base audio generated: {output_path}")
        # Return tensor if requested
        if return_tensor:
            audio, sr = sf.read(output_path)
            audio_tensor = torch.from_numpy(audio).float()
            return audio_tensor