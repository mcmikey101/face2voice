import torch
import torch.nn as nn
from typing import Union, Optional
from TTS.api import TTS

class TTSModel(nn.Module): 
    def __init__(
        self,
        model_path: str,
        config_path: str,
        speakers_path: str,
        speaker: str,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ):
        super().__init__()
        
        self.device = device
        self.speaker = speaker

        try:
            self.tts_model = TTS(model_path=model_path, config_path=config_path, speakers_file_path=speakers_path)
            self.tts_model.to(self.device)
            
        except Exception as e:
            print(f"Error loading XTTS: {e}")
            raise
    
    def synthesize(
        self,
        text: str,
        output_path: Optional[str] = None,
        language="ru"
    ) -> Union[str, torch.Tensor]:
        
        print(f"Synthesizing base speech")
        
        self.tts_model.tts_to_file(text=text,
                                   file_path=output_path,
                                   speaker=self.speaker,
                                   language=language)
        
        print(f"Base audio generated: {output_path}")
