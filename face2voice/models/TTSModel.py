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

if __name__ == "__main__":
    tts = TTS(model_path=r"face2voice\checkpoints\xtts", config_path=r"face2voice\checkpoints\xtts\config.json", speakers_file_path=r"face2voice\checkpoints\xtts\speakers_xtts.pth")
    speakers = tts.speakers
    for speaker in speakers:
        speaker_name = speaker.replace(" ", "_")
        tts.tts_to_file(text="Радуга - атмосферное, оптическое и метеорологическое явление",
                                   file_path=rf"C:\Users\user\Desktop\projects\face2voice\outputs\xtts_speakers\{speaker_name}.wav",
                                   speaker=speaker,
                                   language="ru")