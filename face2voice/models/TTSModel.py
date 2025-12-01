import torch
import torch.nn as nn
from typing import Union, Optional

# Patch the weights_only issue
from TTS.tts.configs.xtts_config import XttsConfig
torch.serialization.add_safe_globals([XttsConfig])
_raw_load = torch.load

def _patched_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _raw_load(*args, **kwargs)

torch.load = _patched_load

# CRITICAL: Patch GPT2InferenceModel BEFORE importing TTS
import sys
from unittest.mock import MagicMock

# Create a mock module to patch the GPT issue
class PatchedGPT2:
    @staticmethod
    def patch():
        try:
            from TTS.tts.layers.xtts.gpt import GPT2InferenceModel
            
            # Add the missing generate method if it doesn't exist
            if not hasattr(GPT2InferenceModel, 'generate'):
                def generate(self, *args, **kwargs):
                    # Call the parent class generate or fallback
                    if hasattr(super(GPT2InferenceModel, self), 'generate'):
                        return super(GPT2InferenceModel, self).generate(*args, **kwargs)
                    else:
                        # Use the model's forward pass
                        return self(*args, **kwargs)
                
                GPT2InferenceModel.generate = generate
                print("✓ Patched GPT2InferenceModel.generate")
        except Exception as e:
            print(f"Warning: Could not patch GPT2InferenceModel: {e}")

# Apply patch
PatchedGPT2.patch()

# NOW import TTS
from TTS.api import TTS

class TTSModel(nn.Module): 
    def __init__(
        self,
        model_name,
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
            print("Loading XTTS model...")
            self.tts_model = TTS(
                model_name="tts_models/multilingual/multi-dataset/xtts_v2",
                progress_bar=True,
                gpu=(device == 'cuda')
            )
            print("✓ Model loaded successfully")
            
        except Exception as e:
            print(f"Error loading XTTS: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def synthesize(
        self,
        text: str,
        output_path: Optional[str] = None,
        language="ru"
    ) -> Union[str, torch.Tensor]:
        
        print(f"Synthesizing base speech")
        
        try:
            self.tts_model.tts_to_file(
                text=text,
                file_path=output_path,
                speaker=self.speaker,
                language=language
            )
            
            print(f"Base audio generated: {output_path}")
            
        except Exception as e:
            print(f"Error during synthesis: {e}")
            import traceback
            traceback.print_exc()
            raise