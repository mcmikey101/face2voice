from face2voice.models.SpeakerEncoder import SpeakerEncoder
from torchvision.transforms import ToTensor
from openvoice.api import ToneColorConverter
import torch
from openvoice import se_extractor

tone_color_converter = ToneColorConverter(f'checkpoints/tone_conv/config.json', device="cpu")
tone_color_converter.load_ckpt(f'checkpoints/tone_conv/checkpoint.pth')

speaker_encoder = SpeakerEncoder(config_path="checkpoints/tone_conv/config.json", ckpt_path="checkpoints/tone_conv/checkpoint.pth")
se_ext = se_extractor.get_se(audio_path="resources/example_reference.mp3", vc_model=tone_color_converter, save_pth=True, target_dir="outputs")

try:
    tgt_emb = speaker_encoder.encode_single(audio="outputs/example_reference_v2_rHQICiKpUG3_^AcmH/spec.pth", input="spec_tensor", return_numpy=True)
except RuntimeError:
    print("something worng")
src_emb = speaker_encoder.encode_single(audio="resources/example_reference.mp3", input="audio", return_numpy=True)

tgt_emb = torch.tensor(tgt_emb)
src_emb = torch.tensor(src_emb)

print(type(tgt_emb), type(src_emb))

speaker_encoder.tone_color_converter.convert(audio_src_path="resources/example_reference.mp3", src_se=src_emb, tgt_se=tgt_emb, output_path="outputs/testwspec.wav")
