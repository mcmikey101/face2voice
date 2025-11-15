"""
Hydra-based inference script for Face2Voice TTS.

Usage:
    # Single synthesis
    python infer_hydra.py inference_mode.mode=single
    
    # Voice cloning
    python infer_hydra.py inference_mode.mode=clone
    
    # Batch processing
    python infer_hydra.py inference_mode.mode=batch
    
    # Voice comparison
    python infer_hydra.py inference_mode.mode=compare
    
    # Custom text and face
    python infer_hydra.py \
        inference_mode.mode=single \
        inference_mode.single.text="Your custom text here" \
        inference_mode.single.face_image="path/to/face.jpg"
    
    # Use different checkpoint
    python infer_hydra.py \
        inference.face2voice_checkpoint=checkpoints/face2voice/epoch_50.pth
"""

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from pathlib import Path
import logging
from typing import List, Optional
import json
import numpy as np

from face2voice.metrics import TTSMetrics, TTSMetricsBatch

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="./configs", config_name="inference")
def main(cfg: DictConfig) -> None:
    """
    Main inference function with Hydra.
    
    Args:
        cfg: Hydra configuration
    """
    
    # Print configuration
    log.info("Inference Configuration:\n" + OmegaConf.to_yaml(cfg))
    
    # Instantiate Face2Voice TTS system
    log.info("Initializing Face2Voice TTS system...")
    tts = hydra.utils.instantiate(cfg.inference)
    
    log.info(f"✓ TTS system initialized")
    log.info(f"  Device: {cfg.experiment.device}")
    log.info(f"  Language: {cfg.inference.language}")
    log.info(f"  Speed: {cfg.inference.speed}")
    
    # Run inference based on mode
    mode = cfg.inference_mode.mode
    log.info(f"\nRunning inference mode: {mode}")
    log.info("=" * 60)
    
    if mode == "single":
        run_single_synthesis(tts, cfg)
    
    elif mode == "batch":
        run_batch_synthesis(tts, cfg)
    
    elif mode == "clone":
        run_voice_cloning(tts, cfg)
    
    elif mode == "compare":
        run_voice_comparison(tts, cfg)
    
    else:
        log.error(f"Unknown inference mode: {mode}")
        log.error("Available modes: single, batch, clone, compare")
        return
    
    log.info("=" * 60)
    log.info("✓ Inference complete!")


def run_single_synthesis(tts, cfg: DictConfig):
    """Run single speech synthesis."""
    log.info("Mode: Single Synthesis")
    
    config = cfg.inference_mode.single
    
    log.info(f"  Text: '{config.text}'")
    log.info(f"  Face image: {config.face_image}")
    log.info(f"  Output: {config.output_path}")
    
    # Create output directory
    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Synthesize
    audio_path = tts.synthesize(
        text=config.text,
        face_image=config.face_image,
        output_path=str(output_path),
        speaker='default'
    )
    
    log.info(f"\n✓ Generated audio: {audio_path}")
    
    # Post-process if configured
    if cfg.postprocess.normalize_audio or cfg.postprocess.trim_silence:
        audio_path = postprocess_audio(audio_path, cfg.postprocess)
        log.info(f"✓ Post-processed audio: {audio_path}")
    
    # Compute TTS metrics if reference audio is provided
    if hasattr(config, 'reference_audio') and config.reference_audio:
        compute_and_log_tts_metrics(audio_path, config.reference_audio, cfg)


def run_batch_synthesis(tts, cfg: DictConfig):
    """Run batch speech synthesis."""
    log.info("Mode: Batch Synthesis")
    
    config = cfg.inference_mode.batch
    
    # Load texts
    texts_file = Path(config.texts_file)
    if not texts_file.exists():
        log.error(f"Texts file not found: {texts_file}")
        return
    
    with open(texts_file, 'r') as f:
        texts = [line.strip() for line in f if line.strip()]
    
    log.info(f"  Loaded {len(texts)} texts from {texts_file}")
    
    # Load face images
    faces_dir = Path(config.faces_dir)
    if not faces_dir.exists():
        log.error(f"Faces directory not found: {faces_dir}")
        return
    
    face_images = sorted(list(faces_dir.glob("*.jpg")) + list(faces_dir.glob("*.png")))
    log.info(f"  Found {len(face_images)} face images in {faces_dir}")
    
    # Match texts and faces
    if len(texts) != len(face_images):
        log.warning(f"Number of texts ({len(texts)}) != number of faces ({len(face_images)})")
        min_len = min(len(texts), len(face_images))
        texts = texts[:min_len]
        face_images = face_images[:min_len]
        log.info(f"  Using {min_len} pairs")
    
    # Create output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate batch
    log.info(f"  Generating {len(texts)} audio files...")
    
    output_paths = tts.synthesize_batch(
        texts=texts,
        face_images=[str(f) for f in face_images],
        output_dir=str(output_dir),
        speaker='default'
    )
    
    # Save metadata
    metadata = {
        'num_samples': len(output_paths),
        'texts': texts,
        'face_images': [str(f) for f in face_images],
        'audio_files': output_paths
    }
    
    metadata_path = output_dir / "batch_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    log.info(f"\n✓ Generated {len(output_paths)} audio files")
    log.info(f"✓ Metadata saved: {metadata_path}")


def run_voice_cloning(tts, cfg: DictConfig):
    """Run voice cloning (one face, multiple texts)."""
    log.info("Mode: Voice Cloning")
    
    config = cfg.inference_mode.clone
    
    face_image = Path(config.face_image)
    if not face_image.exists():
        log.error(f"Face image not found: {face_image}")
        return
    
    texts = config.texts
    if not texts:
        log.error("No texts provided for voice cloning")
        return
    
    log.info(f"  Face image: {face_image}")
    log.info(f"  Number of texts: {len(texts)}")
    
    # Create output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Clone voice
    log.info(f"  Cloning voice...")
    
    output_paths = tts.clone_voice_from_face(
        face_image=str(face_image),
        texts=texts,
        output_dir=str(output_dir)
    )
    
    # Save metadata
    metadata = {
        'face_image': str(face_image),
        'num_samples': len(output_paths),
        'texts': texts,
        'audio_files': output_paths
    }
    
    metadata_path = output_dir / "clone_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    log.info(f"\n✓ Cloned voice into {len(output_paths)} samples")
    log.info(f"✓ Metadata saved: {metadata_path}")


def run_voice_comparison(tts, cfg: DictConfig):
    """Run voice comparison (same text, different faces)."""
    log.info("Mode: Voice Comparison")
    
    config = cfg.inference_mode.compare
    
    text = config.text
    face_images = [Path(f) for f in config.face_images]
    
    # Validate face images
    missing_faces = [f for f in face_images if not f.exists()]
    if missing_faces:
        log.error(f"Face images not found: {missing_faces}")
        return
    
    log.info(f"  Text: '{text}'")
    log.info(f"  Number of faces: {len(face_images)}")
    
    # Create output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate comparison samples
    log.info(f"  Generating comparison samples...")
    
    output_paths = []
    speaker_embeddings = []
    
    for i, face_image in enumerate(face_images):
        log.info(f"\n  Sample {i+1}/{len(face_images)}")
        log.info(f"    Face: {face_image.name}")
        
        output_path = output_dir / f"voice_{i+1}_{face_image.stem}.wav"
        
        # Extract speaker embedding
        speaker_emb = tts.extract_speaker_embedding(str(face_image))
        speaker_embeddings.append(speaker_emb.cpu().numpy().tolist())
        
        # Generate audio
        audio_path = tts.synthesize(
            text=text,
            face_image=str(face_image),
            output_path=str(output_path),
            speaker='default'
        )
        
        output_paths.append(audio_path)
        log.info(f"    ✓ Generated: {output_path.name}")
    
    # Compute embedding similarities if requested
    if config.get('generate_metadata', False):
        import numpy as np
        
        # Compute pairwise cosine similarities
        embeddings = torch.tensor(np.array(speaker_embeddings))
        embeddings_norm = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        similarity_matrix = torch.matmul(embeddings_norm, embeddings_norm.T)
        
        # Save detailed metadata
        metadata = {
            'text': text,
            'num_faces': len(face_images),
            'face_images': [str(f) for f in face_images],
            'audio_files': output_paths,
            'speaker_embeddings': speaker_embeddings,
            'similarity_matrix': similarity_matrix.tolist()
        }
        
        metadata_path = output_dir / "comparison_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        log.info(f"\n✓ Generated {len(output_paths)} comparison samples")
        log.info(f"✓ Metadata saved: {metadata_path}")
        
        # Print similarity matrix
        log.info("\nSpeaker Embedding Similarities:")
        for i in range(len(face_images)):
            for j in range(i+1, len(face_images)):
                sim = similarity_matrix[i, j].item()
                log.info(f"  {face_images[i].stem} <-> {face_images[j].stem}: {sim:.4f}")
    else:
        log.info(f"\n✓ Generated {len(output_paths)} comparison samples")


def postprocess_audio(audio_path: str, postprocess_cfg: DictConfig) -> str:
    """
    Post-process generated audio.
    
    Args:
        audio_path: Path to audio file
        postprocess_cfg: Post-processing configuration
    
    Returns:
        Path to processed audio file
    """
    import soundfile as sf
    import numpy as np
    
    # Load audio
    audio, sr = sf.read(audio_path)
    
    # Normalize audio
    if postprocess_cfg.normalize_audio:
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val * 0.95  # Normalize to -0.95 to 0.95
        log.info("  Applied normalization")
    
    # Trim silence
    if postprocess_cfg.trim_silence:
        # Simple energy-based trimming
        threshold = 0.01
        energy = np.abs(audio)
        
        # Find start
        start_idx = 0
        for i, val in enumerate(energy):
            if val > threshold:
                start_idx = max(0, i - int(0.1 * sr))  # Include 0.1s before
                break
        
        # Find end
        end_idx = len(audio)
        for i in range(len(energy) - 1, -1, -1):
            if energy[i] > threshold:
                end_idx = min(len(audio), i + int(0.1 * sr))  # Include 0.1s after
                break
        
        audio = audio[start_idx:end_idx]
        log.info(f"  Trimmed silence: {start_idx/sr:.2f}s to {end_idx/sr:.2f}s")
    
    # Resample if needed
    target_sr = postprocess_cfg.target_sample_rate
    if sr != target_sr:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
        log.info(f"  Resampled to {target_sr}Hz")
    
    # Save processed audio
    processed_path = Path(audio_path).parent / f"{Path(audio_path).stem}_processed.wav"
    sf.write(processed_path, audio, sr)
    
    return str(processed_path)


def compute_and_log_tts_metrics(
    generated_audio_path: str,
    reference_audio_path: str,
    cfg: DictConfig
):
    """
    Compute and log TTS metrics between generated and reference audio.
    
    Args:
        generated_audio_path: Path to generated audio file
        reference_audio_path: Path to reference audio file
        cfg: Configuration object
    """
    try:
        # Initialize metrics calculator
        sample_rate = getattr(cfg.postprocess, 'target_sample_rate', 24000)
        metrics_calc = TTSMetrics(sample_rate=sample_rate)
        
        # Compute metrics
        metrics = metrics_calc.compute_all_metrics(
            generated_audio=generated_audio_path,
            reference_audio=reference_audio_path
        )
        
        # Log metrics
        log.info("\n" + "="*60)
        log.info("TTS Metrics:")
        log.info("="*60)
        for metric_name, metric_value in metrics.items():
            if not np.isinf(metric_value):
                log.info(f"  {metric_name}: {metric_value:.4f}")
            else:
                log.info(f"  {metric_name}: N/A")
        log.info("="*60)
        
        # Save metrics to JSON if output path exists
        if hasattr(cfg.inference_mode.single, 'output_path'):
            output_path = Path(cfg.inference_mode.single.output_path)
            metrics_path = output_path.parent / f"{output_path.stem}_metrics.json"
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            log.info(f"✓ Metrics saved to: {metrics_path}")
        
    except Exception as e:
        log.warning(f"Failed to compute TTS metrics: {e}")


def create_demo_files():
    """Create demo text files for batch processing."""
    demo_texts = [
        "Welcome to face-to-voice text-to-speech synthesis.",
        "This system generates speech from facial images.",
        "Each face produces a unique voice with distinct characteristics.",
        "The technology uses deep learning to predict voice from appearance.",
        "Thank you for trying our face-to-voice demonstration."
    ]
    
    output_path = Path("data/batch_texts.txt")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for text in demo_texts:
            f.write(text + '\n')
    
    log.info(f"✓ Created demo texts file: {output_path}")


# ==================== CLI Helpers ====================

def list_available_checkpoints():
    """List available Face2Voice checkpoints."""
    checkpoint_dir = Path("checkpoints/face2voice")
    
    if not checkpoint_dir.exists():
        log.info("No checkpoints found in checkpoints/face2voice")
        return
    
    checkpoints = sorted(checkpoint_dir.glob("*.pth"))
    
    log.info("\nAvailable Face2Voice checkpoints:")
    log.info("-" * 60)
    
    for ckpt in checkpoints:
        # Load checkpoint info
        try:
            checkpoint = torch.load(ckpt, map_location='cpu')
            epoch = checkpoint.get('epoch', 'unknown')
            metrics = checkpoint.get('metrics', {})
            
            log.info(f"\n  {ckpt.name}")
            log.info(f"    Epoch: {epoch}")
            
            if 'val_cosine_similarity' in metrics:
                log.info(f"    Val Cosine Sim: {metrics['val_cosine_similarity']:.4f}")
            if 'val_loss' in metrics:
                log.info(f"    Val Loss: {metrics['val_loss']:.4f}")
        
        except Exception as e:
            log.info(f"\n  {ckpt.name}")
            log.info(f"    Error loading: {e}")
    
    log.info("\n" + "-" * 60)

# ==================== Main Entry Point ====================

if __name__ == "__main__":
    import sys
    
    # Handle special commands
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        
        if cmd == "list-checkpoints":
            list_available_checkpoints()
            sys.exit(0)
        
        elif cmd == "create-demo":
            create_demo_files()
            sys.exit(0)
        
        elif cmd == "help":
            print(__doc__)
            sys.exit(0)
    
    # Run main inference
    main()