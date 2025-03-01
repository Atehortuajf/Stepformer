#!/usr/bin/env python
# Add repository root to sys.path so that 'src' package can be imported
import os
import sys
import time
import argparse
import logging

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Import the audio encoders
from src.audio_encoders import AudioEncoder, DummyAudioEncoder, MT3Encoder

def has_note_seq():
    """Check if note_seq is available for visualization."""
    try:
        import note_seq
        return True
    except ImportError:
        return False

def visualize_tokens(tokens, title="MT3 Transcription Tokens", filename=None):
    """Visualize tokens as a bar chart."""
    plt.figure(figsize=(12, 6))
    
    # If we have a lot of tokens, show only a subset
    if len(tokens) > 200:
        # Show first and last 100 tokens
        first_chunk = tokens[:100]
        last_chunk = tokens[-100:]
        
        # Plot first chunk
        plt.subplot(2, 1, 1)
        plt.bar(range(len(first_chunk)), first_chunk, alpha=0.7)
        plt.title(f"{title} (First 100 Tokens)")
        plt.xlabel("Token Index")
        plt.ylabel("Token Value")
        
        # Plot last chunk
        plt.subplot(2, 1, 2)
        plt.bar(range(len(last_chunk)), last_chunk, alpha=0.7, color='orange')
        plt.title(f"{title} (Last 100 Tokens)")
        plt.xlabel("Token Index")
        plt.ylabel("Token Value")
    else:
        # Plot all tokens if we have a reasonable number
        plt.bar(range(len(tokens)), tokens, alpha=0.7)
        plt.title(title)
        plt.xlabel("Token Index")
        plt.ylabel("Token Value")
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename)
        logger.info(f"Visualization saved to {filename}")
    
    plt.show()

def visualize_note_sequence(note_sequence, title="MT3 Transcription", filename=None):
    """Visualize note sequence if note_seq is available."""
    if not has_note_seq():
        logger.warning("note_seq not available for visualization")
        return
    
    import note_seq
    
    plt.figure(figsize=(12, 6))
    note_seq.plot_sequence(note_sequence)
    plt.title(title)
    
    if filename:
        plt.savefig(filename)
        logger.info(f"Note sequence visualization saved to {filename}")
    
    plt.show()

def find_test_audio_file():
    """Find a test audio file in the repository."""
    # Try several common locations
    test_directories = [
        os.path.join(repo_root, 'data', 'raw'),
        os.path.join(repo_root, 'data', 'test'),
        os.path.join(repo_root, 'tests', 'data'),
        repo_root
    ]
    
    audio_extensions = ['.mp3', '.wav', '.ogg']
    
    for directory in test_directories:
        if os.path.exists(directory):
            for root, _, files in os.walk(directory):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in audio_extensions):
                        return os.path.join(root, file)
    
    return None

def test_dummy_encoder():
    """Test the DummyAudioEncoder."""
    logger.info("Testing DummyAudioEncoder...")
    
    encoder = DummyAudioEncoder()
    
    # The DummyAudioEncoder should always be available
    assert encoder.is_available(), "DummyAudioEncoder should always be available"
    
    # Find a test audio file
    test_file = find_test_audio_file()
    if test_file:
        logger.info(f"Using test audio file: {test_file}")
    else:
        # Create a dummy filename since DummyAudioEncoder doesn't actually use the file
        test_file = "dummy_audio.mp3"
        logger.warning(f"No test audio file found, using dummy filename: {test_file}")
    
    # The DummyAudioEncoder should return a fixed sequence of tokens
    tokens = encoder.transcribe(test_file)
    logger.info(f"DummyAudioEncoder returned {len(tokens)} tokens: {tokens}")
    
    assert len(tokens) > 0, "DummyAudioEncoder should return some tokens"
    
    # Visualize the tokens
    visualize_tokens(tokens, "DummyAudioEncoder Tokens", 
                     os.path.join(repo_root, 'outputs', 'plots', 'dummy_encoder_tokens.png'))
    
    logger.info("DummyAudioEncoder test completed successfully")
    return tokens

def test_mt3_encoder(model_type='mt3', checkpoint_path=None):
    """Test the MT3Encoder if available."""
    logger.info(f"Testing MT3Encoder with model_type={model_type}...")
    
    # Create the encoder
    encoder = MT3Encoder(model_type=model_type, checkpoint_path=checkpoint_path)
    
    # Check if MT3 is available
    is_available = encoder.is_available()
    logger.info(f"MT3Encoder available: {is_available}")
    
    if not is_available:
        logger.warning("MT3Encoder is not available; skipping test")
        return None, None
    
    # Find a test audio file
    test_file = find_test_audio_file()
    if not test_file:
        logger.error("No test audio file found; cannot test MT3Encoder")
        return None, None
    
    logger.info(f"Using test audio file: {test_file}")
    
    # Transcribe the audio file
    logger.info("Transcribing audio with MT3Encoder (this may take a while)...")
    start_time = time.time()
    tokens = encoder.transcribe(test_file)
    elapsed_time = time.time() - start_time
    logger.info(f"Transcription completed in {elapsed_time:.2f} seconds")
    
    if len(tokens) > 0:
        logger.info(f"MT3Encoder returned {len(tokens)} tokens")
        logger.info(f"First 10 tokens: {tokens[:10]}")
        if len(tokens) > 10:
            logger.info(f"Last 10 tokens: {tokens[-10:]}")
        
        # Visualize the tokens
        visualize_tokens(tokens, f"MT3Encoder Tokens ({model_type})", 
                         os.path.join(repo_root, 'outputs', 'plots', f'mt3_encoder_{model_type}_tokens.png'))
    else:
        logger.warning("MT3Encoder returned no tokens")
    
    # Get the note sequence if possible
    note_sequence = None
    try:
        logger.info("Getting note sequence from MT3Encoder...")
        note_sequence = encoder.get_note_sequence(test_file)
        
        if note_sequence:
            logger.info("Successfully retrieved note sequence")
            
            # Visualize the note sequence if note_seq is available
            if has_note_seq():
                visualize_note_sequence(note_sequence, f"MT3 Note Sequence ({model_type})", 
                                        os.path.join(repo_root, 'outputs', 'plots', f'mt3_note_sequence_{model_type}.png'))
            
            # Save MIDI file if note_seq is available
            if has_note_seq():
                import note_seq
                midi_path = os.path.join(repo_root, 'outputs', 'midi', f'mt3_transcription_{model_type}.mid')
                os.makedirs(os.path.dirname(midi_path), exist_ok=True)
                note_seq.sequence_proto_to_midi_file(note_sequence, midi_path)
                logger.info(f"Saved MIDI file to {midi_path}")
        else:
            logger.warning("Failed to get note sequence from MT3Encoder")
    except Exception as e:
        logger.exception(f"Error getting note sequence: {e}")
    
    logger.info("MT3Encoder test completed")
    return tokens, note_sequence

def main():
    """Run the MT3 encoder tests."""
    parser = argparse.ArgumentParser(description='Test MT3 audio encoding')
    parser.add_argument('--model-type', choices=['mt3', 'ismir2021'], default='mt3',
                        help='Model type to use (default: mt3)')
    parser.add_argument('--checkpoint-path', default=None,
                        help='Path to MT3 checkpoint directory')
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(os.path.join(repo_root, 'outputs', 'plots'), exist_ok=True)
    os.makedirs(os.path.join(repo_root, 'outputs', 'midi'), exist_ok=True)
    
    # Test DummyAudioEncoder
    dummy_tokens = test_dummy_encoder()
    
    # Test MT3Encoder
    mt3_tokens, note_sequence = test_mt3_encoder(
        model_type=args.model_type,
        checkpoint_path=args.checkpoint_path
    )
    
    logger.info("All tests completed")

if __name__ == '__main__':
    main()
