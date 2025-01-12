"""Script to test the data pipeline."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from src.data.dataset import DDRDataset

def plot_song_data(song_data, save_path=None):
    """Plot audio features and step patterns for a song."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8))
    
    # Plot mel spectrogram
    mel_spec = song_data['audio_features']
    ax1.imshow(mel_spec, aspect='auto', origin='lower')
    ax1.set_title('Mel Spectrogram')
    ax1.set_ylabel('Mel Band')
    
    # Plot step patterns
    for chart_type, chart in song_data['charts'].items():
        times = [t for t, _ in chart['steps']]
        # Convert times to spectrogram frames
        frames = [int(t * 44100 / 512) for t in times]  # sr/hop_length
        patterns = [p for _, p in chart['steps']]
        
        ax2.vlines(frames, 0, 1, label=f'{chart_type} ({chart["difficulty"]})')
    
    ax2.set_title('Step Patterns')
    ax2.set_xlabel('Frame')
    ax2.legend()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to data directory')
    parser.add_argument('--output_dir', type=str, default='plots',
                       help='Directory to save plots')
    parser.add_argument('--n_songs', type=int, default=5,
                       help='Number of songs to plot')
    args = parser.parse_args()
    
    # Create splits
    data_dir = Path(args.data_dir)
    DDRDataset.create_splits(
        data_dir,
        splits={'train': 0.8, 'val': 0.1, 'test': 0.1},
        by_song=True
    )
    
    # Load dataset
    dataset = DDRDataset(data_dir, split='train')
    print(f'Loaded {len(dataset)} songs')
    
    # Create plots
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    for i in tqdm(range(min(args.n_songs, len(dataset)))):
        song_data = dataset[i]
        plot_song_data(
            song_data,
            save_path=output_dir / f'song_{i}.png'
        )
        
        # Print song info
        print(f"\nSong {i}:")
        print(f"Title: {song_data['metadata']['title']}")
        print(f"Artist: {song_data['metadata']['artist']}")
        print(f"BPM: {song_data['metadata']['bpm']}")
        print("Charts:")
        for chart_type, chart in song_data['charts'].items():
            print(f"  {chart_type}: {chart['difficulty']} "
                  f"(level {chart['meter']}, {len(chart['steps'])} steps)")

if __name__ == '__main__':
    main() 