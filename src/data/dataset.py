"""Dataset class for loading and preprocessing DDR data."""
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import librosa
import numpy as np
from tqdm import tqdm

from .parsers.sm_parser import SMParser, SongMetadata

class DDRDataset:
    """Dataset class for loading and preprocessing DDR data."""
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        split: str = 'train',
        audio_sr: int = 44100,
        hop_length: int = 512,
        n_mels: int = 128,
        measures_per_segment: int = 4,
        filter_mines: bool = True,
        filter_lifts: bool = True
    ):
        """Initialize the dataset.
        
        Args:
            data_dir: Path to data directory containing song folders
            split: One of 'train', 'val', or 'test'
            audio_sr: Audio sample rate
            hop_length: Hop length for mel spectrogram
            n_mels: Number of mel bands
            measures_per_segment: Number of measures per training segment
            filter_mines: Whether to filter out mines
            filter_lifts: Whether to filter out lifts (holds)
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.audio_sr = audio_sr
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.measures_per_segment = measures_per_segment
        self.filter_mines = filter_mines
        self.filter_lifts = filter_lifts
        
        # Load split
        split_file = self.data_dir.parent / 'splits' / f'{split}.txt'
        if split_file.exists():
            with open(split_file) as f:
                self.song_dirs = [self.data_dir / line.strip() for line in f]
        else:
            # If no split file exists, use all songs
            self.song_dirs = [p for p in self.data_dir.iterdir() if p.is_dir()]
            
        # Parse all songs
        self.songs: List[SongMetadata] = []
        self._load_songs()
    
    def _load_songs(self):
        """Load and parse all songs in the dataset."""
        parser = SMParser(self.data_dir)
        
        for song_dir in tqdm(self.song_dirs, desc=f'Loading {self.split} songs'):
            sm_files = list(song_dir.glob('*.sm'))
            if not sm_files:
                continue
                
            try:
                song = parser.parse_file(sm_files[0])
                if self._validate_song(song):
                    self.songs.append(song)
            except Exception as e:
                print(f'Error loading {song_dir}: {e}')
    
    def _validate_song(self, song: SongMetadata) -> bool:
        """Validate that a song has required files and charts."""
        if not song.music_file.exists():
            return False
        if not song.charts:
            return False
        return True
    
    def _load_audio_features(self, music_file: Path) -> np.ndarray:
        """Load and preprocess audio file into mel spectrogram."""
        y, _ = librosa.load(music_file, sr=self.audio_sr)
        mel_spec = librosa.feature.melspectrogram(
            y=y,
            sr=self.audio_sr,
            n_mels=self.n_mels,
            hop_length=self.hop_length
        )
        return librosa.power_to_db(mel_spec, ref=np.max)
    
    def _filter_steps(self, steps: List[Tuple[float, str]]) -> List[Tuple[float, str]]:
        """Apply filters to step patterns."""
        filtered = []
        for time, pattern in steps:
            if self.filter_mines and 'M' in pattern:
                continue
            if self.filter_lifts and any(c in pattern for c in '234'):
                continue
            filtered.append((time, pattern))
        return filtered
    
    def get_song_data(self, idx: int) -> Dict:
        """Get preprocessed data for a song.
        
        Returns:
            Dict containing:
                audio_features: Mel spectrogram array
                charts: Dict of chart_type -> {
                    difficulty: str,
                    meter: int,
                    steps: List of (time, pattern) pairs
                }
                metadata: Song metadata
        """
        song = self.songs[idx]
        audio_features = self._load_audio_features(song.music_file)
        
        charts = {}
        for chart_type, chart in song.charts.items():
            charts[chart_type] = {
                'difficulty': chart.difficulty,
                'meter': chart.meter,
                'steps': self._filter_steps(chart.steps)
            }
            
        return {
            'audio_features': audio_features,
            'charts': charts,
            'metadata': {
                'title': song.title,
                'artist': song.artist,
                'bpm': song.bpm
            }
        }
    
    def __len__(self) -> int:
        return len(self.songs)
    
    def __getitem__(self, idx: int) -> Dict:
        return self.get_song_data(idx)
    
    @staticmethod
    def create_splits(
        data_dir: Union[str, Path],
        splits: Dict[str, float],
        seed: int = 42,
        by_song: bool = True
    ):
        """Create train/val/test splits.
        
        Args:
            data_dir: Path to data directory
            splits: Dict of split_name -> fraction
            seed: Random seed
            by_song: If True, split by song, otherwise split by time
        """
        data_dir = Path(data_dir)
        splits_dir = data_dir.parent / 'splits'
        splits_dir.mkdir(exist_ok=True)
        
        # Get all song directories
        song_dirs = [p.relative_to(data_dir) for p in data_dir.iterdir() if p.is_dir()]
        
        if by_song:
            # Shuffle and split by song
            rng = np.random.RandomState(seed)
            rng.shuffle(song_dirs)
            
            start_idx = 0
            for split_name, fraction in splits.items():
                n_songs = int(len(song_dirs) * fraction)
                split_songs = song_dirs[start_idx:start_idx + n_songs]
                
                with open(splits_dir / f'{split_name}.txt', 'w') as f:
                    for song_dir in split_songs:
                        f.write(f'{song_dir}\n')
                        
                start_idx += n_songs
        else:
            # Sort by modification time and split
            song_dirs = sorted(
                song_dirs,
                key=lambda p: (data_dir / p).stat().st_mtime
            )
            
            start_idx = 0
            for split_name, fraction in splits.items():
                n_songs = int(len(song_dirs) * fraction)
                split_songs = song_dirs[start_idx:start_idx + n_songs]
                
                with open(splits_dir / f'{split_name}.txt', 'w') as f:
                    for song_dir in split_songs:
                        f.write(f'{song_dir}\n')
                        
                start_idx += n_songs 