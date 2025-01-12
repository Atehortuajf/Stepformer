"""StepMania (.sm/.ssc) file parser."""
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

@dataclass
class ChartMetadata:
    """Metadata for a single chart within a song."""
    difficulty: str
    meter: int
    groove_radar: Optional[Dict[str, float]]
    steps: List[Tuple[float, str]]  # (time_in_seconds, step_pattern)

@dataclass
class SongMetadata:
    """Metadata for a song and its associated charts."""
    title: str
    subtitle: Optional[str]
    artist: str
    credit: str
    music_file: Path
    offset: float
    bpm: float
    charts: Dict[str, ChartMetadata]  # type -> chart

class SMParser:
    """Parser for StepMania (.sm/.ssc) files."""
    
    def __init__(self, base_path: Path):
        """Initialize parser with base path for resolving relative paths."""
        self.base_path = Path(base_path)
        
    def parse_file(self, sm_path: Path) -> SongMetadata:
        """Parse a .sm/.ssc file and return song metadata with charts."""
        with open(sm_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Extract main metadata
        title = self._extract_tag(content, 'TITLE')
        subtitle = self._extract_tag(content, 'SUBTITLE')
        artist = self._extract_tag(content, 'ARTIST')
        credit = self._extract_tag(content, 'CREDIT')
        music = Path(self._extract_tag(content, 'MUSIC'))
        offset = float(self._extract_tag(content, 'OFFSET'))
        bpms = self._parse_bpms(self._extract_tag(content, 'BPMS'))
        
        # Create song metadata
        song = SongMetadata(
            title=title,
            subtitle=subtitle,
            artist=artist,
            credit=credit,
            music_file=self.base_path / music,
            offset=offset,
            bpm=bpms[0][1],  # Use first BPM as primary
            charts={}
        )
        
        # Parse each chart
        charts = self._extract_charts(content)
        for chart_type, chart_data in charts.items():
            difficulty = chart_data['difficulty']
            meter = int(chart_data['meter'])
            radar = chart_data.get('radar')
            steps = self._parse_steps(chart_data['notes'], bpms, offset)
            
            song.charts[chart_type] = ChartMetadata(
                difficulty=difficulty,
                meter=meter,
                groove_radar=radar,
                steps=steps
            )
        
        return song
    
    def _extract_tag(self, content: str, tag: str) -> str:
        """Extract value of a tag from the file content."""
        pattern = f'#{tag}:([^;]*);'
        match = re.search(pattern, content)
        if not match:
            raise ValueError(f'Required tag {tag} not found')
        return match.group(1).strip()
    
    def _parse_bpms(self, bpm_str: str) -> List[Tuple[float, float]]:
        """Parse BPM changes from BPM string."""
        bpms = []
        for bpm_change in bpm_str.strip().split(','):
            if not bpm_change.strip():
                continue
            beat, bpm = bpm_change.split('=')
            bpms.append((float(beat), float(bpm)))
        return sorted(bpms)
    
    def _extract_charts(self, content: str) -> Dict[str, Dict]:
        """Extract all charts from the file content."""
        charts = {}
        # Match chart sections
        pattern = r'#NOTES:\s*([^;]*);'
        for match in re.finditer(pattern, content):
            chart_data = match.group(1).strip().split(':')
            if len(chart_data) < 5:
                continue
                
            chart_type = chart_data[0].strip()
            difficulty = chart_data[2].strip()
            meter = chart_data[3].strip()
            notes = chart_data[4].strip()
            
            charts[chart_type] = {
                'difficulty': difficulty,
                'meter': meter,
                'notes': notes
            }
        
        return charts
    
    def _parse_steps(self, notes: str, bpms: List[Tuple[float, float]], offset: float) -> List[Tuple[float, str]]:
        """Parse step data into (time, pattern) pairs."""
        measures = notes.split(',')
        steps = []
        current_beat = 0.0
        current_bpm_idx = 0
        
        for measure in measures:
            rows = [row.strip() for row in measure.split('\n') if row.strip()]
            if not rows:
                continue
                
            # Each measure is 4 beats, divided into len(rows) parts
            beats_per_row = 4.0 / len(rows)
            
            for row in rows:
                # Convert beat to time
                while (current_bpm_idx < len(bpms) - 1 and 
                       current_beat >= bpms[current_bpm_idx + 1][0]):
                    current_bpm_idx += 1
                
                bpm = bpms[current_bpm_idx][1]
                time = (current_beat / bpm * 60.0) + offset
                
                if any(c != '0' for c in row):
                    steps.append((time, row))
                
                current_beat += beats_per_row
        
        return steps 