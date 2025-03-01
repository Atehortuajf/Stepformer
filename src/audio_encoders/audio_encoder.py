from abc import ABC, abstractmethod
import numpy as np
from typing import List, Any, Dict, Optional, Union


class AudioEncoder(ABC):
    """Abstract interface for audio transcription.
    
    This class defines the common interface that all audio encoders should implement.
    Implementations can range from simple dummy encoders to complex models like MT3.
    """
    
    @abstractmethod
    def transcribe(self, audio_path: str) -> List[int]:
        """Transcribe audio into a sequence of music tokens.
        
        Args:
            audio_path: Path to the audio file to transcribe.
            
        Returns:
            A list of integers representing music tokens.
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if this encoder is available for use.
        
        This can be used to check if required dependencies are installed
        or if necessary models are available.
        
        Returns:
            True if the encoder is ready to use, False otherwise.
        """
        pass


class DummyAudioEncoder(AudioEncoder):
    """A basic implementation that returns dummy tokens without actual transcription.
    
    This is useful for testing or when audio transcription is not needed.
    """
    
    def transcribe(self, audio_path: str) -> List[int]:
        """Return a fixed sequence of dummy tokens without actual transcription.
        
        Args:
            audio_path: Path to the audio file (not actually used).
            
        Returns:
            A fixed list of dummy tokens.
        """
        return [10, 20, 30, 40]  # Fixed dummy tokens
    
    def is_available(self) -> bool:
        """Always returns True as this implementation has no dependencies.
        
        Returns:
            True
        """
        return True
