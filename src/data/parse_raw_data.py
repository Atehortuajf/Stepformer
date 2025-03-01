import os
import sys
import json
import logging
import tempfile

# Set up logging for debugging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Import the audio encoder interface
from src.audio_encoders import AudioEncoder, DummyAudioEncoder, MT3Encoder

# Attempt to import a simfile parser if available
try:
    import simfile
except ImportError:
    simfile = None
    logger.warning("simfile library not installed. Using fallback dummy tokens.")

# Global encoder instance - will be initialized on first use
_audio_encoder = None

def get_audio_encoder() -> AudioEncoder:
    """Get the audio encoder instance, initializing if needed.
    
    This function tries to find the best available audio encoder.
    It first attempts to use MT3Encoder if available, then falls back to DummyAudioEncoder.
    
    Returns:
        An instance of AudioEncoder
    """
    global _audio_encoder
    if _audio_encoder is None:
        # Try to use MT3 encoder first
        mt3_encoder = MT3Encoder()
        if mt3_encoder.is_available():
            logger.info("Using MT3 encoder for audio transcription")
            _audio_encoder = mt3_encoder
        else:
            # Fall back to dummy encoder
            logger.info("MT3 encoder not available, using dummy encoder")
            _audio_encoder = DummyAudioEncoder()
    
    return _audio_encoder


def parse_simfile(simfile_path):
    """Parses a simfile to extract dance token information.
    Each token represents a full measure of steps from a chart, including timing, direction, and step type information.

    Args:
        simfile_path (str): Path to the simfile (.sm or .ssc file).

    Returns:
        list: A list of tokens representing dance steps by measure, with each token being a representation
             of all steps in that measure.
    """
    try:
        if simfile is not None:
            try:
                sf = simfile.open(simfile_path)
            except Exception as e:
                logger.error("Error loading simfile %s: %s", simfile_path, e)
                raise

            # Select the chart to use - typically choosing the most interesting difficulty
            if not sf.charts:
                logger.warning("No charts found in simfile %s", simfile_path)
                return [{'measure': 0, 'steps': []}]  # Return a single empty measure token
            
            # Choose the chart with the highest difficulty, preferring singles mode
            # Sort by stepstype (prefer dance-single) and then by difficulty
            chart = sorted(sf.charts, 
                          key=lambda c: (0 if c.stepstype == 'dance-single' else 1, 
                                         -1 * int(c.difficulty if hasattr(c, 'difficulty') and c.difficulty.isdigit() else 0)))[0]
            
            logger.info("Using chart: %s, difficulty: %s", 
                       chart.stepstype if hasattr(chart, 'stepstype') else "unknown", 
                       chart.difficulty if hasattr(chart, 'difficulty') else "unknown")
            
            # Process the notes to extract measures
            # Notes in simfiles are typically divided by measures using commas
            measures = chart.notes.strip().split(',')
            
            # Function to map step character to a step type
            def get_step_type(step_char):
                """Maps step character to step type.
                
                In simfiles:
                - '0': No step
                - '1': Normal step/tap
                - '2': Hold head
                - '3': Hold tail
                - '4': Roll head
                - 'M': Mine
                etc.
                """
                step_types = {
                    '0': 'none',     # No step
                    '1': 'tap',      # Normal step
                    '2': 'hold_start', # Beginning of a hold
                    '3': 'hold_end',   # End of a hold
                    '4': 'roll_start', # Beginning of a roll
                    'M': 'mine'      # Mine
                }
                return step_types.get(step_char, 'unknown')
            
            # Process each measure to create a token
            dance_tokens = []
            for measure_idx, measure in enumerate(measures):
                measure_token = {'measure': measure_idx, 'steps': []}
                
                # Process empty measures too
                if not measure.strip():
                    dance_tokens.append(measure_token)  # Empty measure: no steps
                    continue
                
                # Split the measure into individual steps (rows)
                steps = [step.strip() for step in measure.strip().split('\n') if step.strip()]
                
                # Calculate the number of rows for position calculation
                rows_in_measure = len(steps)
                
                for row_idx, step_row in enumerate(steps):
                    # A step row in DDR typically has 4 characters (Left, Down, Up, Right)
                    # '0' means no step, '1' means a regular step, other values for holds, etc.
                    
                    # Skip rows with all zeros (no steps)
                    if all(arrow == '0' for arrow in step_row):
                        continue
                    
                    # Calculate position within measure (0.0 to 1.0)
                    position = row_idx / rows_in_measure if rows_in_measure > 0 else 0
                    
                    # Create arrow step info for each direction
                    arrow_directions = ['left', 'down', 'up', 'right']
                    
                    # Only include directions with actual steps
                    steps_in_row = []
                    for direction_idx, arrow in enumerate(step_row[:4]):  # Limit to first 4 for dance-single
                        if arrow != '0':  # If not empty
                            step_info = {
                                'direction': arrow_directions[direction_idx],
                                'type': get_step_type(arrow),
                                'position': position
                            }
                            steps_in_row.append(step_info)
                    
                    if steps_in_row:
                        measure_token['steps'].extend(steps_in_row)
                
                dance_tokens.append(measure_token)
            
            return dance_tokens
        else:
            logger.warning("simfile library not available. Returning fallback dummy tokens for %s", simfile_path)
            # Return a structured dummy format that matches our real tokens
            return [{'measure': i, 'steps': [{'direction': 'left', 'type': 'tap', 'position': 0.5}]} for i in range(4)]
    except Exception as e:
        logger.exception("Failed to process simfile '%s': %s", simfile_path, e)
        raise


def transcribe_audio(audio_path):
    """Transcribes an audio file into a sequence of music tokens.
    
    This is a wrapper function that delegates to the active AudioEncoder instance.

    Args:
        audio_path (str): Path to the audio file.

    Returns:
        list: A list of music tokens.
    """
    logger.debug("Transcribing audio: %s", audio_path)
    encoder = get_audio_encoder()
    return encoder.transcribe(audio_path)


def process_song_folder(song_folder):
    """Processes a song folder by locating an audio file and a simfile, then parsing them into token sequences.

    Args:
        song_folder (str): Path to the folder containing a song's data.

    Returns:
        dict: A dictionary with track_id, music_tokens, and dance_tokens.

    Raises:
        FileNotFoundError: if required audio or simfile files are missing.
    """
    audio_file = None
    simfile_file = None
    for file in os.listdir(song_folder):
        lower_file = file.lower()
        if lower_file.endswith(('.mp3', '.ogg', '.wav')):
            audio_file = os.path.join(song_folder, file)
        elif lower_file.endswith(('.sm', '.ssc')):
            simfile_file = os.path.join(song_folder, file)

    if not audio_file:
        msg = f"Audio file missing in song folder: '{song_folder}'"
        logger.error(msg)
        raise FileNotFoundError(msg)
    if not simfile_file:
        msg = f"Simfile (.sm or .ssc) missing in song folder: '{song_folder}'"
        logger.error(msg)
        raise FileNotFoundError(msg)

    logger.debug("Processing folder: %s", song_folder)
    music_tokens = transcribe_audio(audio_file)
    dance_tokens = parse_simfile(simfile_file)

    track_id = os.path.basename(song_folder)
    return {"track_id": track_id, "music_tokens": music_tokens, "dance_tokens": dance_tokens}


def main(input_dir, output_file):
    """Walks through the raw data directory, processes each song, and saves the tokenized data to a JSON file.

    Args:
        input_dir (str): Root directory of raw data (containing song folders).
        output_file (str): Path to the output JSON file.
    """
    processed_data = []

    # Walk through the input directory; assume each song is in a subdirectory
    for root, dirs, files in os.walk(input_dir):
        # Only process directories that contain a simfile and an audio file
        if any(file.lower().endswith('.sm') for file in files) and any(file.lower().endswith(('.mp3', '.ogg', '.wav')) for file in files):
            try:
                song_data = process_song_folder(root)
                processed_data.append(song_data)
            except Exception as e:
                logger.exception("Error processing folder '%s':", root)

    # Save the processed data to the output JSON file
    with open(output_file, 'w') as f:
        json.dump(processed_data, f, indent=4)
    logger.info("Saved processed data for %d songs to %s", len(processed_data), output_file)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python parse_raw_data.py <input_raw_folder> <output_json_file>")
        sys.exit(1)
    input_dir = sys.argv[1]
    output_file = sys.argv[2]
    main(input_dir, output_file) 