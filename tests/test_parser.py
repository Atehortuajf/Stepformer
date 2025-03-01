# Add repository root to sys.path so that 'src' package can be imported
import os
import sys
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import logging
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import datetime
import json

# Ensure the logs directory exists
if not os.path.exists('logs'):
    os.makedirs('logs')

log_filename = os.path.join('logs', f"test_parser_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Import the process_song_folder from parse_raw_data in src/data
from src.data.parse_raw_data import process_song_folder


def main(song_folder):
    if not os.path.isdir(song_folder):
        logger.error('Provided path is not a directory: %s', song_folder)
        sys.exit(1)
    try:
        result = process_song_folder(song_folder)
        print('Parsed song data:')
        print(json.dumps(result, indent=2))  # Pretty print the result

        # Visualization of one measure/chart pair: using music and dance tokens
        music_tokens = result.get('music_tokens', [])
        dance_tokens = result.get('dance_tokens', [])
        if not music_tokens or not dance_tokens:
            logger.error('Parsed data does not contain valid music or dance tokens.')
            sys.exit(1)

        # Ensure outputs/plots directory exists
        output_dir = os.path.join('outputs', 'plots')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Create figure with subplots
        plt.figure(figsize=(12, 10))
        
        # VISUALIZATION 1: Music tokens bar chart (simple)
        plt.subplot(3, 1, 1)
        if isinstance(music_tokens[0], (int, float)):  # Simple numeric tokens
            plt.bar(range(len(music_tokens)), music_tokens, color='blue')
            plt.title('Music Tokens')
            plt.ylabel('Token Value')
            plt.xlabel('Token Index')
        else:
            plt.text(0.5, 0.5, "Complex music token format - see console for details", 
                     ha='center', va='center', transform=plt.gca().transAxes)
            plt.axis('off')
        
        # VISUALIZATION 2: Dance steps - focus on a few measures
        plt.subplot(3, 1, 2)
        
        # Define measure range to display (first few measures with steps)
        measures_with_steps = [m['measure'] for m in dance_tokens if len(m['steps']) > 0][:4]  # First 4 measures with steps
        if not measures_with_steps:
            measures_with_steps = [0, 1, 2, 3]  # Fallback if no measures have steps
        
        # Filter steps for selected measures only
        filtered_steps = {}  # Dictionary to organize steps by measure
        for measure_token in dance_tokens:
            measure_num = measure_token['measure']
            if measure_num in measures_with_steps:
                if measure_num not in filtered_steps:
                    filtered_steps[measure_num] = []
                filtered_steps[measure_num].extend(measure_token['steps'])
        
        # Create a subplot for each selected measure
        num_measures = len(measures_with_steps)
        for i, measure_num in enumerate(sorted(filtered_steps.keys())):
            ax = plt.subplot(3, num_measures, num_measures + i + 1)
            
            # Directions map to y positions
            direction_to_y = {
                'left': 0,
                'down': 1,
                'up': 2,
                'right': 3
            }
            
            # Colors for step types
            type_colors = {
                'tap': 'blue',
                'hold_start': 'green',
                'hold_end': 'darkgreen',
                'roll_start': 'orange',
                'mine': 'red',
                'unknown': 'gray'
            }
            
            # Extract x, y, and colors
            x_vals = []  # Position within measure
            y_vals = []  # Direction 
            c_vals = []  # Colors based on step type
            
            for step in filtered_steps[measure_num]:
                x_vals.append(step['position'])
                y_vals.append(direction_to_y.get(step['direction'], -1))
                c_vals.append(type_colors.get(step['type'], 'black'))
            
            # Plot steps for this measure
            plt.scatter(x_vals, y_vals, c=c_vals, alpha=0.8, s=80, marker='s')
            
            # Set up the axes
            plt.title(f"Measure {measure_num}")
            plt.xlabel("Position in Measure")
            plt.xlim(-0.05, 1.05)
            plt.yticks([0, 1, 2, 3], ['Left', 'Down', 'Up', 'Right'])
            plt.ylim(-0.5, 3.5)
            plt.grid(True, alpha=0.3)
            
            # Create legend for first subplot only to avoid repetition
            if i == 0:
                legend_elements = [
                    mpatches.Patch(color=color, label=step_type)
                    for step_type, color in type_colors.items()
                    if any(step['type'] == step_type for step in filtered_steps[measure_num])
                ]
                if legend_elements:
                    plt.legend(handles=legend_elements, loc='upper center', 
                              bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize='small')
        
        # VISUALIZATION 3: Step counts per measure
        plt.subplot(3, 1, 3)
        
        measure_numbers = [token['measure'] for token in dance_tokens]
        step_counts = [len(token['steps']) for token in dance_tokens]
        
        plt.bar(measure_numbers, step_counts, color='teal')
        plt.title('Number of Steps per Measure')
        plt.xlabel('Measure Number')
        plt.ylabel('Step Count')
        
        # Finalize and save
        plt.tight_layout()
        output_path = os.path.join(output_dir, f"{result.get('track_id', 'unknown')}_visualization.png")
        plt.savefig(output_path)
        logger.info('Visualization saved to %s', output_path)
        plt.show()

    except FileNotFoundError as fnfe:
        logger.error('Required file missing: %s', fnfe)
        print(f"Folder '{song_folder}' does not contain the required files (audio and simfile). Please provide a valid folder.")
        sys.exit(1)
    except Exception as e:
        logger.exception('Error while processing folder %s', song_folder)
        sys.exit(1)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python test_parser.py <song_folder_path>')
        sys.exit(1)
    song_folder_path = sys.argv[1]
    main(song_folder_path) 