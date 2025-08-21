# Slide to Video with Voice-over

This tool converts presentation slides into a video with automated voice-over using Whisper AI.

![](images/pptsux.png)

## Features

- Extracts text from slide images using Whisper
- Converts text to speech (currently saves text to file as a placeholder)
- Creates a video with each slide displayed with its corresponding audio
- Combines all slides into a single MP4 file

## Prerequisites

- Python 3.8+
- FFmpeg (for audio/video processing)

## Installation

1. Clone this repository
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Install FFmpeg (if not already installed):
   - On macOS: `brew install ffmpeg`
   - On Ubuntu: `sudo apt install ffmpeg`
   - On Windows: Download from [FFmpeg's website](https://ffmpeg.org/download.html)

## Usage

1. Place your slide images in the `slides/` directory
2. Run the script:
   ```bash
   python slides_to_video.py
   ```
3. The output video will be saved as `presentation.mp4` in the current directory

## Customization

You can modify the following parameters in the script:
- `output_file`: Name of the output video file
- `fps`: Frames per second for the output video
- `audio_speed`: Playback speed of the audio

## Notes

- The current implementation uses Whisper for text extraction but doesn't include a full TTS system. You'll need to integrate a TTS service for complete functionality.
- Supported image formats: .jpg, .jpeg, .png, .bmp
- The script creates temporary files during processing but cleans them up automatically

## License

MIT
