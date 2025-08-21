#!/usr/bin/env python3
"""
Slide Reader with Voice-over Generator

This program reads slides from the slides/ directory, extracts text using OCR,
generates speech using TTS, and creates an MP4 video with synchronized audio and visuals.
"""

import os
import glob
import tempfile
import shutil
from pathlib import Path
from typing import List, Tuple
import subprocess

import cv2
import numpy as np
from PIL import Image
import pytesseract
from gtts import gTTS
from moviepy import ImageClip, AudioFileClip, concatenate_videoclips
from tqdm import tqdm


class SlideReader:
    def __init__(self, slides_dir: str = "slides", output_file: str = "presentation.mp4", 
                 language: str = "en", speech_speed: float = 1.0):
        """
        Initialize the SlideReader.
        
        Args:
            slides_dir: Directory containing slide images
            output_file: Output MP4 filename
            language: Language for text-to-speech
            speech_speed: Speed multiplier for speech (1.0 = normal)
        """
        self.slides_dir = Path(slides_dir)
        self.output_file = output_file
        self.language = language
        self.speech_speed = speech_speed
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # Ensure slides directory exists
        if not self.slides_dir.exists():
            raise FileNotFoundError(f"Slides directory '{slides_dir}' not found")
    
    def extract_text_from_slide(self, image_path: Path) -> str:
        """
        Extract text from a slide image using OCR.
        
        Args:
            image_path: Path to the slide image
            
        Returns:
            Extracted text string
        """
        try:
            # Read image
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Convert to PIL Image for better OCR
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            
            # Enhance image for better OCR
            # Convert to grayscale and increase contrast
            pil_image = pil_image.convert('L')
            
            # Use pytesseract to extract text
            text = pytesseract.image_to_string(pil_image, lang='eng')
            
            # Clean up the text
            text = ' '.join(text.split())  # Remove extra whitespace
            text = text.strip()
            
            return text if text else "No text found in this slide."
            
        except Exception as e:
            print(f"Error extracting text from {image_path}: {e}")
            return f"Error reading slide {image_path.name}"
    
    def text_to_speech(self, text: str, output_path: Path) -> bool:
        """
        Convert text to speech using Google Text-to-Speech.
        
        Args:
            text: Text to convert to speech
            output_path: Path to save the audio file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not text.strip():
                # Create a short silence for empty text
                silence_duration = 2.0
                sample_rate = 44100
                samples = int(silence_duration * sample_rate)
                silence = np.zeros(samples, dtype=np.float32)
                
                # Save as WAV using scipy or create with ffmpeg
                subprocess.run([
                    'ffmpeg', '-y', '-f', 'f32le', '-ar', str(sample_rate), 
                    '-ac', '1', '-i', 'pipe:', str(output_path)
                ], input=silence.tobytes(), capture_output=True)
                return True
            
            # Create TTS object
            tts = gTTS(text=text, lang=self.language, slow=False)
            
            # Save to temporary mp3 file
            temp_mp3 = self.temp_dir / f"{output_path.stem}.mp3"
            tts.save(str(temp_mp3))
            
            # Convert MP3 to WAV and adjust speed if needed
            speed_filter = f"atempo={self.speech_speed}" if self.speech_speed != 1.0 else ""
            
            cmd = ['ffmpeg', '-y', '-i', str(temp_mp3)]
            if speed_filter:
                cmd.extend(['-af', speed_filter])
            cmd.append(str(output_path))
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"FFmpeg error: {result.stderr}")
                return False
            
            # Clean up temp MP3
            temp_mp3.unlink(missing_ok=True)
            
            return True
            
        except Exception as e:
            print(f"Error in text-to-speech conversion: {e}")
            return False
    
    def get_slide_files(self) -> List[Path]:
        """
        Get all slide image files from the slides directory.
        
        Returns:
            List of slide file paths, sorted by name
        """
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']
        slide_files = []
        
        for ext in image_extensions:
            slide_files.extend(self.slides_dir.glob(f"*{ext}"))
            slide_files.extend(self.slides_dir.glob(f"*{ext.upper()}"))
        
        return sorted(slide_files)
    
    def create_slide_clip(self, image_path: Path, audio_path: Path, duration: float = None):
        """
        Create a video clip from a slide image and audio.
        
        Args:
            image_path: Path to slide image
            audio_path: Path to audio file
            duration: Duration in seconds (auto-detected if None)
            
        Returns:
            MoviePy ImageClip with audio
        """
        try:
            # Get audio duration
            if audio_path.exists():
                audio_clip = AudioFileClip(str(audio_path))
                audio_duration = audio_clip.duration
                audio_clip.close()
            else:
                audio_duration = 3.0  # Default duration
            
            # Add padding
            total_duration = audio_duration + 0.5  # 0.5 second pause between slides
            
            # Create image clip
            image_clip = ImageClip(str(image_path), duration=total_duration)
            
            # Add audio if it exists
            if audio_path.exists():
                audio = AudioFileClip(str(audio_path))
                # Use with_audio instead of set_audio for newer MoviePy versions
                image_clip = image_clip.with_audio(audio)
            
            return image_clip
            
        except Exception as e:
            print(f"Error creating clip for {image_path}: {e}")
            return None
    
    def process_slides(self) -> bool:
        """
        Process all slides and create the final video.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get all slide files
            slide_files = self.get_slide_files()
            
            if not slide_files:
                print(f"No slide images found in {self.slides_dir}")
                return False
            
            print(f"Found {len(slide_files)} slide(s)")
            
            clips = []
            
            # Process each slide
            for i, slide_path in enumerate(tqdm(slide_files, desc="Processing slides")):
                print(f"\nProcessing slide {i+1}: {slide_path.name}")
                
                # Extract text from slide
                text = self.extract_text_from_slide(slide_path)
                print(f"Extracted text: {text[:100]}{'...' if len(text) > 100 else ''}")
                
                # Generate audio file path
                audio_path = self.temp_dir / f"slide_{i+1:03d}.wav"
                
                # Convert text to speech
                if self.text_to_speech(text, audio_path):
                    print(f"Generated audio: {audio_path.name}")
                else:
                    print(f"Failed to generate audio for slide {i+1}")
                    continue
                
                # Create video clip
                clip = self.create_slide_clip(slide_path, audio_path)
                if clip:
                    clips.append(clip)
                    print(f"Created clip for slide {i+1}")
                else:
                    print(f"Failed to create clip for slide {i+1}")
            
            if not clips:
                print("No valid clips created")
                return False
            
            # Combine all clips
            print(f"\nCombining {len(clips)} clips into final video...")
            final_video = concatenate_videoclips(clips, method="compose")
            
            # Write final video
            print(f"Writing final video to {self.output_file}...")
            final_video.write_videofile(
                self.output_file,
                fps=24,
                codec='libx264',
                audio_codec='aac',
                temp_audiofile=str(self.temp_dir / 'temp_audio.m4a'),
                remove_temp=True
            )
            
            # Close clips to free memory
            for clip in clips:
                clip.close()
            final_video.close()
            
            print(f"\nVideo successfully created: {os.path.abspath(self.output_file)}")
            return True
            
        except Exception as e:
            print(f"Error processing slides: {e}")
            return False
        
        finally:
            # Clean up temporary directory
            try:
                shutil.rmtree(self.temp_dir, ignore_errors=True)
            except:
                pass
    
    def __del__(self):
        """Clean up temporary files when object is destroyed."""
        try:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        except:
            pass


def main():
    """Main function to run the slide reader."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert slides to video with voice-over')
    parser.add_argument('--slides-dir', default='slides', 
                       help='Directory containing slide images (default: slides)')
    parser.add_argument('--output', default='presentation.mp4',
                       help='Output video filename (default: presentation.mp4)')
    parser.add_argument('--language', default='en',
                       help='Language for text-to-speech (default: en)')
    parser.add_argument('--speed', type=float, default=1.0,
                       help='Speech speed multiplier (default: 1.0)')
    
    args = parser.parse_args()
    
    # Create slide reader and process slides
    reader = SlideReader(
        slides_dir=args.slides_dir,
        output_file=args.output,
        language=args.language,
        speech_speed=args.speed
    )
    
    success = reader.process_slides()
    
    if success:
        print("✅ Slides successfully converted to video!")
    else:
        print("❌ Failed to convert slides to video.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())