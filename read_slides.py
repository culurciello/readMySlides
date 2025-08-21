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
from typing import List, Tuple, Union
import subprocess

import cv2
import numpy as np
from PIL import Image
import pytesseract
from gtts import gTTS
from moviepy import ImageClip, AudioFileClip, concatenate_videoclips
from tqdm import tqdm
import fitz  # PyMuPDF for PDF processing


class SlideReader:
    def __init__(self, input_source: Union[str, Path] = "slides", output_file: str = "presentation.mp4", 
                 language: str = "en", speech_speed: float = 1.0, input_type: str = "auto"):
        """
        Initialize the SlideReader.
        
        Args:
            input_source: Path to slides directory or PDF file
            output_file: Output MP4 filename
            language: Language for text-to-speech
            speech_speed: Speed multiplier for speech (1.0 = normal)
            input_type: Type of input ('auto', 'images', 'pdf')
        """
        self.input_source = Path(input_source)
        self.output_file = output_file
        self.language = language
        self.speech_speed = speech_speed
        self.input_type = input_type
        self.temp_dir = Path(tempfile.mkdtemp())
        self.slides_dir = self.temp_dir / "slides"
        self.slides_dir.mkdir(exist_ok=True)
        
        # Auto-detect input type if not specified
        if self.input_type == "auto":
            self.input_type = self._detect_input_type()
        
        # Process input based on type
        self._prepare_slides()
    
    def _detect_input_type(self) -> str:
        """
        Auto-detect the input type based on the source.
        
        Returns:
            Detected input type ('images', 'pdf')
        """
        if self.input_source.is_file():
            suffix = self.input_source.suffix.lower()
            if suffix == '.pdf':
                return 'pdf'
            else:
                return 'images'
        elif self.input_source.is_dir():
            return 'images'
        else:
            raise FileNotFoundError(f"Input source '{self.input_source}' not found")
    
    def _prepare_slides(self):
        """
        Prepare slides based on input type.
        """
        if self.input_type == 'images':
            self._prepare_from_images()
        elif self.input_type == 'pdf':
            self._prepare_from_pdf()
        else:
            raise ValueError(f"Unsupported input type: {self.input_type}")
    
    def _prepare_from_images(self):
        """
        Prepare slides from image directory.
        """
        if not self.input_source.exists():
            raise FileNotFoundError(f"Images directory '{self.input_source}' not found")
        
        # Copy images to temp slides directory
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(self.input_source.glob(f"*{ext}"))
            image_files.extend(self.input_source.glob(f"*{ext.upper()}"))
        
        for i, img_file in enumerate(sorted(image_files)):
            dest = self.slides_dir / f"slide_{i+1:03d}{img_file.suffix}"
            shutil.copy2(img_file, dest)
    
    
    def _prepare_from_pdf(self):
        """
        Prepare slides from PDF by converting pages to images.
        """
        if not self.input_source.exists():
            raise FileNotFoundError(f"PDF file '{self.input_source}' not found")
        
        # Open PDF
        pdf_document = fitz.open(str(self.input_source))
        
        # Convert each page to image
        for page_num in range(pdf_document.page_count):
            page = pdf_document.load_page(page_num)
            
            # Render page as image (high DPI for better OCR)
            mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better quality
            pix = page.get_pixmap(matrix=mat)
            
            # Save as PNG
            img_path = self.slides_dir / f"slide_{page_num+1:03d}.png"
            pix.save(str(img_path))
        
        pdf_document.close()
    
    def extract_text_from_slide(self, slide_path: Path) -> str:
        """
        Extract text from a slide (image or text file).
        
        Args:
            slide_path: Path to the slide file
            
        Returns:
            Extracted text string
        """
        try:
            if slide_path.suffix.lower() == '.txt':
                # For PPTX-generated text files, read directly
                return slide_path.read_text(encoding='utf-8')
            else:
                
                # For image files, use OCR
                image = cv2.imread(str(slide_path))
                if image is None:
                    raise ValueError(f"Could not load image: {slide_path}")
                
                # Convert to PIL Image for better OCR
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(image_rgb)
                
                # Enhance image for better OCR
                pil_image = pil_image.convert('L')
                
                # Use pytesseract to extract text
                text = pytesseract.image_to_string(pil_image, lang='eng')
                
                # Clean up the text
                text = ' '.join(text.split())
                text = text.strip()
                
                return text if text else "No text found in this slide."
            
        except Exception as e:
            print(f"Error extracting text from {slide_path}: {e}")
            return f"Error reading slide {slide_path.name}"
    
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
        Get all slide files from the slides directory.
        
        Returns:
            List of slide file paths, sorted by name
        """
        # Return image files for images and PDF
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']
        slide_files = []
        
        for ext in image_extensions:
            slide_files.extend(self.slides_dir.glob(f"*{ext}"))
            slide_files.extend(self.slides_dir.glob(f"*{ext.upper()}"))
        
        return sorted(slide_files)
    
    def create_slide_clip(self, slide_path: Path, audio_path: Path, duration: float = None):
        """
        Create a video clip from a slide and audio.
        
        Args:
            slide_path: Path to slide file (image or text)
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
            
            # For text files (PPTX), create a simple black image with text
            if slide_path.suffix.lower() == '.txt':
                # Create a simple black background image
                img_array = np.zeros((720, 1280, 3), dtype=np.uint8)
                temp_img_path = self.temp_dir / f"{slide_path.stem}.png"
                cv2.imwrite(str(temp_img_path), img_array)
                image_clip = ImageClip(str(temp_img_path), duration=total_duration)
            else:
                # Use the actual image
                image_clip = ImageClip(str(slide_path), duration=total_duration)
            
            # Add audio if it exists
            if audio_path.exists():
                audio = AudioFileClip(str(audio_path))
                image_clip = image_clip.with_audio(audio)
            
            return image_clip
            
        except Exception as e:
            print(f"Error creating clip for {slide_path}: {e}")
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
            final_video = concatenate_videoclips(clips, method="chain")
            
            # Write final video
            print(f"Writing final video to {self.output_file}...")
            final_video.write_videofile(
                self.output_file,
                fps=24,
                codec='libx264',
                audio_codec='aac',
                temp_audiofile=str(self.temp_dir / 'temp_audio.m4a'),
                remove_temp=True,
                threads=16
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
    parser.add_argument('input_source', nargs='?', default='slides',
                       help='Path to slides directory or PDF file (default: slides)')
    parser.add_argument('--output', default='presentation.mp4',
                       help='Output video filename (default: presentation.mp4)')
    parser.add_argument('--type', choices=['auto', 'images', 'pdf'], default='auto',
                       help='Input type (default: auto-detect)')
    parser.add_argument('--language', default='en',
                       help='Language for text-to-speech (default: en)')
    parser.add_argument('--speed', type=float, default=1.0,
                       help='Speech speed multiplier (default: 1.0)')
    
    args = parser.parse_args()
    
    # Create slide reader and process slides
    reader = SlideReader(
        input_source=args.input_source,
        output_file=args.output,
        input_type=args.type,
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