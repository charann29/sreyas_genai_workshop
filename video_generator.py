import torch
from diffusers import DiffusionPipeline, AnimateDiffPipeline
import cv2
import numpy as np
from PIL import Image
import os

class VideoGenerator:
    def __init__(self):
        # Use AnimateDiff or similar video generation model
        self.pipe = AnimateDiffPipeline.from_pretrained(
            "guoyww/animatediff-motion-adapter-v1-5-2",
            torch_dtype=torch.float16
        )
        if torch.cuda.is_available():
            self.pipe = self.pipe.to("cuda")
    
    def generate_video_frames(self, prompt, num_frames=16, fps=8):
        """Generate video frames from text prompt"""
        try:
            # Generate video using AnimateDiff
            video = self.pipe(
                prompt,
                num_frames=num_frames,
                guidance_scale=7.5,
                num_inference_steps=25,
                generator=torch.manual_seed(42)
            ).frames[0]
            
            return video
        except Exception as e:
            print(f"Error generating video: {e}")
            return None
    
    def create_video_from_frames(self, frames, output_path, fps=8):
        """Convert frames to video file"""
        if not frames:
            return False
        
        height, width = frames[0].size[1], frames[0].size[0]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for frame in frames:
            # Convert PIL to OpenCV format
            frame_array = np.array(frame)
            frame_bgr = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)
            video_writer.write(frame_bgr)
        
        video_writer.release()
        return True
    
    def generate_video_with_transitions(self, prompts, transition_frames=8):
        """Generate video with smooth transitions between scenes"""
        all_frames = []
        
        for i, prompt in enumerate(prompts):
            frames = self.generate_video_frames(prompt, num_frames=16)
            if frames:
                all_frames.extend(frames)
                
                # Add transition frames between scenes
                if i < len(prompts) - 1:
                    next_frames = self.generate_video_frames(prompts[i+1], num_frames=4)
                    if next_frames:
                        # Simple transition: blend frames
                        for j in range(transition_frames):
                            alpha = j / transition_frames
                            blended = self.blend_images(frames[-1], next_frames[0], alpha)
                            all_frames.append(blended)
        
        return all_frames
    
    def blend_images(self, img1, img2, alpha):
        """Blend two images with given alpha"""
        arr1 = np.array(img1).astype(float)
        arr2 = np.array(img2).astype(float)
        blended = (1 - alpha) * arr1 + alpha * arr2
        return Image.fromarray(blended.astype(np.uint8))
    
    def add_text_overlay(self, video_path, text, output_path):
        """Add text overlay to video"""
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        color = (255, 255, 255)
        thickness = 2
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Add text
            cv2.putText(frame, text, (50, height - 50), font, font_scale, color, thickness)
            out.write(frame)
        
        cap.release()
        out.release()
    
    def generate_complete_video(self, script, output_path="generated_video.mp4"):
        """Generate complete video from script"""
        # Parse script into scenes
        scenes = script.split('\n\n')  # Assume double newline separates scenes
        
        # Generate frames for each scene
        frames = self.generate_video_with_transitions(scenes)
        
        # Create video
        success = self.create_video_from_frames(frames, output_path)
        
        if success:
            print(f"Video generated successfully: {output_path}")
        else:
            print("Failed to generate video")
        
        return success

# Usage
video_gen = VideoGenerator()

# Simple video generation
prompt = "A serene lake with mountains in the background, golden hour lighting"
frames = video_gen.generate_video_frames(prompt, num_frames=24, fps=8)
video_gen.create_video_from_frames(frames, "lake_video.mp4", fps=8)

# Multi-scene video
script = """A peaceful morning in a forest with sunlight filtering through trees.
A deer gracefully walking through the forest clearing.
The sun setting behind the mountains with warm orange light."""

video_gen.generate_complete_video(script, "forest_story.mp4")
