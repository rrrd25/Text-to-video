!pip install diffusers transformers accelerate torch moviepy imageio

import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
import imageio
from moviepy.editor import ImageSequenceClip
import os

# Load the pipeline
pipe = DiffusionPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16, variant="fp16")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()

# Generate video frames
prompt = "spiderman is surfing" #add prompt as per the video generation
try:
    video_frames = pipe(prompt, num_inference_steps=25).frames
except Exception as e:
    print(f"Error generating frames: {e}")
    video_frames = []

# Save individual frames as images
frame_paths = []
try:
    for i, frame in enumerate(video_frames[0]):
        frame_path = f"frame_{i}.png"
        # Ensure frame is in the correct format for imageio
        imageio.imwrite(frame_path, (frame * 255).astype('uint8'))  # Assuming frame values are between 0 and 1
        frame_paths.append(frame_path)
except Exception as e:
    print(f"Error saving frames: {e}")

# Create video using moviepy
try:
    if frame_paths:
        clip = ImageSequenceClip(frame_paths, fps=4)  # Adjust fps as needed
        clip.write_videofile("generated_video1.mp4")
    else:
        print("No frames to create a video.")
except Exception as e:
    print(f"Error creating video: {e}")

# Clean up individual frame images
try:
    for frame_path in frame_paths:
        os.remove(frame_path)
except Exception as e:
    print(f"Error cleaning up frames: {e}")
