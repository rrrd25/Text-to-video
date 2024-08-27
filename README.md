# Text-to-Video Generation using Diffusion Models

This project demonstrates how to generate videos from text prompts using diffusion models. The code is intended to run in Google Colab and leverages the `diffusers` library, along with `transformers`, `accelerate`, and other essential libraries.

## Description

The script `generate_video.py` generates a short video based on a text prompt using the following steps:

1. **Load the Pipeline:** Load the pre-trained diffusion model with CPU offloading to manage memory usage.
2. **Generate Video Frames:** Use a text prompt to generate individual video frames.
3. **Save Frames:** The generated frames are saved as PNG images.
4. **Create Video:** The frames are compiled into a video using `moviepy`.
5. **Clean Up:** The saved frames are removed to free up space.

## Requirements

To run the code in Google Colab, install the necessary dependencies with the following command:

```bash
!pip install -r requirements.txt
