from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torchaudio
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import io
import librosa
import logging
import torch
import time
import subprocess
import os

logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow requests from this origin
    # allow_credentials=True,  # Allow cookies and credentials if required
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Load Whisper model and processor
processor = WhisperProcessor.from_pretrained("openai/whisper-small")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
model.config.forced_decoder_ids = None

def reduce_noise(audio, noise_reduction_factor=0.00005):
    """
    Reduce background noise using spectral gating.
    """
    noise_sample = audio[:int(0.1 * len(audio))]  # First 10% of the audio
    noise_mean = np.mean(noise_sample)
    noise_std = np.std(noise_sample)
    denoised_audio = np.where(
        np.abs(audio - noise_mean) > noise_reduction_factor * noise_std, 
        audio, 
        0
    )
    return denoised_audio

def filter_silence(audio, threshold=0.05):
    """
    Filter out silent or very low-amplitude parts of the audio.
    """
    return audio[np.abs(audio) > threshold]

@app.post("/transcribe/")
async def transcribe_audio(request: Request):
    try:
        # Read raw audio data from the request body
        audio_data = await request.body()

        # Save to temporary file
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        AUDIO_DIR = os.path.join(BASE_DIR, "audio")
        os.makedirs(AUDIO_DIR, exist_ok=True)

        # Save raw audio as a temporary file
        input_audio_path = os.path.join(AUDIO_DIR, "input_audio.webm")  # Default extension
        output_audio_path = os.path.join(AUDIO_DIR, "output_audio.wav")  # For Whisper processing

        with open(input_audio_path, "wb") as temp_audio_file:
            temp_audio_file.write(audio_data)

        # Convert the input audio to WAV format using FFmpeg
        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-i", input_audio_path,  # Input file
                    "-ar", "16000",          # Resample to 16kHz
                    "-ac", "1",              # Mono audio
                    output_audio_path        # Output file
                ],
                check=True
            )
        except subprocess.CalledProcessError as e:
            raise HTTPException(status_code=500, detail=f"Audio conversion failed: {str(e)}")

        # Load the converted audio file
        waveform, sampling_rate = torchaudio.load(output_audio_path)

        # Preprocess the audio and perform transcription
        waveform = waveform.squeeze().numpy()  # Convert from Tensor to NumPy
        input_features = processor(waveform, sampling_rate=sampling_rate, return_tensors="pt").input_features
        predicted_ids = model.generate(input_features)
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

        return JSONResponse(content={"transcription": transcription})

    except Exception as e:
        logger.error(f"Transcription error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=6002)
