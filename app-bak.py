from fastapi import FastAPI, Request, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torchaudio
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import os
import uuid
from datetime import datetime
import torch
import ffmpeg
from funasr import AutoModel
from pydub import AudioSegment
from pyannote.audio import Pipeline
import wave
from scipy import signal
from scipy.signal import butter, filtfilt
import webrtcvad

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models and set up constants
processor = WhisperProcessor.from_pretrained("openai/whisper-large")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large").to('mps')
model.config.forced_decoder_ids = None

vad_model = AutoModel(model="paraformer-zh", model_revision="v2.0.4",
                  vad_model="fsmn-vad", vad_model_revision="v2.0.4",
                  punc_model="ct-punc-c", punc_model_revision="v2.0.4",
                  spk_model="cam++", spk_model_revision="v2.0.2",
                  )
vad = webrtcvad.Vad()

TEMP_DIR = "temp_audio"
os.makedirs(TEMP_DIR, exist_ok=True)

def butter_bandpass(lowcut, highcut, fs, order=5):
    """
    Design a bandpass filter for human voice frequencies
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_bandpass_filter(data, fs, lowcut=300, highcut=3400):
    """
    Apply bandpass filter to focus on human voice frequencies
    """
    b, a = butter_bandpass(lowcut, highcut, fs)
    y = filtfilt(b, a, data)
    return y

def spectral_gating(audio, sr, n_std_thresh=1.5):
    """
    Apply spectral gating noise reduction
    """
    # Compute spectrogram
    f, t, Zxx = signal.stft(audio, fs=sr, nperseg=2048)
    
    # Estimate noise profile from the first 100ms
    n_samples = int(sr * 0.1)
    noise_profile = np.mean(np.abs(Zxx[:, :n_samples]), axis=1)
    noise_thresh = noise_profile * n_std_thresh
    
    # Create mask and apply
    mask = np.abs(Zxx) > noise_thresh[:, np.newaxis]
    Zxx_cleaned = Zxx * mask
    
    # Inverse STFT
    _, cleaned_audio = signal.istft(Zxx_cleaned, fs=sr)
    return cleaned_audio

def adaptive_noise_reduction(audio, sampling_rate):
    """
    Apply adaptive noise reduction using multiple techniques
    """
    # 1. Apply bandpass filter for human voice range
    filtered_audio = apply_bandpass_filter(audio, sampling_rate)
    
    # 2. Apply spectral gating
    denoised_audio = spectral_gating(filtered_audio, sampling_rate)
    
    # 3. Normalize audio
    normalized_audio = denoised_audio / np.max(np.abs(denoised_audio))
    
    # 4. Apply dynamic threshold based on local statistics
    window_size = int(sampling_rate * 0.02)  # 20ms windows
    threshold = np.zeros_like(normalized_audio)
    
    for i in range(0, len(normalized_audio), window_size):
        window = normalized_audio[i:i + window_size]
        local_energy = np.mean(window ** 2)
        local_threshold = np.sqrt(local_energy) * 0.1
        threshold[i:i + window_size] = local_threshold
    
    # Apply adaptive threshold
    processed_audio = normalized_audio * (np.abs(normalized_audio) > threshold)
    
    return processed_audio

def filter_silence(audio, threshold=0.05, frame_length=1024): 
    """
    Enhanced silence filtering with frame-based analysis
    """
    # Calculate frame energies
    frames = np.array_split(audio, len(audio) // frame_length)
    frame_energies = [np.sum(frame ** 2) for frame in frames]
    
    # Calculate adaptive threshold
    median_energy = np.median(frame_energies)
    threshold = median_energy * threshold
    
    # Create mask for non-silent frames
    mask = np.concatenate([
        np.ones(len(frame)) if energy > threshold else np.zeros(len(frame))
        for frame, energy in zip(frames, frame_energies)
    ])
    
    return audio * mask

def preprocess_audio(audio_buffer, sampling_rate):
    """
    Enhanced audio preprocessing pipeline
    """
    # Convert to numpy array
    waveform = audio_buffer.squeeze().numpy()
    
    # 1. Apply initial noise reduction
    denoised_audio = adaptive_noise_reduction(waveform, sampling_rate)
    
    # 2. Filter out silence
    voice_only = filter_silence(denoised_audio)
    
    # 3. Resample if needed
    if sampling_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=16000)
        voice_only = resampler(torch.tensor(voice_only)).numpy()
    
    return voice_only

def process_audio(file_path: str) -> str:
    """
    Process audio with enhanced noise reduction
    """
    # Load audio
    waveform, sampling_rate = torchaudio.load(file_path)
    
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Apply preprocessing pipeline
    processed_audio = preprocess_audio(waveform, sampling_rate)
    
    # Ensure we have valid audio data
    if len(processed_audio) == 0:
        raise ValueError("No valid audio data after preprocessing")
    
    # Convert to input features
    input_features = processor(processed_audio, sampling_rate=16000, return_tensors="pt").input_features
    
    # Generate transcription
    predicted_ids = model.generate(input_features.to('mps'), language='hi')
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    
    return transcription[0]

# [Rest of the FastAPI routes and WebSocket handling code remains the same]