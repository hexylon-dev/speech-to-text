from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torchaudio
import numpy as np
from scipy.signal import butter, lfilter

# Load model and processor
processor = WhisperProcessor.from_pretrained("openai/whisper-small")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
model.config.forced_decoder_ids = None

def reduce_noise(audio, noise_reduction_factor=0.8):
    """
    Reduce background noise using spectral gating.
    """
    # Estimate noise from the beginning of the audio
    noise_sample = audio[:int(0.1 * len(audio))]  # First 10% of the audio
    noise_mean = np.mean(noise_sample)
    noise_std = np.std(noise_sample)
    
    # Reduce noise by subtracting mean and scaling
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

# Load the audio file
audio_path = "temp_audio/raw_1.wav"  # Replace with your audio file path
waveform, sampling_rate = torchaudio.load(audio_path)

# Resample the audio if necessary (Whisper models expect 16 kHz audio)
if sampling_rate != 16000:
    resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=16000)
    waveform = resampler(waveform)
    sampling_rate = 16000

# Preprocess the audio
waveform = waveform.squeeze().numpy()  # Convert from Tensor to NumPy
# denoised_waveform = reduce_noise(waveform)  # Apply noise reduction
# filtered_waveform = filter_silence(denoised_waveform)  # Remove silence

# Ensure the processed waveform has data to avoid errors
if len(waveform) == 0:
    raise ValueError("No valid audio data after preprocessing (too much noise or silence).")

# Convert the processed audio to input features
input_features = processor(waveform, sampling_rate=sampling_rate, return_tensors="pt").input_features

# Generate token IDs
predicted_ids = model.generate(input_features)

# Decode token IDs to text
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
print("Transcription:", transcription[0])
