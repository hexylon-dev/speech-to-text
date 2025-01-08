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
from transformers import pipeline

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow requests from this origin
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Load model and processor once to reuse across requests
transcribe = pipeline(task="automatic-speech-recognition", model="vasista22/whisper-hindi-large-v2", chunk_length_s=30, device="mps")
transcribe.model.config.forced_decoder_ids = transcribe.tokenizer.get_decoder_prompt_ids(language="hi", task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large").to('mps')
model.config.forced_decoder_ids = None

vad_model = AutoModel(model="paraformer-zh", model_revision="v2.0.4",
                        vad_model="fsmn-vad", vad_model_revision="v2.0.4",
                        punc_model="ct-punc-c", punc_model_revision="v2.0.4",
                        disable_update=True
                        # spk_model="cam++", spk_model_revision="v2.0.2",
                        )

# Directory for temporary files
TEMP_DIR = "temp_audio"
os.makedirs(TEMP_DIR, exist_ok=True)

def filter_silence(audio, threshold=0.05): 
    """
    Filter out silent or very low-amplitude parts of the audio.
    """
    normalized_audio = audio / np.max(np.abs(audio))
    return normalized_audio[np.abs(normalized_audio) > threshold]

def reduce_noise(audio, noise_reduction_factor=0.9):
    """
    Reduce background noise using spectral gating.
    """
    # Estimate noise from the beginning of the audio
    noise_sample = audio[:int(0.1 * len(audio))]  # First 10% of the audio
    noise_mean = np.mean(audio)
    noise_std = np.std(audio)
    
    # Reduce noise by subtracting mean and scaling
    denoised_audio = np.where(
        np.abs(audio - noise_mean) > noise_reduction_factor * noise_std, 
        audio, 
        0
    )
    return denoised_audio

def preprocess_audio(audio_buffer, sampling_rate):
    """
    Preprocess audio: filter noise, silence, and resample to 16 kHz.
    """
    # Convert from Tensor to NumPy
    waveform = audio_buffer.squeeze().numpy()

    # Noise reduction
    noise_sample = waveform[:int(0.1 * len(waveform))]  # Use first 10% as noise sample

    # Silence filtering
    filtered_waveform = filter_silence(denoised_waveform)

    # Check if valid audio remains
    if len(filtered_waveform) == 0:
        raise ValueError("Audio too noisy or silent after preprocessing.")

    # Resample to 16 kHz if needed
    if sampling_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=16000)
        filtered_waveform = resampler(torch.tensor(filtered_waveform)).numpy()

    return filtered_waveform

@app.post("/transcribe-buffer")
async def transcribe_buffer(request: Request):
    # Create unique paths for temporary files
    temp_raw_path = f"temp_audio/raw_{'1'}.webm"  # Temporary WebM file
    temp_wav_path = temp_raw_path.replace(".webm", ".wav")      # WAV file path

    try:
        # Read the audio buffer from the request body
        audio_buffer = await request.body()

        # Save the buffer to a temporary file
        with open(temp_raw_path, "ab") as temp_raw_file:
            temp_raw_file.write(audio_buffer)

        # Debugging: Confirm the file is saved correctly
        print(f"Audio buffer saved as: {temp_raw_path}")

        ffmpeg.input(temp_raw_path).output(temp_wav_path, acodec='pcm_s16le', ss='-1').run(overwrite_output=True, cmd=["ffmpeg", "-y"], capture_stdout=True, capture_stderr=True)

        audio = AudioSegment.from_file(temp_wav_path)
        start_time = len(audio) - 1000  # Duration is in milliseconds

        # Trim the audio to the last one second
        last_one_second = audio[start_time:]

        # Export the trimmed audio (replace 'output_audio.mp3' with your desired output file path)
        last_one_second.export(f"temp_audio/temp{audio_id}_.wav", format="wav")


        res = vad_model.generate(input=f'temp_audio/temp{audio_id}_.wav', 
                            batch_size_s=300, 
                            hotword='魔搭')

        # Use ffmpeg to convert the WebM file to WAV format
        # ffmpeg.input(temp_raw_path).output(temp_wav_path, acodec='pcm_s16le').run(overwrite_output=True, cmd=["ffmpeg", "-y"], capture_stdout=True, capture_stderr=True)

        # # Process the WAV file with Whisper
        print(res[0])
        if res[0]['text'] == '':
            if proc == True:
                transcription = process_audio(temp_wav_path)
                proc = False
                return {"transcription": transcription}
            else:
                return 0

        proc = True

    except Exception as e:
        print(e)
        raise HTTPException(status_code=400, detail=f"Error processing audio: {str(e)}")

@app.websocket("/ws/transcribe-buffer")
async def transcribe_websocket(websocket: WebSocket):
    await websocket.accept()
    sec = 0

    try:
        while True:
            # Receive audio data from WebSocket
            message = await websocket.receive_json()
            
            audio_id = message.get("id")
            audio_buffer = message.get("buffer")

            temp_raw_path = f"temp_audio/raw_{audio_id}.webm"
            temp_wav_path = temp_raw_path.replace(".webm", ".wav")  # WAV file path

            # Save the buffer to a temporary file
            with open(temp_raw_path, "ab") as temp_raw_file:
                temp_raw_file.write(bytes(audio_buffer))

            print(f"Audio buffer saved as: {temp_raw_path}")  # Debugging

            # Convert WebM to WAV using ffmpeg
            ffmpeg.input(temp_raw_path).output(temp_wav_path, acodec='pcm_s16le').run(
                overwrite_output=True, cmd=["ffmpeg", "-y"], capture_stdout=True, capture_stderr=True
            )

            # Load the WAV file and extract the last one second of audio
            audio = AudioSegment.from_file(temp_wav_path)
            start_time = len(audio) - 2000  # Duration is in milliseconds
            if start_time < 0:
                start_time = len(audio) - 1000  # Duration is in milliseconds

            last_one_second = audio[start_time:]
            last_one_second.export(f"temp_audio/temp{audio_id}_.wav", format="wav")  # Export trimmed audio

            # Perform VAD model inference
            res = [{'text': ''}]
            try:
                res = vad_model.generate(
                    input=f'temp_audio/temp{audio_id}_.wav',
                    batch_size_s=300,
                    hotword='魔搭'
                )
            except:
                print('error')

            print(f"Seconds: {sec}")  # Debugging output
            if len(res) > 0 and res[0]['text'] == '':
                if sec > 0:
                    audio = AudioSegment.from_file(temp_wav_path)
                    start_time = len(audio) - (sec + 1000)  # Duration is in milliseconds

                    print(f"temp_audio/temp{audio_id}_.wav")
                    last_one_second = audio[start_time:]
                    last_one_second.export(f"temp_audio/temp{audio_id}_.wav", format="wav")
                    transcription = process_audio(f"temp_audio/temp{audio_id}_.wav")
                    sec = 0
                    await websocket.send_json({"transcription": transcription})
                else:
                    await websocket.send_json(0)
            else:
                await websocket.send_json("Audio recognised")
                sec += 1000

    except WebSocketDisconnect:
        print("WebSocket disconnected")
    except Exception as e:
        print(e)
        raise e
        await websocket.send_json({"error": f"Error processing audio: {str(e)}"})
    # finally:
        # Clean up temporary files
        # if os.path.exists(temp_raw_path):
        #     os.remove(temp_raw_path)
        # if os.path.exists(temp_wav_path):
        #     os.remove(temp_wav_path)
        # if os.path.exists(f"temp_audio/temp{audio_id}_.wav"):
        #     os.remove(f"temp_audio/temp{audio_id}_.wav")

def process_audio(file_path: str) -> str:
    """
    Process the WAV file using Whisper (your existing Whisper processing code).
    """
    # Load and preprocess the audio
    # waveform, sampling_rate = torchaudio.load(file_path)

    # # Resample the audio if necessary (Whisper models expect 16 kHz audio)
    # if sampling_rate != 16000:
    #     resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=16000)
    #     waveform = resampler(waveform)
    #     sampling_rate = 16000

    # # Preprocess the audio
    # waveform = waveform.squeeze().numpy()  # Convert from Tensor to NumPy
    # denoised_waveform = reduce_noise(waveform)  # Apply noise reduction
    # filtered_waveform = filter_silence(denoised_waveform)  # Remove silence

    # Ensure the processed waveform has data to avoid errors
    # if len(filtered_waveform) == 0:
    #     raise ValueError("No valid audio data after preprocessing (too much noise or silence).")

    # Convert the processed audio to input features
    input_features = transcribe(file_path)
    return input_features['text']
    # Generate transcription
    # predicted_ids = model.generate(input_features.to('mps'))
    # transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

    return transcription[0]

# Clean up temporary files (Optional)
# @app.on_event("shutdown")
# def cleanup_temp_files():
#     for filename in os.listdir(TEMP_DIR):
#         file_path = os.path.join(TEMP_DIR, filename)
#         os.remove(file_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
