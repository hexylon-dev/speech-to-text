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
import webrtcvad
import wave
import requests
import sseclient
import noisereduce as nr
from scipy.io import wavfile
from df.enhance import enhance, init_df, load_audio, save_audio
from df.utils import download_file

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow requests from this origin
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Load model and processor once to reuse across requests
processor = WhisperProcessor.from_pretrained("openai/whisper-large")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large").to('mps')
model.config.forced_decoder_ids = None

# vad_model = AutoModel(model="paraformer-zh", model_revision="v2.0.4",
#                   vad_model="fsmn-vad", vad_model_revision="v2.0.4",
#                   punc_model="ct-punc-c", punc_model_revision="v2.0.4",
#                   spk_model="cam++", spk_model_revision="v2.0.2",
#                   )
vad = webrtcvad.Vad()

# Directory for temporary files
TEMP_DIR = "temp_audio"
os.makedirs(TEMP_DIR, exist_ok=True)

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

def re_sample_audio(audio_path, sr):
    print("Resampling audio")
    audio, _ = librosa.load(audio_path, sr=16000)
    audio_resampled = librosa.resample(audio, target_sr=sr, orig_sr=16000)
    sf.write(audio_path, audio_resampled, 48000)
    print("Resampling done")

@app.websocket("/ws/transcribe-buffer")
async def transcribe_websocket(websocket: WebSocket):
    await websocket.accept()
    sec = 0
    buffer = []
    bufferLength = 2000
    threashold = 90

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

            # Load the WAV file and extract the last one separate_filecond of audio
            audio = AudioSegment.from_file(temp_wav_path)
            start_time = len(audio) - 2000  # Duration is in milliseconds
            if start_time < 0:
                start_time = len(audio) - 1000  # Duration is in milliseconds

            bufferFileName = f"temp_audio/temp{audio_id}_.wav"
            last_one_second = audio[start_time:]
            last_one_second.export(bufferFileName, format="wav")  # Export trimmed audio

            # rate, data = wavfile.read(bufferFileName)
            # reduced_noise = nr.reduce_noise(y=data, sr=rate)
            # wavfile.write(bufferFileName, rate, reduced_noise)

            model, df_state, _ = init_df()
            audio, _ = load_audio(bufferFileName, sr=df_state.sr() * 2)
            enhanced = enhance(model, df_state, audio)
            save_audio(bufferFileName, enhanced, df_state.sr())

            # Perform VAD model inference
            res = [{'text': ''}]
            try:
                # res = vad_model.generate(
                #     input=f'temp_audio/temp{audio_id}_.wav',
                #     batch_size_s=300,
                #     # hotword='魔搭'
                # )

                vad.set_mode(3)  # 0 = least aggressive, 3 = most sensitive

                frame_duration_ms = 10  # Frame duration in milliseconds (10ms, 20ms, or 30ms)

                # Open the WAV file
                with wave.open(f"temp_audio/temp{audio_id}_.wav", 'rb') as wf:
                    rate = wf.getframerate()
                    channels = wf.getnchannels()
                    width = wf.getsampwidth()

                    # Check for supported audio format
                    if rate not in [8000, 16000, 32000, 48000]:
                        raise ValueError(f"Unsupported sample rate: {rate}. Supported rates are 8000, 16000, 32000, 48000 Hz.")
                    if channels != 1:
                        raise ValueError("Audio must be mono (1 channel).")
                    if width != 2:
                        raise ValueError("Audio must be 16-bit PCM.")

                    frames = wf.readframes(wf.getnframes())

                # Calculate frame size
                frame_size = int(rate * frame_duration_ms / 1000) * width  # Number of bytes per frame
                count = 0
                # Process the audio in chunks
                for i in range(0, len(frames), frame_size):
                    frame = frames[i:i + frame_size]
                    if len(frame) < frame_size:
                        break  # Ignore incomplete frames

                    # Pass each frame to WebRTC VAD
                    is_speech = vad.is_speech(frame, rate)
                    if is_speech:
                        # print("Speech detected")
                        count += 1
                    # else:
                        # print("No speech detected")
                buffer.append(count)
                count = 0

                print(buffer)
            except Exception as e:
                raise e

            # print(res)

            print(f"Seconds: {sec}")  # Debugging output
            lastCount = buffer[len(buffer) - 1]
            if lastCount < threashold:
                if sec > 0:
                    audio = AudioSegment.from_file(temp_wav_path)
                    start_time = len(audio) - (sec + bufferLength)  # Duration is in milliseconds

                    last_one_second = audio[start_time:]
                    last_one_second.export(f"temp_audio/temp{audio_id}_.wav", format="wav")
                    transcription = process_audio(f"temp_audio/temp{audio_id}_.wav")
                    sec = 0
                    print(f"Transcription: {transcription}")
                    query(transcription)
                    # await websocket.send_json({"transcription": transcription})
                else:
                    await websocket.send_json(0)
            else:
                await websocket.send_json("Audio recognised")
                sec += bufferLength

    except WebSocketDisconnect:
        print("WebSocket disconnected")
    except Exception as e:
        print(e)
        await websocket.send_json({"error": f"Error processing audio: {str(e)}"})
        raise e
    finally:
        # Clean up temporary files
        if os.path.exists(temp_raw_path):
            os.remove(temp_raw_path)
        if os.path.exists(temp_wav_path):
            os.remove(temp_wav_path)
        if os.path.exists(f"temp_audio/temp{audio_id}_.wav"):
            os.remove(f"temp_audio/temp{audio_id}_.wav")

def query(content):
    # Define the URL and payload
    url = "http://192.168.1.22:4040/generate/stream"
    payload = {
        "role": "user",
        "content": content
    }
    headers = {
        "Content-Type": "application/json"
    }

    # Send a POST request
    response = requests.post(url, json=payload, headers=headers, stream=True)

    # Check for successful connection
    if response.status_code == 200:
        # Process the SSE stream
        client = sseclient.SSEClient(response)
        print("Listening to SSE stream...")
        for event in client.events():
            print(f"Received event: {event.data}")
    else:
        print(f"Failed to connect: {response.status_code}, {response.text}")

def process_audio(file_path: str) -> str:
    """
    Process the WAV file using Whisper (your existing Whisper processing code).
    """
    # Load and preprocess the audio
    waveform, sampling_rate = torchaudio.load(file_path)

    # Resample the audio if necessary (Whisper models expect 16 kHz audio)
    if sampling_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=16000)
        waveform = resampler(waveform)
        sampling_rate = 16000

    # Preprocess the audio
    waveform = waveform.squeeze().numpy()  # Convert from Tensor to NumPy
    # Ensure the processed waveform has data to avoid errors
    if len(waveform) == 0:
        raise ValueError("No valid audio data after preprocessing (too much noise or silence).")

    # Convert the processed audio to input features
    input_features = processor(waveform, sampling_rate=sampling_rate, return_tensors="pt").input_features

    # Generate transcription
    predicted_ids = model.generate(input_features.to('mps'), language='hi')
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

    return transcription[0]

# Clean up temporary files (Optional)
@app.on_event("shutdown")
def cleanup_temp_files():
    for filename in os.listdir(TEMP_DIR):
        file_path = os.path.join(TEMP_DIR, filename)
        os.remove(file_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=4041)
