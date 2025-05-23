# 🗣️ Speech Recognition Project

This project takes an audio file and transcribes the speech into text and make speaker Diarization.
The final output is saved in JSON format.

## ✅ Prerequisites

- Python 3.8 or later
- `pip` package manager
- GPU with CUDA support and min 12 gb VRAM
- ffmpeg (for audio conversion, if required)
- A Hugging Face API token (set as the environment variable `HF_TOKEN`) or create .env file with this variable.
- Take access to model Speaker Diarization https://huggingface.co/pyannote/speaker-diarization 

## ⚙️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/RailwaymanUZ/Transcription_and_Speaker_Diarization
   cd Transcription_and_Speaker_Diarization
   ```
   ```bash
   pip install -r requirements.txt
   ```
    ```bash
    pip uninstall triton -y
    ```
   triton now dont stable
    ```bash
    apt install ffmpeg -y
    ```
   If ffmpeg don't install on your PC

   Set the environment variable `HF_TOKEN`
   ```bash
     export HF_TOKEN="your_token_here"
   ```
   or create in directory project .env file with this variable
   ```bash
     HF_TOKEN="your_token_here"
   ```

## 🚀 Usage
If your want to save .json on src dir on project use:
   ```bash
     python app.py path_to_audio_file
   ```
If save in other dir append the dir to save as parameters:
   ```bash
     python app.py path_to_audio_file path_to_dir_save
   ```

### 👨‍💻 Developer
Valera Lemeshko<br>
Email - lemeshkovalerij@gmail.com
