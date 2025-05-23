import os
import dotenv
import torch
import warnings

from loguru import logger

warnings.filterwarnings("ignore", category=UserWarning)
logger.add("app.log", format="{time} {level} {message}", level="INFO", rotation="10 MB", compression="zip")
dotenv.load_dotenv()


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32
MODEL_TRANSCRIPTION = "openai/whisper-large-v3" # https://huggingface.co/openai/whisper-large-v3
HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_SPEAKER_DIARIZATION = "pyannote/speaker-diarization@2.1" # https://huggingface.co/pyannote/speaker-diarization
NUM_SPEAKERS = 2
MIN_SEGMENT_DURATION = 0.1
DELTA_DIARIZATION_BUTCH_SIZE = 0.3
