import os
import torch
from loguru import logger
from transformers import pipeline, AutoProcessor

from .abstract_class_transcription import AbstractTranscriber
import config


class TranscriberHF(AbstractTranscriber):
    """
    Class to transcription audio file.
    Used HF.
    """
    def __init__(self):
        self.__MODEL_ID = config.MODEL_TRANSCRIPTION
        self.__device = config.DEVICE
        self.__torch_dtype = config.TORCH_DTYPE
        self.__processor = AutoProcessor.from_pretrained(self.__MODEL_ID)
        self.__pipe = pipeline(
            "automatic-speech-recognition",
            model=self.__MODEL_ID,
            tokenizer=self.__processor.tokenizer,
            feature_extractor=self.__processor.feature_extractor,
            torch_dtype=self.__torch_dtype,
            device=self.__device,
            generate_kwargs={
                "is_multilingual": True,
                "condition_on_prev_tokens": True,
                "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
                "compression_ratio_threshold": 2.4,
                "logprob_threshold": -1.0,
            },
            return_timestamps="word",   # This parameter return word timestamp
            chunk_length_s=30,  # This parameter return chunk size second to model ! THIS PARAMETERS IN FUTURE IN EXPERIMENT
            stride_length_s=5   # Stride seconds to chunk
        )

    def make_transcription(self, path_to_file: str) -> dict:
        """
        Method to get transcription.
        :param path_to_file: path to file
        :return: dict() result transcription
        Exemple
        {
        'text': 'Возьмите, я з.....'
         'chunks': [{'text': ' Возьмите,', 'timestamp': (0.0, 5.46)},
                    {'text': ' я', 'timestamp': (5.46, 5.54)}]
        }
        """
        if not os.path.exists(path_to_file):
            logger.error(f"File not found: {path_to_file}")
            raise FileNotFoundError(f"File not found: {path_to_file}")
        logger.info(f"Start transcription {path_to_file}")
        result = self.__pipe(path_to_file)
        logger.info(f"Transcription file {path_to_file} success")
        torch.cuda.empty_cache()
        return result

    @classmethod
    def make_to_standards(cls, result_transcription: dict) -> list:
        """
        Method to make dict to standards
        :param result_transcription: list() with dict
        on answer make_transcription ['chunks'] key
        :return: list with dict renamed elements.
        """
        words = []
        for segment in result_transcription["chunks"]:
            words.append({
                "start": segment['timestamp'][0],
                "end": segment['timestamp'][-1],
                "text": segment['text']
            })
        return words
