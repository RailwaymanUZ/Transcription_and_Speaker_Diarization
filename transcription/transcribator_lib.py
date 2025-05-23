import os
import whisper
from loguru import logger

import config
from .abstract_class_transcription import AbstractTranscriber


class TranscriberLib(AbstractTranscriber):
    """
    Class to transcription audio file.
    Used openai-whisper lib.
    """
    def __init__(self):
        self.__MODEL_ID = "-".join(config.MODEL_TRANSCRIPTION.split("/")[-1].split("-")[1:])
        self.__kwargs = {
            "word_timestamps": True,
            "condition_on_previous_text": False

        }
        self.__model = whisper.load_model(self.__MODEL_ID)


    def make_transcription(self, path_to_file: str) -> dict:
        """
        Method to get transcription.
        :param path_to_file: path to file
        :return: dict() result transcription
        Exemple
        {
        'text': ' Добрый день, меня ...',
        'segments': [
        {'id': 0,
       'seek': 0,
       'start': np.float6
         'end': np.float64(8.42),
       'text': ' Добрый день, меня зовут Алена, я эксперт в бизнес-технике, я могу помочь?',
       'tokens': [50365, 3401, 13829, ...]
        'temperature': 0.0,
       'avg_logprob': -0.25669393626921766,
       'compression_ratio': 1.9259259259259258,
       'no_speech_prob': 0.13781683146953583,
       'words': [
       {'word': ' Добрый',
       'start': np.float64(4.679999999999997),
       'end': np.float64(5.14),
       'probability': np.float64(0.8027739326159159)},
       {'word': ' день,',
       'start': np.float64(5.14),
       'end': np.float64(5.24),
       'probability': np.float64(0.9944362640380859)},
        {'word': ' меня',
        'start': np.float64(5.3),
        'end': np.float64(5.44),
        'probability': np.float64(0.9985602498054504)},
        ...
        ]...]
        }
        :param path_to_file:
        :return:
        """
        if not os.path.exists(path_to_file):
            logger.error(f"File not found: {path_to_file}")
            raise FileNotFoundError(f"File not found: {path_to_file}")
        logger.info(f"Start transcription {path_to_file}")
        result = self.__model.transcribe(audio=path_to_file, **self.__kwargs)
        logger.info(f"Transcription file {path_to_file} success")
        return result

    @classmethod
    def make_to_standards(cls, result_transcription: dict) -> list:
        """
        Method to make dict to standards
        :param result_transcription: dict() answer function make_transcription
        :return: list with dict renamed elements and make to standards
        """
        words = [
            {
                "start": float(element['start']),
                "end": float(element['end']),
                "text": element['text']
            }
            for element in result_transcription['segments']
        ]
        return words
