import os
import torch
from loguru import logger
from pyannote.audio import Pipeline
from pyannote.core import Annotation

import config

class SDModel:
    """
    Class diarization model pyannote
    """
    def __init__(self):
        self.__HF_TOKEN = config.HF_TOKEN
        self.__pipeline = Pipeline.from_pretrained(
          config.MODEL_SPEAKER_DIARIZATION,
          use_auth_token=self.__HF_TOKEN
        )
        self.__pipeline.to(torch.device(config.DEVICE))
        self.__NUM_SPEAKER = config.NUM_SPEAKERS

    def make_speaker_diarization(self, path_to_file: str) -> Annotation:
        """
        Method to diarization audio file
        :param path_to_file: str() path to audio file
        :return: pyannote.core.Annotation diarization data
        Exemple:
        [{'start': 4.82346875, 'end': 8.77221875, 'speaker': 'SPEAKER_01'},
         {'start': 9.14346875, 'end': 17.78346875, 'speaker': 'SPEAKER_00'},
         {'start': 18.17159375, 'end': 20.60159375, 'speaker': 'SPEAKER_01'},
         {'start': 20.83784375, 'end': 21.27659375, 'speaker': 'SPEAKER_00'},
         {'start': 21.78284375, 'end': 22.86284375, 'speaker': 'SPEAKER_01'}]
        """
        if not os.path.exists(path_to_file):
            logger.error(f"File not found: {path_to_file}")
            raise FileNotFoundError(f"File not found: {path_to_file}")

        logger.info(f"Start speaker diarization file {path_to_file}")
        diarization = self.__pipeline(path_to_file, num_speakers=self.__NUM_SPEAKER)
        logger.info(f"End speaker diarization file {path_to_file}")
        torch.cuda.empty_cache()
        return diarization

    @classmethod
    def make_standard_dict(cls, diarization: Annotation) -> list:
        """
        Method to make from answer make_speaker_diarization list()
        :param diarization: result work function make_speaker_diarization
        :return: list()
        """
        segments = []
        for segment, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                "start": segment.start,
                "end": segment.end,
                "speaker": speaker
            })
        return segments

    def work(self, path_to_file: str) -> list:
        """
        Base class to work model and return diarization data
        :param path_to_file: str() path to file to diarization
        :return: list() result data diarization
        """
        result = self.make_speaker_diarization(path_to_file)
        return self.make_standard_dict(result)
