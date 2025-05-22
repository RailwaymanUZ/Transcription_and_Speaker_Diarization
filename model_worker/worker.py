import os
import json
from loguru import logger

import config
from transcription import Transcriber
from speaker_diarization import SDModel
from pre_processor import PreProcessor

class Worker:
    def __init__(self):
        self.__transcriber = Transcriber()
        self.__sd_model = SDModel()
        self.__MIN_SEGMENT_DURATION = config.MIN_SEGMENT_DURATION
        self.__DELTA_DIARIZATION_BUTCH_SIZE = config.DELTA_DIARIZATION_BUTCH_SIZE

    def filter_diarization_data(self, result_diarization: list) -> list:
        """
        Method return filtered segments after diarization
        :param result_diarization: result
        :return: list() with dict()
        """
        filtered_result = [
            seg for seg in result_diarization
            if seg['end'] - seg['start'] > self.__MIN_SEGMENT_DURATION
        ]
        return filtered_result


    def merge_diarization_with_transcription(self, diarization: list, transcription: list) -> list:
        """
        Method to make 1 data list merged diarization data and transcription data.
        :param: diarization list() diarization data
        :param: transcription list() transcription data
        :param: delta time to -start time and + end time.
        :return: list() with merged data
        """
        merged = []
        for seg in diarization:
            seg_start = seg['start'] - self.__DELTA_DIARIZATION_BUTCH_SIZE
            seg_end   = seg['end']   + self.__DELTA_DIARIZATION_BUTCH_SIZE

            words = [
                    w['text']
                    for w in transcription
                    if w.get('start') is not None
                       and w.get('end') is not None
                       and not (w['start'] < seg_start or w['end'] > seg_end)
                ]

            text = ' '.join(words).strip()

            merged.append({**seg, 'text': text})
        return merged

    def work_models(self, path_to_file: str) -> list:
        """
        Method to make transcription, postprocessing and preprocessing
        :param path_to_file: str() path to file
        :return: list() merged data after speaker diarization ant transcription
        """
        current_file = os.path.abspath(__file__)
        parent_dir = os.path.dirname(os.path.dirname(current_file))
        target_dir = os.path.join(parent_dir, "src")

        try:
            path_to_file = PreProcessor.resampling(path_to_file, target_dir)
        except Exception as e:
            logger.error(f"Check the audio file. Error when open. {e}")
            raise e

        diarization_data = self.__sd_model.work(path_to_file)
        transcription_data = self.__transcriber.work(path_to_file)
        diarization_data = self.filter_diarization_data(diarization_data)
        result = self.merge_diarization_with_transcription(diarization_data, transcription_data)
        return result

    def result(self, path_to_file: str, output_dir: str = "src/") -> None:
        """
        Method to call method work_models and save result in .json
        :param path_to_file: str() path to file
        :param output_dir: output dir to save .json file. Default = src/ directory project
        :return:
        """
        if not os.path.exists(path_to_file):
            logger.error(f"File not found: {path_to_file}")
            raise FileNotFoundError(f"File not found: {path_to_file}")

        result_work = self.work_models(path_to_file)
        file_name = f"{output_dir}result.json"
        with open(file_name, "w", encoding="utf-8") as f:
            json.dump(result_work, f, ensure_ascii=False, indent=4)
        logger.info(f"Transcription in file {file_name}")
