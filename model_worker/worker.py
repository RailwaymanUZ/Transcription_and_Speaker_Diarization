import os
import json
from loguru import logger

from transcription import Transcriber
from speaker_diarization import SDModel
from pre_processor import PreProcessor
from post_processor import PostProcessor

class Worker:
    def __init__(self):
        self.__transcriber = Transcriber()
        self.__sd_model = SDModel()


    @classmethod
    def _postprocessing(cls, transcription_data: list, sd_data: list) -> list:
        """
        Method to postprocessing result models and merge.
        :param transcription_data: list() with dict() result transcription.
        :param sd_data: list() with dict() result Speaker Digitization.
        :return: list() postprocessing result
        """
        sd_data = PostProcessor.separate_overlap(sd_data)
        merge = PostProcessor.merged_result(sd_data=sd_data, tr_data=transcription_data)
        merge = PostProcessor.group_by_speaker(merge)
        return merge


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
        result = self._postprocessing(transcription_data=transcription_data, sd_data=diarization_data)
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

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        file_name = f"{output_dir}result.json"
        with open(file_name, "w", encoding="utf-8") as f:
            json.dump(result_work, f, ensure_ascii=False, indent=4)
        logger.info(f"Transcription in file {file_name}")
