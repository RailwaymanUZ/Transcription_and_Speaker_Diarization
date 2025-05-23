import os
from loguru import logger
from pydub import AudioSegment




class PreProcessor:
    """
    Class with static method to preprocessing
    """

    @staticmethod
    def resampling(
            audio_path: str,
            output_dir: str,
            output_filename: str = "res_audio.mp3",
            sample_rate=16000
    ):
        """
        Method to resampling audio. Make semple rate = 16000 and save on new file
        :param audio_path: str() path to audio file
        :param output_dir: str() path to output dir
        :param output_filename: str() output file name
        :param sample_rate: str() semple rate final file default 16000
        :return:
        """
        if not os.path.exists(audio_path):
            logger.error(f"Video not found: {audio_path}")
            raise FileNotFoundError(f"Video not found: {audio_path}")

        output_path = os.path.join(output_dir, output_filename)
        audio = AudioSegment.from_file(audio_path)
        original_sample_rate = audio.frame_rate
        if original_sample_rate != sample_rate:
            audio = audio.set_frame_rate(sample_rate).set_channels(1)
            audio.export(output_path, format="mp3", bitrate="128k")
            logger.info(f"Access resampling audio -{output_path}")
            return output_path
        else:
            logger.info(f"File {audio_path} semple rate - {sample_rate}")
            return audio_path


    @staticmethod
    def extract_audio(
            video_path: str,
            output_dir: str,
            output_filename: str = "audio.mp3",
            sample_rate=16000
    ):
        """
        Method to extract audio from video and save audio file with semple rate = 16000
        :param video_path: str() path to video file
        :param output_dir: str() path to output dir
        :param output_filename: str() output file name
        :param sample_rate: str() semple rate final file default 16000
        :return:
        """
        if not os.path.exists(video_path):
            logger.error(f"Video not found: {video_path}")
            raise FileNotFoundError(f"Video not found: {video_path}")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        output_path = os.path.join(output_dir, output_filename)

        audio = AudioSegment.from_file(video_path)
        audio = audio.set_frame_rate(sample_rate)
        audio.export(output_path, format="mp3")
        logger.info(f"Access extract audio -{output_path}")
