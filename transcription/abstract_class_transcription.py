from abc import ABC, abstractmethod



class AbstractTranscriber(ABC):

    @abstractmethod
    def make_to_standards(self, result_transcription: dict):
        pass

    @abstractmethod
    def make_transcription(self, path_to_file: str):
        pass

    def work(self, path_to_file: str) -> list:
        """
        Method base function to run model and take result.
        :param path_to_file: str() path to file
        :return: list() with dict
        Exemple
        [{'text': ' Возьмите,', 'start': 0.0, 'end': 5.46},
        {'text': ' я', 'start': 5.46, 'end': 5.54}]
        """
        result = self.make_transcription(path_to_file)
        return self.make_to_standards(result)
