class PostProcessor:

    @staticmethod
    def separate_overlap(data: list) -> list:
        """
        Method to separated overlap if exist in Speaker Diarization data
        :param data: list() with dict() result speaker diarization
        :return: list() with dict() separated overlap Speaker
        """
        new_data = [data[0]]
        for element in data[1:]:
            if element["start"] < new_data[-1]["end"]:
                new_end = new_data[-1]["end"]
                speaker = new_data[-1]["speaker"]
                new_data[-1]["end"] = element["start"] - 0.1
                new_data.append(element)
                new_data.append({"start": element["end"] + 0.1, "end": new_end, "speaker": speaker})
            else:
                new_data.append(element)

        return new_data


    @staticmethod
    def merged_result(sd_data: list, tr_data: list) -> list:
        """
        Method to merged result model Speaker diarization data and Transcription data.
        :param sd_data: str() with dict() result Speaker diarization model.
        :param tr_data: str() with dict() result Transcription model.
        :return: list() with dict() merged result.
        """
        result = []

        for text in tr_data:
            best_match = None
            max_overlap = 0

            for speaker in sd_data:
                overlap_start = max(text['start'], speaker['start'])
                overlap_end = min(text['end'], speaker['end'])
                overlap_duration = max(0, overlap_end - overlap_start)

                if overlap_duration > max_overlap:
                    max_overlap = overlap_duration
                    best_match = speaker

            if not best_match:
                best_match = min(sd_data, key=lambda s: abs(text['start'] - s['start']))

            result.append({
                'start': text['start'],
                'end': text['end'],
                'text': text['text'],
                'speaker': best_match['speaker']
            })

        return result

    @staticmethod
    def group_by_speaker(data: list) -> list:
        """
        Method to group speaker text.
        :param data: list() with dict() result merge model
        :return: list() with group speaker
        """
        new_data = [data[0]]
        for element in data[1:]:
            if element["speaker"] == new_data[-1]["speaker"]:
                new_data[-1]["text"] += element["text"]
                new_data[-1]["end"] = element["end"]
            else:
                new_data.append(element)

        return new_data
