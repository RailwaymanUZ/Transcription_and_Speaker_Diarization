import argparse

from model_worker import Worker


def main() -> dict:
    parser = argparse.ArgumentParser(
        description="Make transcription and speaker diarization"
    )
    parser.add_argument(
        "path_to_file",
        type=str,
        help="Path to audio file"
    )
    parser.add_argument(
        "output_dir",
        type=str, nargs="?",
        default="src/",
        help="Path to dir to save result in .json"
    )

    args = parser.parse_args()
    return vars(args)


if __name__ == "__main__":
    param = main()
    worker = Worker()
    worker.result(**param)




