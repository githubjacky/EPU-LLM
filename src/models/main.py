from dotenv import load_dotenv
from utils import Prompt, EPUClassifier, Param

import json
from pathlib import Path


def main():
    load_dotenv()

    param = Param()

    prompt = Prompt(
        param.country,
        param.system_message_template,
        param.question,
        param.examples
    )

    clf = EPUClassifier(
        prompt,
        param.model,
        param.temperature,
        param.batch_size
    )

    data = json.loads(Path(param.data_path).read_text())
    clf.preprocess(data[:100])
    clf.predict()
    clf.output(param.output_path)


if __name__ == "__main__":
    main()
