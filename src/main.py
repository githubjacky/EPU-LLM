from pydantic import BaseModel
from utils import Prompt, EPUClassifier, Param



def main():
    param = Param()

    prompt = Prompt(
        param.country,
        param.system_message_template,
        param.question,
        param.examples
    )

    clf = EPUClassifier(
        prompt,
        param.openai_api_key,
        param.model,
        param.temperature
    )
    clf.predict(param.data_path)
    clf.output(param.output_path)

    
if __name__ == "__main__":
    main()
