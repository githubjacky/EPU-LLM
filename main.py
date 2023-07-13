from typing import List, Dict
from pathlib import Path
from tqdm import trange
from itertools import repeat
from sklearn.metrics import classification_report
import openai
import json
import os



class EPUClassifier:
    def __init__(
            self, 
            api_key: str | None, 
            prompt: List[Dict[str, str]], 
            pad: str,
            model: str
    ) -> None:
        self.api_key = api_key
        self.prompt = prompt
        self.pad = pad
        self.model = model


    def preprocess(self, path: str) -> List[Dict]:
        return json.loads( Path(path).read_text() )


    def predict(self, path: str) -> None:
        openai.api_key = self.api_key
        data = self.preprocess(path)
        articles = [i['article'] for i in data]

        n = len(data)
        preds = list(repeat(0, n))
        responses = list(repeat('', n))

        for i in trange(n):
            content = f'{self.pad}\n"""\n{articles[i]}\n"""\n'
            messages = self.prompt + [{"role": "user", "content": content}]

            completion = openai.ChatCompletion.create(
                model = self.model,
                messages = messages,
                temperature = 0  # force the response to be deterministic
            )
            response = completion.choices[0].message.content.split(';')
            preds[i] = int(response[0])
            responses[i] = response[1]

        labels = [i['label'] for i in data]
        print( classification_report(labels, preds, zero_division = 1.) )

        self.articles = articles
        self.labels = labels
        self.preds = preds
        self.responses = responses


    def output(self, path: str) -> None:
        res = []
        for i in range(len(self.articles)):
            res.append({
                "article": self.articles[i],
                "label": self.labels[i],
                "pred": self.preds[i],
                "response": self.responses[i]
            })

        with open(path, 'w') as f:
            json.dump(res, f)


def main():
    persona = '''
    As an economist working on constructing Taiwan's Economic Policy Uncertainty Index (EPU index), 
    your task is to identify articles that contain EPU keywords but should not be included 
    when measuring Economic Policy Uncertainty. In other words, you need to identify the 
    noise that should be excluded when constructing Taiwan's EPU index. There are two 
    aspects you need to consider. First, you need to determine whether the main idea of the 
    article is introducing news that will impact Taiwan's economic environment rather than 
    providing historical accounts. Second, you should assess whether the events mentioned 
    in the article actually occur within Taiwan. If the events do not occur within Taiwan, 
    they should be excluded. Your response should maintain a consistent style and consist 
    of two parts separated by a semicolon. The first part should indicate whether the given 
    article is considered to be noise and thus should be excluded, with a value of 1 for 
    yes and 0 for no. The second part should provide your reason for such classification.
    '''

    question = "Should the following article be excluded when constructing Taiwan's EPU index?"

    example1_input = f'''{question}
    """
    半工半讀掙錢幫家裡，減輕媽媽經濟負擔
    """
    '''
    example1_output = "1; Yes, it should be excluded. Although the article contains keywords, it has nothing to do with the Taiwan's economic environment."

    example2_input = f'''{question}
    """
    中國商業氣氛降至低點，習近平主導的中國市場「不再需要外國人了」
    """
    '''
    example2_output = "1; Yes, it should be excluded, as it does not mention any economic policy uncertainty events in Taiwan."

    example3_input = f'''{question}
    """
    美國聯準會與台灣經濟都有一個「6月難題」
    """
    '''
    example3_output = "0; NO, it shouldn't be excluded. It introduces the policy uncertainty of Taiwan."

    prompt = [
        {"role": "system", "content": persona},
        {"role": "user", "content": example1_input},
        {"role": "assistant", "content": example1_output},
        {"role": "user", "content": example2_input},
        {"role": "assistant", "content": example2_output},
        {"role": "user", "content": example3_input},
        {"role": "assistant", "content": example3_output},
    ]

    clf = EPUClassifier(
        api_key = os.getenv('GPTAPI'),
        prompt = prompt,
        pad = question,
        model = "gpt-3.5-turbo",
    )
    clf.predict("./data/EPU_Noise_Examples.json")
    clf.output("./data/pred.json")

    
if __name__ == "__main__":
    main()
