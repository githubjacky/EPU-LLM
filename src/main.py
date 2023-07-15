from utils import Prompt, EPUClassifier


def main():
    instructions = [
        {
            'news': "半工半讀掙錢幫家裡，減輕媽媽經濟負擔",
            'response': "1; Yes, it should be excluded. Although the article contains keywords, it has nothing to do with the Taiwan's economic environment."
        },
        {
            'news': "中國商業氣氛降至低點，習近平主導的中國市場「不再需要外國人了」",
            'response': "1; Yes, it should be excluded, as it does not mention any economic policy uncertainty events in Taiwan."

        },
        {
            'news': "美國聯準會與台灣經濟都有一個「6月難題」",
            'response': "0; NO, it shouldn't be excluded. It introduces the policy uncertainty of Taiwan."
        }
    ]
    prompt = Prompt('Taiwan', instructions)

    clf = EPUClassifier(prompt.template, prompt.pad, "gpt-3.5-turbo-16k")


    clf.predict("./data/EPU_Noise_Test.json")
    clf.output("./data/pred_formal.json")

    
if __name__ == "__main__":
    main()
