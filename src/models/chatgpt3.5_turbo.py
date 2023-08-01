from prompt import Prompt


class ChatGPT:
    def __init__(self, prompt: Prompt, model: str, temperature: float, batch_size: int) -> None:
        self.batch_size = batch_size
        llm = ChatOpenAI(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            model=model,
            temperature=temperature
        )
        self.chat = LLMChain(llm=llm, prompt=prompt.chat_message)

    def preprocess(self, source: str | List[Dict]) -> None:
        if isinstance(source, str):
            self.data = json.loads(Path(source).read_text())
        else:
            self.data = source

    def predict_instance(self, article: str) -> ClassificationResult:
        return self.chat.predict_and_parse(article=article)

    def predict_batch(self, batch_articles: List[str]) -> List[ClassificationResult]:
        input_list = [
            {'article': i}
            for i in batch_articles
        ]
        return self.chat.apply_and_parse(input_list)

    def predict(self) -> None:
        n = len(self.data)
        predictions = []

        if n >= self.batch_size:
            for idx in trange(0, n, self.batch_size):
                articles = [
                    i['article']
                    for i in self.data[idx:min(idx+self.batch_size, n)]
                ]
                predictions.append(self.predict_batch(articles))

            predictions = chain.from_iterable(predictions)
        else:
            for i in trange(n):
                predictions.append(
                    self.predict_instance(self.data[i]['article']))

        preds = [i.pred for i in predictions]
        labels = [i['label'] for i in self.data]
        print(classification_report(labels, preds, zero_division=1.))

        self.predictions = predictions

    def output(self, path: str) -> None:
        res = [json.loads(i.json()) for i in self.predictions]
        with open(path, 'w') as f:
            json.dump(res, f)
