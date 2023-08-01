class ClassificationResult(BaseModel):
    pred: int = Field(
        description="If the given article is to be included, return 0; if not, return 1.")
    reason: str = Field(
        description="reason for why or why not it should be included for constructing EPU index")


class Prompt():
    example_prompt_template = """Quesion: {question}

    {article}

    Response: {response}
    """

    # suffix of the example prompt
    suffix = """Quesion: {question}

    {article}

    {format_instructions}
    """

    parser = PydanticOutputParser(pydantic_object=ClassificationResult)

    def __init__(self,
                 country: str,
                 system_message: str,
                 question: str,
                 examples: List[Dict[str, str]],
                 ):

        example_prompt = PromptTemplate(
            input_variables=['article', 'response'],
            partial_variables={'question': question},
            template=self.example_prompt_template
        )

        fewshot_prompt = FewShotPromptTemplate(
            input_variables=['article'],
            examples=examples,
            example_prompt=example_prompt,
            suffix=self.suffix,
            partial_variables={
                'question': question,
                'format_instructions': self.parser.get_format_instructions()
            }
        )

        messages = [
            SystemMessagePromptTemplate.from_template(
                system_message).format(country=country),
            HumanMessagePromptTemplate(prompt=fewshot_prompt)
        ]

        self.chat_message = ChatPromptTemplate(
            input_variables=['article'],
            output_parser=self.parser,
            messages=messages
        )
