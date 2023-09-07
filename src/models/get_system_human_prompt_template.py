from langchain import PromptTemplate
from loguru import logger


def run() -> None:
    system_template = """\
    I am an economist working on constructing {country}'s Economic Policy Uncertainty Index (EPU index).
My primary goal is to classify wheter a news should be excluded when constructing EPU index in {country}.
There are two criteria I'm considering to exclude a news.

Criterion1:
The main idea of the news is either historical accounts or abstract subjective inferences, which won't impact {country}'s economics for sure.
Hence, this kind of news should be excluded.
Notice that you should access wheter the news is related to {country}'s enconomics by yourself, insead of relying on the conclusion of the news.

Criterion2:
There main idea of the news is not related with {country}.
For example, the people or companies mentioned in the news have nothing to do with {country} or the events
in the news don't actually happen within {country}. I will excluded the news as well.

Help me complete the classification task identifying what type of news should be excluded.
Notice that you can first justify wheter there is a person, company or event in news related to {country}.
If there isn't any, it should be excluded with high probability.
"""
    system_prompt_template = PromptTemplate.from_template(system_template)

    human_template = """\
{correct_instructions}
Question: Should the following article be excluded when constructing EPU index?

{news}

Output Instructions:
{output_instructions}
Besides, don't forget to escape a single quote in the reason section.
"""
    human_prompt_template = PromptTemplate.from_template(human_template)

    logger.info("save the prompt template")
    human_prompt_template.save("prompt_template/human.json")
    system_prompt_template.save("prompt_template/system.json")


if __name__ == "__main__":
    run()
