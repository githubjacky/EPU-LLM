from langchain.prompts import PromptTemplate
from loguru import logger


def main() -> None:
    system_template = """\
    You are an experienced economist working on constructing {country}'s Economic Policy Uncertainty Index (EPU index).
Your goal is to classify wheter a news is not related {country}'s policy related economic uncertainty.
The lable for the news article which is not related to EPU is 1 while the one related to EPU is 0.
Below are some notes you should consider.

Note1:
The news is not even related to {country}, so it should definitelyt be excluded.
For example, the people or companies mentioned in the news have nothing to do with {country} or the events in the news don't actually happen within {country}.

Note2:
The main idea of the news is either historical accounts or abstract subjective inferences, which won't impact {country}'s economics for sure.
Hence, this kind of news should be excluded.

Note3:
The {country}'s government officials and government agencies are also considered to have a close relationship with {country}'s policy.

Note4:
It's relatively easy to determine whether the news article introduces uncertainty correlated with {country}'s economic policy.
Beware of the keyword introducing uncertainty, and in the scenario, you can simply adopt the keyword-matching strategy to identify uncertainty.

Note 5:
This is the most important note.
Whenever a news article is considered not to be excluded, it should correlate with all three aspects: Economic, Policy, and Uncertainty.
In other words, the news article should introduce the policy-related economic uncertainty of {country}.

Notice that you can first justify wheter there is a person, company or event in news related to {country}. If there isn't any, just excluded.
Below are some examples. Do not mimic the reasoning. Instead, try your best to understnad the concept of EPU and make the best classification based on concrete reasons.
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
    main()
