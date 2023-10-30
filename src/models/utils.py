from dotenv import load_dotenv
from loguru import logger
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from pathlib import Path
import orjson
import openai
from os import getenv
from typing import List, Dict


def env_setup() -> str:
    load_dotenv()
    key = getenv("OPENAI_API_KEY")

    if key is None:
        key = input("input your OpenAI API key: ")

        file = Path(".env")
        if file.is_file():
            with file.open("a") as f:
                f.write(f"\nOPENAI_API_KEY={key}")
        else:
            with file.open("w") as f:
                f.write(f"OPENAI_API_KEY={key}")

    return key


def read_jsonl(
    path: str | Path, 
    n: int = -1, 
    return_str: bool = False
) -> List[Dict] | List[str]:
    return (
        [orjson.loads(i) for i in Path(path).read_text().split("\n")[:n]]
        if not return_str
        else [i for i in Path(path).read_text().split("\n")[:n]]
    )


def format_handler(
    chain: LLMChain,
    i: int,
    instruction: str = "",
    strategy: str = "few_shot_with_reason",
):
    while True:
        try:
            res = orjson.loads(chain.run(correct_instructions=instruction))
            if (
                strategy in ["few_shot_with_reason", "zero_shot_with_reason"]
                and res.get("pred") is not None
                and res.get("reason") is not None
            ):
                break
            elif strategy in ["few_shot", "zero_shot"] and res.get("pred") is not None:
                break
            else:
                logger.info(
                    f"formatting error(KeyError) for {i} th sample, re-generate"
                )
                instruction = " ".join(
                    (
                        "Your answer which is a json string don't",
                        "have the specified key. Follow the schema carefully.",
                    )
                )
                continue

        except orjson.JSONDecodeError:
            logger.info(
                f"formatting error(JSONDecodeError) for {i} th sample, re-generate"
            )
            instruction = " ".join(
                (
                    "Formatting error. It might because",
                    "not all single quotes have been escaped or",
                    "the answering has been truncated.ry to answer precisely",
                    "and reduce the number of token.",
                )
            )
            continue

        except openai.error.InvalidRequestError:
            res = {"pred": -1}
            break

    return res


def refresh_handler(
    chain: LLMChain, instruction: str, i: int, strategy: str = "few_shot_with_reason"
):
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    chain.memory = memory
    chain.llm = ChatOpenAI(
        model="gpt-3.5-turbo-16k", temperature=0.6, request_timeout=120
    )
    return chain, format_handler(chain, i, instruction, strategy)
