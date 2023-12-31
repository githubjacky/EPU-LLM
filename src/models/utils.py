from dotenv import load_dotenv
from loguru import logger
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.runnables.base import RunnableSequence
from langchain.callbacks.tracers import ConsoleCallbackHandler
from pathlib import Path
import orjson
from os import getenv
from operator import itemgetter
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
        else
        [i for i in Path(path).read_text().split("\n")[:n]]
    )


def log_init(log_file_path: Path, exist_ok = True):
    if log_file_path.exists() and not exist_ok:
        log_file_path.unlink()
    logger.remove()
    logger.add(log_file_path, level = "INFO")


def add_memory(chain: RunnableSequence):
    memory = ConversationBufferMemory(return_messages=True)
    return (
        RunnablePassthrough.assign(
            history=RunnableLambda(memory.load_memory_variables) | itemgetter('history')
        )
        | chain
    )


def format_handler(chain: RunnableSequence,
                   i: int, # the ith news
                   instruction: str = "",
                   strategy: str = "few_shot_with_reason",
                   verbose = False
                   ):
    retry = True
    while retry:
        try:
            res = (
                chain.invoke(
                    {'correct_instructions': instruction},
                    config={'callbacks': [ConsoleCallbackHandler()]}
                )
                if verbose
                else
                chain.invoke({'correct_instructions': instruction})
            )
            if (
                strategy in ["few_shot_with_reason", "zero_shot_with_reason"]
                and res.get("pred") is not None
                and res.get("reason") is not None
            ):
                retry = False
            elif (
                strategy in ["few_shot", "zero_shot"]
                and res.get("pred") is not None
            ):
                retry = False
            else:
                logger.info(f"formatting error(KeyError) for {i+1} th sample, re-generate")
                instruction = " ".join((
                    "Your answer which is a json string don't",
                    "have the specified key. Follow the schema carefully.",
                ))

        except orjson.JSONDecodeError:
            logger.info(f"formatting error(JSONDecodeError) for {i+1} th sample, re-generate")
            instruction = " ".join((
                "Formatting error. It might because",
                "not all single quotes have been escaped or",
                "the answering has been truncated.ry to answer precisely",
                "and reduce the number of token.",
            ))

    return res
