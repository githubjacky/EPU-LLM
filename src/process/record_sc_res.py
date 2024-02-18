from ckip_transformers.nlp import CkipWordSegmenter
import orjson
from pathlib import Path
import pandas as pd
from typing import Optional



class RECORD_SC_RES:
    def __init__(self,
                 country: str,
                 model: str,
                 language_model: str = 'bert-base',
                 device: Optional[int] = None):
        self.ws_driver  = CkipWordSegmenter(
            model = language_model,
            device = device if device is not None else 0
        )
        self.country = country
        self.model = model
        self.i = 0


    def run(self,
            title: str,
            news: str,
            econ_word: str,
            policy_word: str,
            uncertainty_word: str):
        res = self.ws_driver([news, econ_word, policy_word, uncertainty_word])

        econ_df = pd.read_excel(
            f'data/processed/auto-epu/{self.country}/{self.model}/economic.xlsx', 
            sheet_name='definition with SC', header=None
        )[self.i].to_list()
        policy_df = pd.read_excel(
            f'data/processed/auto-epu/{self.country}/{self.model}/policy.xlsx', 
            sheet_name='definition with SC', header=None
        )[self.i].to_list()
        uncertainty_df = pd.read_excel(
            f'data/processed/auto-epu/{self.country}/{self.model}/uncertainty.xlsx', 
            sheet_name='definition with SC', header=None
        )[self.i].to_list()

        item = {
            'title': title,
            'news': news,
            'econ_word': econ_word,
            'policy_word': policy_word,
            'uncertainty_word': uncertainty_word,
            'is_econ_in_news': bool(sum([i in res[0] for i in res[1]])),
            'is_policy_in_news': bool(sum([i in res[0] for i in res[2]])),
            'is_uncertainty_in_news': bool(sum([i in res[0] for i in res[3]])),
            'is_econ_in_def': econ_word in econ_df,
            'is_policy_in_def': policy_word in policy_df,
            'is_uncertainty_in_def': uncertainty_word in uncertainty_df
        }

        fpath = Path(f'data/processed/auto-epu/{self.country}/{self.model}/sc_eval.jsonl')
        if not fpath.exists():
            with fpath.open('w') as f:
                f.write(orjson.dumps(item, option=orjson.OPT_APPEND_NEWLINE).decode())
        else:
            with fpath.open('a') as f:
                f.write(orjson.dumps(item, option=orjson.OPT_APPEND_NEWLINE).decode())
        self.i += 1
