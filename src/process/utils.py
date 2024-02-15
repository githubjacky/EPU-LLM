import pandas as pd
from .query_keywords import MergeQuery


def compare_country(category, paradigm, countries, model):
    dict = {}
    for country in countries:
        fpath = f'data/processed/auto-epu/{country}/{model}/llm_keywords.xlsx'
        df = pd.read_excel(fpath, sheet_name=category)
        dict[country] = df.get(paradigm)
    df = pd.DataFrame(dict).fillna('')

    print(df.to_latex(index=False))

    return df


def compare_human(category, paradigm, countries, model):
    dict = {}
    for country in countries:
        fpath = f'data/processed/auto-epu/{country}/{model}/llm_keywords.xlsx'
        llm = pd.read_excel(fpath, sheet_name=category).get(paradigm).dropna()
        paper = (
            pd
            .read_excel('data/processed/auto-epu/paper_keywords.xlsx', sheet_name=category)
            .get(country)
            .dropna()
        )
        dict[country] = llm.isin(paper).sum() / len(paper) * 100
    
    return dict


def rm_empty(x):
    return list(filter(lambda a: a!= '', x))

def add_col(category, paradigm, country, model):
    agent = MergeQuery(f'data/processed/auto-epu/{country}/{model}/llm_{category}_keywords.xlsx')
    target_col = rm_empty(agent.filter_keywords(paradigm))
    raw_data_path = f'data/processed/auto-epu/{country}/{model}/llm_keywords.xlsx'
    df = pd.read_excel(raw_data_path, sheet_name=category)
    
    n_df = len(df)
    n_col = len(target_col)
    if n_col > n_df:
        N = n_col - n_df
        dict = {}
        for col in df.columns:
            dict[col] = df.get(col).to_list() + ['']*N
        dict[paradigm] = target_col
        df = pd.DataFrame(dict)
    else:
        df[paradigm] = target_col + ['']*(n_df-n_col)

    MergeQuery(raw_data_path).output_excel(df, category)

    return df