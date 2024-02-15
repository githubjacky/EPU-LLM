import pandas as pd
from typing import List, Optional


class MergeQuery:
    def __init__(self, data_path='data/raw/llm_policy_keywords.xlsx'):
        self.data_path = data_path


    @staticmethod
    def __data_generator_split_lower(df: pd.DataFrame):
        """Loop through all the cells within a sheet of the excel.

        Note: usually, this function is used to merge responses of the first
        round of LLM keywords suggestion which aims to reduce the variance
        among responses with the same prompt for one of the EPU cateogories.
        """
        for col in df.columns:
            for cell in df[col]:
                for word in cell.split(' '):
                    yield word.lower()


    @staticmethod
    def __data_generator(df: pd.DataFrame):
        """Loop through all the cells within a sheet of the excel.

        Note: usually, this function is used to merge responses of the second
        round of LLM keywords suggestion which aims to reduce the variance
        among queries between different roles for one of the EPU cateogories.
        """
        for col in df.columns:
            for i in df[col]:
                yield i


    def __freq_dict(self, df: pd.DataFrame, lower = False):
        """Calculate the frequency of each keyword candidate."""
        freq_dict = {}
        if lower:
            for i in self.__data_generator_split_lower(df):
                if i not in freq_dict:
                    freq_dict[i] = 1
                else:
                    freq_dict[i] += 1
        else:
            for i in self.__data_generator(df):
                if i not in freq_dict:
                    freq_dict[i] = 1
                else:
                    freq_dict[i] += 1

        return freq_dict


    def filter_keywords(self,
                        sheet_name: str = 'editor_US',
                        header: Optional[str] = None,
                        lower = False):
        """Merge LLM responses within excel sheet.

        Args:
            sheet_name: the sheet of the excel
            header: header of the sheet, if there isn't any, leave it
            lower: wheter to lower the string of keyword candidates

        Returns:
            keyword list distiled through the major vote within a sheet 
        """
        df = pd.read_excel(
            self.data_path,
            sheet_name=sheet_name,
            header=header
        ).fillna("")

        selected = []
        for key, value in self.__freq_dict(df, lower).items():
            if value >= df.shape[1] // 2 + 1:  # majority vote
                selected.append(key)

        return selected


    def output_excel(self, df: pd.DataFrame, sheet_name: str) -> None:

        with pd.ExcelWriter(path = self.data_path,
                            mode = 'a',
                            if_sheet_exists = 'replace'
                            ) as writer:

            df.to_excel(
                writer,
                sheet_name = sheet_name,
                index = False
            )


    def merge(self, sheet_list: List[str]):
        """Merge responses from different roles. For each role, first merge multiple 
        reponses through major vote, and then use the major vote again to
        merge among different roles. Write the results as a shetinto the excel 
        file.

        Args:
            sheet_list: collection of sheets that recods responses from multiple
                roles

        Returns:
            data frame of merged results
        """
        _res_list = [
            self.filter_keywords(sheet, lower=True)
            for sheet in sheet_list
        ]
        col_name = sheet_list

        max_n_item = max([len(i) for i in _res_list])
        res_list = [
            _res + [""]*(max_n_item-len(_res))
            for _res in _res_list
        ]
        df = pd.DataFrame(
            {
                col_name[i]: res_list[i]
                for i in range(len(col_name))
            }
        )

        new_sheet = f'paradigm vote'
        self.output_excel(df, new_sheet)

        final_res = self.filter_keywords(new_sheet)
        df['vote'] = final_res + [""]*(max_n_item-len(final_res))
        self.output_excel(df, new_sheet)
        print(df.to_latex(index=False))

        return df