import random
from dataclasses import dataclass, field

from typing import Generator, Dict, Optional, Literal, TypedDict, List, Any, Set
from typing import cast

import numpy as np
from tqdm.auto import tqdm
from langchain_core.language_models import BaseChatModel

from pathlib import Path

from client.concrete.sparq_mutation_gen import MutationGenerator

from util.file_tools import JsonHandler, JsonlHandler
from util.path_tools import OutputPathCreator
from util.path_tools import SuffixFilteredPathBuilder

from logging import getLogger, NullHandler

logger = getLogger(__name__)
logger.addHandler(NullHandler())


@dataclass
class MutationGenerateConfig:
    """
    seedとなる情報を読み込む際、一つのファイルに複数のデータが含まれるケースと、１つのファイルに１つのデータが含まれるケースがある。
    MATHは後者であるため、処理するファイルを指定するのではなく、複数の対象ファイルが格納されたディレクトリを指定してまとめて処理する。
    {
        "problem": "How many ...",
        "level": "Level 3",
        "type": "Algebra",
        "solution": "The denominator of the ...",
    }
    ファイルを読み込んで、データ処理する際に必要になる情報をdataclassとしてまとめて扱うことにする
    """
    input_dir: str
    output_dir: str
    seed_data_keys: Set[str] = field(default_factory=lambda: {"problem", "level", "type", "solution"})


class GenerateManager(object):
    def __init__(
            self,
            main_llm: BaseChatModel,
    ):
        """
        シードとして与えられたproblemとsolutionを元に、そのミューテーション（変種）である新しいproblemとsolutionを生成する処理を取り纏める

        Parameters
        ----------
        main_llm
            ミューテーションの生成時に用いるLLMモデル
        """
        self._model = main_llm

    def __call__(
            self,
            cfg: MutationGenerateConfig
    ) -> None:
        self.file_handling(cfg)

    def file_handling(
            self,
            cfg: MutationGenerateConfig
    ):
        input_base_dir = cfg.input_dir
        output_base_dir = cfg.output_dir
        seed_data_keys = cfg.seed_data_keys

        fp_input_base = Path(input_base_dir)
        fp_output_base = Path(output_base_dir)

        # 指定したディレクトリ配下のファイルのPathオブジェクトを取得するツール
        fp_picker = SuffixFilteredPathBuilder(['.json', ])
        # jsonファイルを読み書きするツール
        jh = JsonHandler()

        # 指定したモデルでミューテーションを生成するインスタンス
        mutation_gen = MutationGenerator(self._model)

        # 子ディレクトリを持たないディレクトリのみを取得
        input_dir_fp_list = [
            d for d in fp_input_base.rglob('*')
            if d.is_dir() and not any(item.is_dir() for item in d.iterdir())
        ]

        # ディレクトリ毎に処理していく
        for fp_d in tqdm(input_dir_fp_list, desc='input dir'):  # List
            relative_path = fp_d.relative_to(fp_input_base)
            fp_out_dir = fp_output_base / relative_path
            ''' OutputPathCreator
            インプットファイル名に基づいで、OUTPUT用のファイル名を作成するツール
            同じファイル名が出力先に存在する場合、ファイル名に'-dup*'を追加して重複を避ける 
            '''
            op_gen = OutputPathCreator(output_dir=str(fp_out_dir), out_suffix='.json', avoid_dup=True)

            # ディレクトリに格納されてされているjsonファイルを取り出して処理していく
            for fp in tqdm(fp_picker(str(fp_d)), desc='files'):  # yield
                # 処理後のファイルを格納する予定の出力先のファイルパスを生成しておく
                fp_output_file = op_gen(fp)
                if Path(fp_output_file).stem != Path(fp).stem:
                    ''' 出力先に同じファイル名がある場合、もととのファイル名と異なるファイル名が生成される
                    → 同一ファイル名が存在するので、すでにミュータントを生成済みと判断
                    '''
                    logger.info(f"Ignore file '{Path(fp).name}' as it already exists.")
                    continue

                seed_data = jh.read(fp)

                # seedに想定したキーが含まれているか確認
                missing_keys = seed_data_keys - set(seed_data.keys())

                if missing_keys:
                    error_msg = (
                        "Key required for dictionary data is missing.\n"
                        f"Missing keys: {missing_keys}\n"
                        f"file: {Path(fp).name}\n"
                    )
                    logger.error(error_msg)
                    raise ValueError(error_msg)

                seed_problem = seed_data['problem']
                seed_solution = seed_data['solution']
                seed_level = seed_data['level']
                seed_type = seed_data['type']

                inst = {
                    "problem": seed_problem,
                    "solution": seed_solution,
                }

                mutations = mutation_gen(inst)
                mutation_problem = mutations['problem']
                mutation_solution = mutations['solution']

                mutation_dict = dict(
                    problem=mutation_problem,
                    level=seed_level,
                    type=seed_type,
                    solution=mutation_solution,
                )

                jh.write(mutation_dict, fp_output_file)


if __name__ == "__main__":
    """
    python -m manager.mutation_gen_manager
    """


    def fix_seed(seed):
        random.seed(seed)
        np.random.seed(seed)


    SEED = 46
    fix_seed(SEED)

    from logging import DEBUG, INFO, WARN, ERROR, basicConfig

    basicConfig(level=WARN)

    # from model.groq_llm import GroqChatBase
    # llm_main = GroqChatBase(
    #     model_name="llama-3.3-70b-versatile",
    #     # max_tokens=2048,
    #     requests_per_second=0.01,
    #     temperature=0.6,
    # )

    from langchain_experimental.llms.ollama_functions import OllamaFunctions

    # 構造化出力用のllm
    llm = OllamaFunctions(model="qwen3:4b", format="json", temperature=0.5)

    test_cfg = dict(
        input_dir='./data/seed',
        output_dir='./data/mutation_gen',
        seed_data_keys={"problem", "level", "type", "solution"}
    )

    gm = GenerateManager(
        main_llm=llm,
    )

    gm(MutationGenerateConfig(**cast(Dict, test_cfg)))
