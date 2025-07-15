import argparse
import random

import numpy as np

from manager.mutation_gen_manager import MutationGenerateConfig, GenerateManager


def fix_seeds(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)


INPUT_DIR = './data/seed'
OUTPUT_DIR = './data/mutation_gen'
RANDOM_SEED = 0


def parse_option():
    dc = """
    Configuration information for generating mutation problems from MATH datasets.
    """
    parser = argparse.ArgumentParser(description=dc)

    parser.add_argument('--input_dir', type=str, dest='input_dir', default=INPUT_DIR,
                        help=f""" The input directory stores the *.json files containing the seed problems. e.g. {INPUT_DIR}""")
    parser.add_argument('--output_dir', type=str, dest='output_dir', default=OUTPUT_DIR,
                        help=f""" The output directory. e.g. {OUTPUT_DIR}""")
    return parser.parse_args()


def main():
    fix_seeds(RANDOM_SEED)

    # from model.groq_llm import GroqChatBase
    # llm = GroqChatBase(
    #     model_name="llama-3.3-70b-versatile",
    #     # requests_per_second=0.05,  # 1 request every 10 seconds or so.
    #     requests_per_second=0.07,
    #     # max_tokens=2048,
    #     temperature=0.5,
    # )

    from langchain_experimental.llms.ollama_functions import OllamaFunctions

    # 構造化出力用のllm
    llm = OllamaFunctions(model="qwen3:4b", format="json", temperature=0.5)

    gm = GenerateManager(
        main_llm=llm,
    )

    args = parse_option()
    gen_cfg = MutationGenerateConfig(**vars(args))
    gm(gen_cfg)


if __name__ == "__main__":
    """
    python mutation_gen_main.py \
	--input_dir './data/seed' \
	--output_dir './data/mutation_gen'
    """
    main()
