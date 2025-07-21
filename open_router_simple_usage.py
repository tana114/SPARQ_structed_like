import random

import numpy as np


RANDOM_SEED = 0

def fix_seeds(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)


def main(human_message:str):
    """
    OpenRouter導入の参考HP

    【LLMは無料で使え！】OpenRouterのススメ【CLINEにも！】
    https://zenn.dev/asap/articles/5cda4576fbe7cb

    """
    fix_seeds(RANDOM_SEED)

    from model.open_router_llm import OpenRouter

    llm = OpenRouter(
        # model_name="deepseek/deepseek-chat-v3-0324:free",
        # model_name="deepseek/deepseek-r1-0528:free",
        model_name="deepseek/deepseek-r1:free",
        # model_name="moonshotai/kimi-k2:free",
        # model_name="qwen/qwq-32b:free",
        # model_name="qwen/qwen3-32b:free",
        # model_name="qwen/qwen3-235b-a22b:free",
        # model_name="google/gemma-3-27b-it:free",
        # requests_per_second=0.32,
    )

    # human_message = 'hello'
    res = llm.invoke(human_message)

    if res:
        # print(type(res))
        # print(res)
        print(res.content)


if __name__ == "__main__":
    """
    python open_router_simple_usage.py 
    """

    prompt = '''
Please create a math problem that is as difficult as HLE. One that even you cannot solve. However, it must be a valid math problem, including one solution and the conditions necessary to obtain that solution.
    '''

    # prompt = 'hello'
    main(prompt)