import warnings
import os
from typing import Generator, Dict, Optional, Literal, TypedDict, List, Any

from langchain_openai import ChatOpenAI
from langchain_core._api.beta_decorator import LangChainBetaWarning
from langchain_core.rate_limiters import InMemoryRateLimiter

# InMemoryRateLimiter使用時の警告を消す
warnings.filterwarnings("ignore", category=LangChainBetaWarning)

from dotenv import load_dotenv

load_dotenv()  # .envの内容が環境変数として読み込まれる

# deplayment name
valid_model_names = Literal[
    "deepseek/deepseek-chat-v3-0324:free",
    "deepseek/deepseek-r1-0528:free",
    "deepseek/deepseek-r1:free",
    "moonshotai/kimi-k2:free",
    "qwen/qwq-32b:free",
    "qwen/qwen3-32b:free",
    "qwen/qwen3-235b-a22b:free",
    "google/gemma-3-27b-it:free",
]


class OpenRouter(ChatOpenAI):
    def __init__(
            self,
            model_name: Literal[valid_model_names],
            requests_per_second: Optional[float] = None,
            **kwargs,
    ):
        api_key = os.getenv("OPENROUTER_API_KEY")
        endpoint = "https://openrouter.ai/api/v1"
        shared_kwargs = dict(
            openai_api_key=api_key,
            openai_api_base=endpoint,
            model_name=model_name,
            **kwargs,
        )

        if requests_per_second:
            r_limiter = InMemoryRateLimiter(requests_per_second=requests_per_second)
            shared_kwargs["rate_limiter"] = r_limiter

        super().__init__(**shared_kwargs)


if __name__ == "__main__":
    """
    python -m model.open_router_llm
    """

    llm = OpenRouter(
        model_name="deepseek/deepseek-r1:free",
        # requests_per_second=0.32,
    )

    res = llm.invoke("hello, i am tired.")
    print(res)
