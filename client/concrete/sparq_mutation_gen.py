from typing import Generator, Dict, Optional, Literal, TypedDict, List, Any
from typing import cast

from pydantic import BaseModel, Field
from langchain_core.language_models import BaseChatModel

from client.chain_base import ChainDirector, ConcreteChainBase

SYSTEM_PROMPT_FORMAT = (
    "You are a helpful math problem and solution writer.\n"
    "You are tasked with generating a mutation conditioned on a set of input problems."
    " You will be shown the problems below.\n"
    "#Seed Problem#\n"
    "#Seed Solution#\n\n"
    "Given the #Seed Problem# and the #Seed Solution#,"
    " you will be tasked with generating the #Problem# and its solution, the #Solution#.\n"
    "Here are the requirements:\n"
    "- Generate a #Problem# that is as complex as the given #Seed Problem#."
    " However, the #Problem# and the #Seed Problem# must not be identical."
    " The #Problem# must be designed so that it is not just a partial replacement for the figures in the #Seed Problem#,"
    " but a different version.\n"
    "- Provide a step-by-step solution that addresses the #Problem#.\n"
    " Clearly explain the purpose and execution of each operation performed to reach the final answer.\n"
    "- The #Solution# must be an appropriate response to the #Problem#.\n"
    "- Make sure to include the intended final answer in #Solution# enclosed in the latex style."
    " If there are multiple numerical answers, write them as a comma separated list (n1, n2, ...).\n"
)

''' HUMAN_PROMPT_FORMAT
{problem}と{solution}はinvoke時に与える
{problem} : シードとして与えるproblem
{solution} : シードとして与えるsolution
'''
HUMAN_PROMPT_FORMAT = (
    "The seed problem and solution are below.\n"
    "#Seed Problem#: {problem}\n"
    "#Seed Solution#: {solution}\n"
    "Generate a pair of mutations based on a pair of input problems and solutions."
)


class ProblemData(BaseModel):
    """The Problem record includes problem and solution."""
    problem: str = Field(description="Math problem.")
    solution: str = Field(description="Solution and final answer enclosed in the latex style.")


class MutationGenerator(ConcreteChainBase):
    def __init__(
            self,
            chat_model: BaseChatModel,
    ):
        """
        :param chat_model:
        :param num_task_to_generate: int
        生成するタスクの数（Few-Shotで例示した個数も含まれる）
        """
        super().__init__()
        self._llm = chat_model

    def _create_chain_director(
            self,
            director_config: Optional[Dict],
    ) -> ChainDirector:
        system_prompt = SYSTEM_PROMPT_FORMAT
        human_prompt = HUMAN_PROMPT_FORMAT

        return ChainDirector(
            chat_model=self._llm,
            system_prompt=system_prompt,
            human_prompt=human_prompt,  # {problem}, {solution}
            struct_type=ProblemData,
        )

    def _invoke_handling(
            self,
            input: Dict[Literal["problem", "solution"], str],
            **kwargs
    ) -> Dict:
        chain_d = cast(ChainDirector, self._chain_director)
        res = cast(ProblemData, chain_d.invoke(input, **kwargs, ))

        # 辞書型にdumpしたものを返す
        mutations = res.model_dump()

        return mutations


if __name__ == "__main__":
    """
    python -m client.concrete.sparq_mutation_gen
    """

    # from model.groq_llm import GroqChatBase
    #
    # llm = GroqChatBase(
    #     model_name="llama-3.3-70b-versatile",
    #     # requests_per_second=0.32,
    #     temperature=0.5
    # )

    from langchain_experimental.llms.ollama_functions import OllamaFunctions

    # 構造化出力用のllm
    llm = OllamaFunctions(model="qwen3:4b", format="json", temperature=0.5)
    # llm = OllamaFunctions(model="gemma3:12b", format="json", temperature=0.5)

    gen = MutationGenerator(chat_model=llm)

    seeds = [
        {
            "problem": "How many vertical asymptotes does the graph of $y=\\frac{2}{x^2+x-6}$ have?",
            "solution": "The denominator of the rational function factors into $x^2+x-6=(x-2)(x+3)$. Since the numerator is always nonzero, there is a vertical asymptote whenever the denominator is $0$, which occurs for $x = 2$ and $x = -3$.  Therefore, the graph has $\\boxed{2}$ vertical asymptotes.",
        },
        {
            "problem": "You have two circles, one with radius $r$ and the other with radius $R$. You wish for the difference in the areas of these two circles to be less than or equal to 5$\\pi$. If $r+R=10$, what is the maximum difference in the lengths of the radii?",
            "solution": "We want $\\pi R^{2}-\\pi r^{2}\\leq 5\\pi$. Dividing by $\\pi$, we have $R^{2}-r^{2}\\leq 5$. Factor the left-hand side to get $(R+r)(R-r)\\leq 5$. Substituting 10 for $R+r$ gives $10(R-r)\\leq 5 \\implies R-r \\leq 1/2$. So the maximum difference in the lengths of the radii is $\\boxed{\\frac{1}{2}}$.",
        },
        {
            "problem": "Find the distance between the vertex of the graph of the equation $f(x) = x^2 - 8x + 15$ and the point $(0, 2)$.",
            "solution": "Completing the square, we get $f(x) = (x-4)^2 - 1$. The vertex of the graph of this equation is thus $(4, -1)$. Using the Pythagorean Theorem, it follows that the distance between $(0, 2)$ and $(4, -1)$ is $\\boxed{5}$.",
        },
    ]

    i = 0
    seed_problem = seeds[i]["problem"]
    seed_solution = seeds[i]["solution"]

    inst = {
        "problem": seed_problem,
        "solution": seed_solution,
    }

    result = gen(inst)

    print('------------ seed -------------')
    print(seed_problem)
    print(seed_solution)
    print('------------ mutation -------------')
    print(result["problem"])
    print(result["solution"])

'''model="qwen3:4b"
------------ seed -------------
How many vertical asymptotes does the graph of $y=\frac{2}{x^2+x-6}$ have?
The denominator of the rational function factors into $x^2+x-6=(x-2)(x+3)$. Since the numerator is always nonzero, there is a vertical asymptote whenever the denominator is $0$, which occurs for $x = 2$ and $x = -3$.  Therefore, the graph has $\boxed{2}$ vertical asymptotes.
------------ mutation -------------
How many vertical asymptotes does the graph of $y=\frac{3x^2 - 2x + 1}{x^2 - 4x + 4}$ have?
The denominator of the rational function factors into $x^2 - 4x + 4 = (x - 2)^2$. Since the numerator is always nonzero, there is a vertical asymptote whenever the denominator is $0$, which occurs for $x = 2$. However, since the denominator has a repeated root, the graph has a vertical asymptote at $x = 2$. Therefore, the graph has $\boxed{1}$ vertical asymptote.
'''
