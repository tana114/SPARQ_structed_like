import json
from typing import Generator, Dict, Optional, Literal, TypedDict, List, Any
from typing import cast

from pydantic import BaseModel, Field
from langchain_core.language_models import BaseChatModel

from client.chain_base import ChainDirector, ConcreteChainBase

"""



"""

# NO_SIGN = "<noinput>"

''' SYSTEM_PROMPT_FORMAT
{task_num}は生成するタスクの個数（few-shotの数も含まれる）
'''


SYSTEM_PROMPT_FORMAT = (
    "You are a helpful math problem and solution writer.\n"
    "You will be asked to generate a diverse a set of {tasks_num} structured maths problems and solutions."
    " You are tasked with generating a mutation conditioned on a set of input problems.\n"
    "Here are the requirements:\n"
    "- Provide a step-by-step solution that addresses the given #problem#."
    " Clearly explain the purpose and execution of each operation performed to reach the final answer.\n"
    "- The #solution# must be an appropriate response to the #problem#.\n"
    "- Make sure to include the intended final answer in #solution# enclosed in the latex style.\n"
    "  If there are multiple numerical answers, write them as a comma separated list (n1, n2, ...).\n"
)

''' HUMAN_PROMPT_FORMAT
{{few_shot}}と{{next_no}}はinvoke時に与える
{{few_shot}} : few-shot文字列
{{next_no}} : few-shot以降に生成するタスクの開始no
'''
HUMAN_PROMPT_FORMAT = (
    "List of {tasks_num} structured math problems: \n\n{{few_shot}}\n"
     "Generate a list of the remaining math problems starting from no: {{next_no}} onwards.\n"
    # "Generate the list of remaining math problems after no: {{next_no}} onwards following the above.\n"
)



class ProblemData(BaseModel):
    """The Problem record includes problem, solution, and no."""
    no: int = Field(description="List number of the generated problem.")
    problem: str = Field(description="Math problem.")
    solution: str = Field(description="Solution and final answer enclosed in the latex style.")


class ProblemList(BaseModel):
    """List for contains ProblemData."""
    problems: List[ProblemData] = Field(description="A list containing multiple ProblemData.")


class SelfInstructGenerator(ConcreteChainBase):
    def __init__(
            self,
            chat_model: BaseChatModel,
            num_task_to_generate: int = 10,
            use_gen_num_check: bool = False,
    ):
        """
        :param chat_model:
        :param num_task_to_generate: int
        生成するタスクの数（Few-Shotで例示した個数も含まれる）
        """
        super().__init__()
        self._llm = chat_model
        self._num_gen = num_task_to_generate
        self._use_check = use_gen_num_check

    @staticmethod
    def encode_few_shot_prompt(
            seed_instructions: List[Dict[str, str]]
    ) -> str:
        """
        このクラスでinvokeする際に渡すFew-Shotプロンプト用の文字を生成する

        :param seed_instructions:
        :return:

        seed_instructions =[
            {"instruction": "hoge1", "input": "", "output": "hogefuga1"},
            {"instruction": "hoge2", "input": "fuga2", "output": "hogefuga2"},
            {"instruction": "hoge3", "input": "fuga3, "output": "hogefuga3"},
        ]

        few_shot_prompts ='''
        {
          no: 1,
          instruction: hoge1,
          input: , "<noinput>"
          output: hogefuga1
        }
        {
          ...
          output: hogefuga3
        }
        """
        # Replace '""' of input  with '"<noinput>"' character.
        # seeds = [{k: NO_SIGN if k == 'input' and v == "" else v for k, v in d.items()} for d in seed_instructions]
        # Add key 'no' and give serial number to value.
        seeds = [{"no": i + 1, **d} for i, d in enumerate(seed_instructions)]

        few_shot_prompt = ""
        # convert dict type to string.
        for d in seeds:
            few_shot_prompt += json.dumps(d, indent=2, ensure_ascii=False)
            few_shot_prompt += "\n"

        return few_shot_prompt

    def _create_chain_director(
            self,
            director_config: Optional[Dict],
    ) -> ChainDirector:
        system_prompt = SYSTEM_PROMPT_FORMAT.format(tasks_num=self._num_gen)
        human_prompt = HUMAN_PROMPT_FORMAT.format(tasks_num=self._num_gen)

        return ChainDirector(
            chat_model=self._llm,
            system_prompt=system_prompt,
            human_prompt=human_prompt,  # {few-shot}, {next_no}
            struct_type=ProblemList,
        )

    def _invoke_handling(
            self,
            input: Dict[Literal["few_shot", "next_no"], str],
            **kwargs
    ) -> List[Dict]:

        if self._use_check:
            res = cast(ProblemList, self._inst_num_check(input, **kwargs))
        else:
            chain_d = cast(ChainDirector, self._chain_director)
            res = cast(ProblemList, chain_d.invoke(input, **kwargs, ))

            # TaskList型を辞書型にdumpしたものを返す#
        task_list = [d.model_dump() for d in res.problems]
        # # 'input' の内容がNO_SIGNのままになっている場合は空文字の""に置き換える
        # tg_key = 'input'
        # for d in task_list:
        #     if d[tg_key].lower() == NO_SIGN.lower():
        #         d[tg_key] = ""

        return task_list

    def _inst_num_check(
            self,
            input: Dict[Literal["few_shot", "next_no"], str],
            **kwargs
    ):
        chain_d = cast(ChainDirector, self._chain_director)
        res = cast(ProblemList, chain_d.invoke(input, **kwargs, ))
        if not res:
            return self._inst_num_check(input, **kwargs)
        # 要素の数が想定した個数になっているかチェックするためにListの中身を確認
        tasks_list = res.problems
        tasks_list = [e for e in tasks_list if e.problem]  # 空文字は削除
        counts = self._num_gen - int(input["next_no"])
        return self._inst_num_check(input, **kwargs) if len(tasks_list) != counts + 1 else res


if __name__ == "__main__":
    """
    python -m client.concrete.sparq_mutation_gen
    """

    # from model.groq_llm import GroqChatBase
    #
    # llm = GroqChatBase(
    #     model_name="llama-3.3-70b-versatile",
    #     # requests_per_second=0.32,
    #     temperature=0
    # )


    from langchain_experimental.llms.ollama_functions import OllamaFunctions

    # 構造化出力用のllm
    # llm = OllamaFunctions(model="qwen3:4b", format="json", temperature=0.5)
    llm = OllamaFunctions(model="gemma3:12b", format="json", temperature=0.5)

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

    gen = SelfInstructGenerator(
        chat_model=llm,
        num_task_to_generate=5,
        # use_gen_num_check=True,
    )
    # few-shotの文字列を作成
    few_shot = gen.encode_few_shot_prompt(seeds)
    print(few_shot)
    next_no = len(seeds) + 1

    inst = {
        "few_shot": few_shot,
        "next_no": next_no,
    }

    result = gen(inst)
    # print(result)

    for r in result:
        print(r['no'])
        print(r['problem'])
        print(len(r['solution']))
        print(r['solution'])
        print('------------------------')


'''model="qwen3:4b"
[outputs]
4
What is the value of $\log_2(8) + \log_3(9)$?
221
We know that $8 = 2^3$ and $9 = 3^2$. Using the logarithmic identity $\log_b(a^n) = n\log_b(a)$, we have $\log_2(8) = 3$ and $\log_3(9) = 2$. Adding these together, we get $3 + 2 = 5$. Therefore, the value is $\boxed{5}$.
------------------------
5
A rectangle has a length of 12 units and a width of 8 units. What is the length of the diagonal of the rectangle?
350
The length of the diagonal of a rectangle can be found using the Pythagorean Theorem. The diagonal is the hypotenuse of a right triangle with legs of 12 and 8 units. So, $d = \sqrt{12^2 + 8^2} = \sqrt{144 + 64} = \sqrt{208}$. Simplifying, $\sqrt{208} = \sqrt{16 \times 13} = 4\sqrt{13}$. Therefore, the length of the diagonal is $\boxed{4\sqrt{13}}$.
------------------------
'''


''' model="gemma3:12b"
[outputs]
4
What is the sum of the first 50 positive even integers?
397
The first 50 positive even integers are 2, 4, 6, ..., 100. This is an arithmetic sequence with first term $a_1 = 2$, common difference $d = 2$, and number of terms $n = 50$. The sum of an arithmetic series is given by $S_n = 
                                  rac{n}{2}(a_1 + a_n)$. In this case, $a_n = a_{50} = 2(50) = 100$. Thus, the sum is $S_{50} = 
                                                                                                                                rac{50}{2}(2 + 100) = 25(102) = 2550$. Therefore, the sum is $\boxed{2550}$.
------------------------
5
If $x + y = 5$ and $xy = 2$, what is the value of $x^2 + y^2$?
317
We can use the identity $(x+y)^2 = x^2 + 2xy + y^2$. We are given $x+y = 5$ and $xy = 2$. Substituting these values into the identity, we have $5^2 = x^2 + 2(2) + y^2$, which simplifies to $25 = x^2 + 4 + y^2$. Subtracting 4 from both sides gives $x^2 + y^2 = 21$. Therefore, the value of $x^2 + y^2$ is $\boxed{21}$.
------------------------
'''