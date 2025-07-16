from typing import Generator, Dict, Optional, Literal, TypedDict, List, Any, TypeVar, Generic

from pydantic import BaseModel, Field
from langchain_core.language_models import BaseChatModel

from client.chain_base import ChainDirector, ConcreteChainBase


class GradeMathSkill(BaseModel):
    """Mathematical skill required to solve a given mathematical problem."""

    binary_score: str = Field(
        description="Mathematical skills required to solve a given problem."
    )


compare_system_prompt = "As a mathematical assessor, your job is to select the most important mathematical skill for solving a given #Problem# from the selection of #Math Skills#, based on the nature of the #Problem# and its #Solution#."

compare_human_prompt = (
    "Here is a pair of #Problem# and #Solution#. What mathematical skill do you consider most important in solving this mathematical problem?\n"
    "Select the skill you consider most important for obtaining the #Solution# to the #Problem# from the list of skills shown for Candidate #Math Skills#.\n\n"
    "#Problem#:\n{problem}\n\n"
    "#Solution#:\n{solution}\n\n"
    "#Math Skills#:\n{math_skills}"
)

T = TypeVar('T')  # 型パラメータTを暫定的に定義しておく


class EquivalenceComparisonGrader(ConcreteChainBase):
    def __init__(
            self,
            chat_model: BaseChatModel,
            allowed_values: List[str]  # 動的にアノテーションを変えるための引数
    ):
        super().__init__()
        self._llm = chat_model
        self._allowed_values = allowed_values
        self.ResponseType = Literal[tuple(allowed_values)]  # Literalを動的に生成

    def _create_chain_director(
            self,
            director_config: Optional[Dict],
    ) -> ChainDirector:
        return ChainDirector(
            chat_model=self._llm,
            system_prompt=compare_system_prompt,
            human_prompt=compare_human_prompt,  # {problem}, {solution}, {math_skills}
            struct_type=GradeMathSkill,
        )

    def _invoke_handling(
            self,
            input: Dict[Literal["problem", "solution", "math_skills"], str],
            **kwargs
    ) -> str:
        res = self._chain_director.invoke(
            input,  # # {problem}, {solution}, {math_skills}
            **kwargs,
        )
        # return res.model_dump()
        return res.binary_score

        # if res.binary_score in self._allowed_values:
        #     return res.binary_score
        # else:
        #     raise ValueError(f"Invalid binary_score: {res.binary_score}. Allowed values are: {self._allowed_values}")


if __name__ == "__main__":
    """
    python -m client.concrete.math_skill_grader
    """

    from model.groq_llm import GroqChatBase

    llm = GroqChatBase(
        model_name="llama-3.3-70b-versatile",
        # requests_per_second=0.32,
        temperature=0.5
    )

    skill_set = {
        "Topic": "Algebra",
        "skills": [
            "combinatorial_operations_and_basic_arithmetic",
            "function_skills",
            "calculation_and_conversion_skills",
            "solving_equations",
            "inequality_skills",
            "graph_and_geometry_skills",
            "number_theory_skills",
            "factoring_skills",
            "complex_number_skills",
            "sequence_and_series_skills",
            "quadratic_equation_skills",
            "geometric_sequence_skills",
            "polynomial_skills",
            "ratio_and_proportion_skills",
            "logarithmic_and_exponential_skills",
            "algebraic_manipulation_skills",
            "distance_and_midpoint_skills",
            "arithmetic_skills",
            "exponent_and_root_skills",
            "algebraic_expression_skills",
            "function_composition_skills"
        ]
    }

    skills = skill_set['skills']

    problem = "How many vertical asymptotes does the graph of $y=\\frac{2}{x^2+x-6}$ have?"
    solution = "The denominator of the rational function factors into $x^2+x-6=(x-2)(x+3)$. Since the numerator is always nonzero, there is a vertical asymptote whenever the denominator is $0$, which occurs for $x = 2$ and $x = -3$.  Therefore, the graph has $\\boxed{2}$ vertical asymptotes."

    # problem = "You have two circles, one with radius $r$ and the other with radius $R$. You wish for the difference in the areas of these two circles to be less than or equal to 5$\\pi$. If $r+R=10$, what is the maximum difference in the lengths of the radii?"
    # solution = "We want $\\pi R^{2}-\\pi r^{2}\\leq 5\\pi$. Dividing by $\\pi$, we have $R^{2}-r^{2}\\leq 5$. Factor the left-hand side to get $(R+r)(R-r)\\leq 5$. Substituting 10 for $R+r$ gives $10(R-r)\\leq 5 \\implies R-r \\leq 1/2$. So the maximum difference in the lengths of the radii is $\\boxed{\\frac{1}{2}}$."

    # problem = "Find the distance between the vertex of the graph of the equation $f(x) = x^2 - 8x + 15$ and the point $(0, 2)$."
    # solution = "Completing the square, we get $f(x) = (x-4)^2 - 1$. The vertex of the graph of this equation is thus $(4, -1)$. Using the Pythagorean Theorem, it follows that the distance between $(0, 2)$ and $(4, -1)$ is $\\boxed{5}$."

    # print(skills)

    inst = {
        "problem": problem,
        "solution": solution,
        "math_skills": skills
    }

    grader = EquivalenceComparisonGrader(llm, allowed_values=skills)

    result = grader(inst)

    print('problem: ', problem)
    print('solution: ', solution)
    print('required skill: ', result)

''' model("llama-3.3-70b-versatile")
problem:  How many vertical asymptotes does the graph of $y=\frac{2}{x^2+x-6}$ have?
solution:  The denominator of the rational function factors into $x^2+x-6=(x-2)(x+3)$. Since the numerator is always nonzero, there is a vertical asymptote whenever the denominator is $0$, which occurs for $x = 2$ and $x = -3$.  Therefore, the graph has $\boxed{2}$ vertical asymptotes.
required skill:  factoring_skills
'''

''' model("llama-3.3-70b-versatile")
problem:  You have two circles, one with radius $r$ and the other with radius $R$. You wish for the difference in the areas of these two circles to be less than or equal to 5$\pi$. If $r+R=10$, what is the maximum difference in the lengths of the radii?
solution:  We want $\pi R^{2}-\pi r^{2}\leq 5\pi$. Dividing by $\pi$, we have $R^{2}-r^{2}\leq 5$. Factor the left-hand side to get $(R+r)(R-r)\leq 5$. Substituting 10 for $R+r$ gives $10(R-r)\leq 5 \implies R-r \leq 1/2$. So the maximum difference in the lengths of the radii is $\boxed{\frac{1}{2}}$.
required skill:  inequality_skills
'''

''' model("llama-3.3-70b-versatile")
problem:  Find the distance between the vertex of the graph of the equation $f(x) = x^2 - 8x + 15$ and the point $(0, 2)$.
solution:  Completing the square, we get $f(x) = (x-4)^2 - 1$. The vertex of the graph of this equation is thus $(4, -1)$. Using the Pythagorean Theorem, it follows that the distance between $(0, 2)$ and $(4, -1)$ is $\boxed{5}$.
required skill:  graph_and_geometry_skills
'''
