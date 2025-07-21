# SPARQ_structured_like 

# 概要
“SPARQ: Synthetic Problem Generation for Reasoning via Quality-Diversity Algorithms” の実装内容の部分的な動作検証。

### 検証時の実行環境

`WSL2(Ubuntu)`において`uv`で構築した`python(3.12)`環境で実行しました。環境構築時に実行したコマンドは下記です(詳細は`pyproject.toml`をご確認ください)。

```bash
uv add langchain
# uv add langchain-experimental
uv add langchain-ollama
uv add langchain-groq
uv add langchain-openai
uv add tqdm
```


# 01_ミューテーションの再現実験

「Phase1-2 : ミューテーションの生成（Mutation）」における下記ミューテーションの生成を予備検討的に実施  

- ワーキングセット $`W^{(t)}`$  からサンプリングされた各親ペア $`(Q^{(t)}_i, A^{(t)}i)`$ を入力として、問題生成モデル $`G_{\theta1}`$  が新しいミューテーション $`(Q'^{(t)}_i, A'^{(t)}_i)`$ を生成する


`client/concrete/sparq_mutation_gen.py`を実行するには、次のコマンドを使用します。

```bash
python -m client.concrete.sparq_mutation_gen
```

スクリプトが実行され、提供されたシードデータに基づいてミューテーション$`(Q'_i, A'_i)`$が生成されます。

#### 例

以下は、ローカル環境に構築した`qwen3:4b`モデルを使用して`client/concrete/sparq_mutation_gen.py`で生成されたミューテーションの例です。  

[//]: # (原著のやりかたとは異なりますが、ここでは別の試みとして、3つのFew-shotに対して、追加で2つのミューテーションを生成するということをしています。  )

`SYSTEM_PROMPT_FORMAT`を書き換えることで大分挙動が変わるとおもいますが、まだ精査できておりません。  
シードとして与えた`problem`に非常に似通ったミューテーションを生成する傾向があるみたいですので、
プロンプトを工夫して言い回しを変えた問題を生成したり、少し生成する問題の難易度を上げたりするような工夫を行ったほうが良いかもしれません。 

シードデータ
```text
problem: 
How many vertical asymptotes does the graph of $y=\frac{2}{x^2+x-6}$ have?
solution:
The denominator of the rational function factors into $x^2+x-6=(x-2)(x+3)$. Since the numerator is always nonzero, there is a vertical asymptote whenever the denominator is $0$, which occurs for $x = 2$ and $x = -3$.  Therefore, the graph has $\boxed{2}$ vertical asymptotes.
```


生成されたミューテーションの例１（model="qwen3:4b”）
```text
problem:
How many vertical asymptotes does the graph of $y=\frac{3x^2 - 2x + 1}{x^2 - 4x + 4}$ have?
solution:
The denominator of the rational function factors into $x^2 - 4x + 4 = (x - 2)^2$. Since the numerator is always nonzero, there is a vertical asymptote whenever the denominator is $0$, which occurs for $x = 2$. However, since the denominator has a repeated root, the graph has a vertical asymptote at $x = 2$. Therefore, the graph has $\boxed{1}$ vertical asymptote.
```
### バッチ処理
指定したディレクトリ配下の`MATH [Hendrycks et al., 2021]`形式のファイルをまとめて処理できるようにしておきました。
```bash
    python mutation_gen_main.py \
	--input_dir './data/seed' \
	--output_dir './data/mutation_gen'
```

# その他

使用するLLMモデルとして、ローカルに構築した`ollama`を用いる方法と`Groq API`を用いる方法を選択できます。

### `ollama`を用いる場合
予めローカルで`ollama run qwen3:4b`を実行できる環境を作っておいてください（`qwen3:4b`使用時の例）。

その上で、`client/concrete/sparq_mutation_gen.py`の下記において、以下のように指定します。

```python
...

if __name__ == "__main__":

    from langchain_experimental.llms.ollama_functions import OllamaFunctions

    # 構造化出力用のllm
    llm = OllamaFunctions(model="qwen3:4b", format="json", temperature=0.5)
    
    ...
```

### `Groq API`を用いる場合
`Groq API`を取得すれば、GPU環境なしでも無料である程度実行できます。

Groq APIを用いる場合は`.env`ファイルを(`README.md`と同じ階層に)作成し、以下のようにkeyを設定してください:  
```  
GROQ_API_KEY=your_api_key_here
```
その上で、`client/concrete/sparq_mutation_gen.py`において、以下のように指定します。

```python
...

if __name__ == "__main__":

    from model.groq_llm import GroqChatBase

    # 構造化出力用のllm
    llm = GroqChatBase(
        model_name="llama-3.3-70b-versatile",
        temperature=0.5
    )
    
    ...
```

### `OpenRouter API`を用いる場合
`OpenRouter API`を取得すれば、GPU環境なしでも`deepseek-r1`などのAPIを無料である程度実行できます。
OpenRouter APIを用いる場合は`.env`ファイルを(`README.md`と同じ階層に)作成し、以下のようにkeyを設定してください:
```
OPENROUTER_API_KEY=your_api_key_here
```
その上で、`open_router_simple_usage.py`において、以下のように指定して実行します。

```python
def main(human_message:str):
    ...
    llm = OpenRouter(
        model_name="deepseek/deepseek-r1:free",
    ) 
    ...

if __name__ == "__main__":
    """
    python open_router_simple_usage.py 
    """
    prompt = '''
    Please create a math problem that is as difficult as HLE. One that even you cannot solve. However, it must be a valid math problem, including one solution and the conditions necessary to obtain that solution.
    '''
    main(prompt)

```

出力例
```text
**Problem Statement:**  
Let \( a, b, c \) be positive real numbers such that \( abc = 1 \). Prove that:  
\[
\frac{a^3 + 1}{b^3 + 1} + \frac{b^3 + 1}{c^3 + 1} + \frac{c^3 + 1}{a^3 + 1} \geq 3
\]  
Determine the conditions under which equality holds.

---

**Conditions for Solution:**  
1. The variables \( a, b, c \) must satisfy \( a, b, c > 0 \) and \( abc = 1 \).  
2. The proof requires advanced inequality techniques, potentially involving substitutions inspired by the constraint \( abc = 1 \) (e.g., setting \( a = \frac{x}{y}, b = \frac{y}{z}, c = \frac{z}{x} \)).  
3. Symmetry and cyclic reasoning are critical. Equality holds if and only if \( a = b = c = 1 \).

**Solution (Summary):**  
- By substituting \( a = b = c = 1 \), the sum equals \( 3 \), satisfying equality.  
- For the inequality, apply the AM-GM inequality and analyze the convexity of the function \( f(x) = \frac{x^3 + 1}{y^3 + 1} \), or use homogenization via \( abc = 1 \) to reduce variables. A full proof leverages transformations and may involve showing that deviations from \( a = b = c \) increase the sum.  

*(Note: A rigorous solution requires intricate steps that are non-trivial even for advanced mathematicians.)*  

**Equality Condition:**  
The equality \( \frac{a^3 + 1}{b^3 + 1} + \frac{b^3 + 1}{c^3 + 1} + \frac{c^3 + 1}{a^3 + 1} = 3 \) holds **only** when \( a = b = c = 1 \).
```


# 99_補足検討

## 99_1_MATHの問題を要求スキルに基づいてさらに細かく分類する

'AI-Assisted Generation of Difficult Math Questions'のTable10に、MATHデータ・セットを要求スキルによってさらに細かく分類した表が示されている。
このスキル表に基づいて各問題を分類した場合にどのスキルが最も要求されるか各問題ごとに判断して結果を出力させてみた。

```bash
python -m client.concrete.math_skill_grader
```

Groq("llama-3.3-70b-versatile")で判定させた結果

```text
problem:  How many vertical asymptotes does the graph of $y=\frac{2}{x^2+x-6}$ have?
solution:  The denominator of the rational function factors into $x^2+x-6=(x-2)(x+3)$. Since the numerator is always nonzero, there is a vertical asymptote whenever the denominator is $0$, which occurs for $x = 2$ and $x = -3$.  Therefore, the graph has $\boxed{2}$ vertical asymptotes.
required skill:  factoring_skills
```
```text
problem:  You have two circles, one with radius $r$ and the other with radius $R$. You wish for the difference in the areas of these two circles to be less than or equal to 5$\pi$. If $r+R=10$, what is the maximum difference in the lengths of the radii?
solution:  We want $\pi R^{2}-\pi r^{2}\leq 5\pi$. Dividing by $\pi$, we have $R^{2}-r^{2}\leq 5$. Factor the left-hand side to get $(R+r)(R-r)\leq 5$. Substituting 10 for $R+r$ gives $10(R-r)\leq 5 \implies R-r \leq 1/2$. So the maximum difference in the lengths of the radii is $\boxed{\frac{1}{2}}$.
required skill:  inequality_skills
```
```text
problem:  Find the distance between the vertex of the graph of the equation $f(x) = x^2 - 8x + 15$ and the point $(0, 2)$.
solution:  Completing the square, we get $f(x) = (x-4)^2 - 1$. The vertex of the graph of this equation is thus $(4, -1)$. Using the Pythagorean Theorem, it follows that the distance between $(0, 2)$ and $(4, -1)$ is $\boxed{5}$.
required skill:  graph_and_geometry_skills
```

## 99_2_OpenRouterを使えるようにする

無料のAPI経由で`deepseek-r1`を利用したかったので[OpenRouter](https://openrouter.ai/)のAPIを使えるようにしました。
APIの取得に関してはこちらのHP[【LLMは無料で使え！】OpenRouterのススメ【CLINEにも！】](https://zenn.dev/asap/articles/5cda4576fbe7cb)を参考にさせていただきました。






