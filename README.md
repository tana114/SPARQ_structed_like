# SPARQ_structured_like 

# 概要
“SPARQ: Synthetic Problem Generation for Reasoning via Quality-Diversity Algorithms” の実装内容の部分的な動作検証。

### 検証時の実行環境

`WSL2(Ubuntu)`において`uv`で構築した`python(3.12)`環境で実行しました。環境構築時に実行したコマンドは下記です(詳細は`pyproject.toml`をご確認ください)。

```bash
uv add langchain
uv add langchain-experimental
uv add langchain-groq
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

原著のやりかたとは異なりますが、ここでは別の試みとして、3つのFew-shotに対して、追加で2つのミューテーションを生成するということをしています。  

`SYSTEM_PROMPT_FORMAT`を書き換えることで大分挙動が変わるとおもいますが、まだ精査できておりません。  

シードデータ
```python
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
```


生成されたミューテーションの例１（model="qwen3:4b”）
```text
[outputs]
4
problem:
Find the value of $x$ in the equation $\frac{2x + 3}{x - 1} = 5$
solution:
Multiply both sides by $x - 1$ to eliminate the denominator: $2x + 3 = 5(x - 1)$. Distribute the 5 on the right: $2x + 3 = 5x - 5$. Subtract $2x$ from both sides: $3 = 3x - 5$. Add 5 to both sides: $8 = 3x$. Divide by 3: $x = \frac{8}{3}$. The solution is $\boxed{\frac{8}{3}}$.
------------------------
5
problem:
Solve for $x$ in the equation $\sqrt{2x + 1} = x - 1$.
solution:
Square both sides to eliminate the square root: $2x + 1 = (x - 1)^2$. Expand the right-hand side: $2x + 1 = x^2 - 2x + 1$. Subtract $2x + 1$ from both sides: $0 = x^2 - 4x$. Factor: $x(x - 4) = 0$. So, $x = 0$ or $x = 4$. Check for extraneous solutions by substituting back into the original equation. For $x = 0$: $\sqrt{1} = -1$, which is false. For $x = 4$: $\sqrt{9} = 3$, which is true. Therefore, the solution is $\boxed{4}$.
------------------------
```

生成されたミューテーションの例１（model="gemma3:12b”）
```text
[outputs]
4
problem:
What is the sum of the first 50 positive even integers?
solution:
The first 50 positive even integers are 2, 4, 6, ..., 100. This is an arithmetic sequence with first term $a_1 = 2$, common difference $d = 2$, and number of terms $n = 50$. The sum of an arithmetic series is given by $S_n = 
                                  rac{n}{2}(a_1 + a_n)$. In this case, $a_n = a_{50} = 2(50) = 100$. Thus, the sum is $S_{50} = 
                                                                                                                                rac{50}{2}(2 + 100) = 25(102) = 2550$. Therefore, the sum is $\boxed{2550}$.
------------------------
5
problem:
If $x + y = 5$ and $xy = 2$, what is the value of $x^2 + y^2$?
solution:
We can use the identity $(x+y)^2 = x^2 + 2xy + y^2$. We are given $x+y = 5$ and $xy = 2$. Substituting these values into the identity, we have $5^2 = x^2 + 2(2) + y^2$, which simplifies to $25 = x^2 + 4 + y^2$. Subtracting 4 from both sides gives $x^2 + y^2 = 21$. Therefore, the value of $x^2 + y^2$ is $\boxed{21}$.
------------------------
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

Groq APIを用いる場合は`.env`ファイルを作成しkeyを設定してください:

```
GROQ_API_KEY=your_api_key_here
```
その上で、`client/concrete/sparq_mutation_gen.py`の下記において、以下のように指定します。

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
