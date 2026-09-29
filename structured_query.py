from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

import re
import subprocess
import tempfile
import os
import sys
import ast


dataframe_schema = """
The DataFrame `df` contains purchase order information.

Columns:

- retailer_name: Name of the retailer that placed the order.
- product_sku: Unique SKU/product identifier for the ordered product.
- quantity_ordered: Number of units ordered for the product.
- unit_price: Price of one unit of the product.
- order_date: Date when the order was placed.
- status: Current status of the order, such as fulfilled, cancelled, or pending or backordered.

"""

prompt_template = """
You are a Python Pandas expert.
 
Your task is to convert the user's natural-language question into
correct, executable Python Pandas code.
 
The DataFrame is named `df`.
 
1. Use only Pandas and standard Python syntax.
2. Use `df` as the DataFrame.
3. Use only columns provided in the schema.
4. Do not use SQL.
5. Do not invent columns.
6. Return only executable Python code.
7. Do not create sample data.
8. Use Pandas boolean indexing for filtering.
9. Use appropriate Pandas aggregation functions.
10. Use `groupby()` when aggregation is required by category.
11. Use `sort_values()` when sorting is required.
12. Use Pandas datetime operations for date-related questions.
13. If calculating total order value, use:
    `quantity_ordered * unit_price`
14. Keep the generated code simple and readable.
 
### Query Interpretation Rules
 
15. Identify ALL conditions expressed in the user's question before
    generating code. Every condition must be represented in the code.
 
16. If the question refers to a status such as "backorders",
    "backordered orders", "cancelled orders", "fulfilled orders",
    or "pending orders", filter `df['status']` to the corresponding
    status BEFORE performing any aggregation.
 
17. If the question asks "how many", return a count.
 
18. If the question asks for an entity with the "highest", "most",
    "largest", or "greatest" value:
    - calculate the requested metric for each entity,
    - then select the entity with the maximum value.
    - Do not return the complete aggregation.
 
19. If the question asks for an entity with the "lowest", "least",
    "smallest", or "fewest" value:
    - calculate the requested metric for each entity,
    - then select the entity with the minimum value.
    - Do not return the complete aggregation.
 
20. When determining which retailer has the highest or lowest number
    of orders satisfying a condition:
    - apply the condition first,
    - count the matching rows by `retailer_name`,
    - select the retailer corresponding to the maximum or minimum count.
 
21. Distinguish between:
    - calculating a metric for every entity
    - selecting the entity with the maximum/minimum metric.
 
    If the question asks "which", "who", or "what retailer" and also
    contains a ranking term such as highest, lowest, most, or least,
    return the requested entity rather than the complete aggregation.
 
22. Do not stop after filtering or aggregation if the question requires
    an additional operation to answer the question.
 
23. The generated code must directly produce the final answer requested
    by the user.
 
24. Before returning the code, verify that every meaningful part of the
    natural-language question is represented in the generated code.
 
25. If context about this retailer from notes is provided below, use it
    only to understand what the user might mean (e.g. which condition or
    status is relevant) - never invent columns or values that aren't in
    the schema just because the notes mention them.
 
26. If the question names a specific retailer (e.g. "Sunbelt School
    Supply"), the FIRST line of code must filter `df` to only that
    retailer's rows, before any other computation:
    df = df[df['retailer_name'] == '<retailer name>']
    Do not compute anything across all retailers when a specific
    retailer is named in the question.
 
27. When combining more than one condition with `&` or `|`, ALWAYS wrap
    EACH individual condition in its own parentheses:
        df[(df['col1'] == 'x') & (df['col2'] == 'y')]
    Never write: df[df['col1'] == 'x' & df['col2'] == 'y']
    (without the inner parentheses this is invalid and will raise a
    TypeError, because Python evaluates `&` before `==`).
 
### Important
Think about what the user is asking for BEFORE generating the code.
 
Also see the schema description to understand the column types and use it accordingly.
 
The code must produce the requested result, not merely part of the calculation.
 
Example 1
Question: How many orders have status 'cancelled'?
Answer:
len(df[df['status'] == 'cancelled'])
 
Example 2
Question: What is the average unit price for SKU-00042?
Answer:
df.loc[df['product_sku'] == 'SKU-00042', 'unit_price'].mean()

The data you're working with:
{schema}

Question: {question}

Context: {context}

Answer:
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["question", "schema", "context"])

TIMEOUT_SECONDS = 100

SETUP_CODE = """
import pandas as pd

df = pd.read_csv('b2b_wholesale_purchase_orders_1m.csv')

"""

# --- FIX 3: banned-pattern safety check, brought over from project 3's sandbox.py ---
BANNED_PATTERNS = [
    r"\bos\.system\b",
    r"\bsubprocess\b",
    r"\beval\s*\(",
    r"\bexec\s*\(",
    r"__import__",
    r"\bopen\s*\(",
    r"\bshutil\b",
    r"\bsys\.exit\b",
    r"\binput\s*\(",
    r"\brequests\b",
    r"\bsocket\b",
]


def is_code_safe(code):
    for pattern in BANNED_PATTERNS:
        if re.search(pattern, code):
            return False, f"Blocked: matched banned pattern '{pattern}'"
    return True, "ok"


# --- FIX 2: extract code from the model's raw response, stripping markdown fences ---
def extract_code_block(answer_text):
    match = re.search(r"```(?:python)?\s*(.*?)```", answer_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return answer_text.strip()


def run_code_safely(code):
    full_code = SETUP_CODE + "\n" + code

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(full_code)
        temp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, temp_path],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
        )
        return {
            "success": result.returncode == 0,
            "output": result.stdout,
            "error": result.stderr,
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "output": "",
            "error": f"Code took longer than {TIMEOUT_SECONDS} seconds to run (possible infinite loop)",
        }
    finally:
        os.remove(temp_path)


def ensure_output_visible(code):
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code

    if not tree.body:
        return code

    last_node = tree.body[-1]
    if isinstance(last_node, ast.Expr):
        already_print = (
            isinstance(last_node.value, ast.Call)
            and isinstance(last_node.value.func, ast.Name)
            and last_node.value.func.id == "print"
        )
        if not already_print:
            print_call = ast.Expr(value=ast.Call(
                func=ast.Name(id="print", ctx=ast.Load()),
                args=[last_node.value], keywords=[],
            ))
            tree.body[-1] = ast.copy_location(print_call, last_node)
            ast.fix_missing_locations(tree)
            return ast.unparse(tree)

    return code


def answer_question(query, context=None):
    if context:
        context_section = f"Known context about this retailer from notes:\n{context}\n"
    else:
        context_section = ""

    final_prompt = prompt.format(question=query, schema=dataframe_schema, context=context_section)

    llm = OllamaLLM(model="llama3.2")
    answer = llm.invoke(final_prompt)

    # FIX 2: extract the real code before doing anything else with it
    code = extract_code_block(answer)

    # FIX 3: safety check before execution
    safe, reason = is_code_safe(code)
    if not safe:
        return {"answer": answer, "code": code, "execution_result": {
            "success": False, "output": "", "error": reason
        }}

    execution_result = run_code_safely(ensure_output_visible(code)) if code else None

    # FIX 1: "code" key now actually exists in the return dict
    return {"answer": answer, "code": code, "execution_result": execution_result}


if __name__ == "__main__":
    test_questions = [
        "which retailer has the highest backorders?"
    ]

    for query in test_questions:
        print("=" * 60)
        print("Question:", query)
        # no context passed here - context defaults to None for a pure
        # structured-only test run, same as before this feature was added
        result = answer_question(query)

        print("\nRaw model answer:\n", result["answer"])
        print("\nExtracted code:\n", result["code"])
        if result["execution_result"]:
            print("\nExecution result:")
            print("  success:", result["execution_result"]["success"])
            print("  output:", result["execution_result"]["output"])
            print("  error:", result["execution_result"]["error"])