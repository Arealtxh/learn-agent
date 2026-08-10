"""
hello_agents/tools/builtin/calculator.py

计算工具 — 安全地执行数学计算
使用 ast.literal_eval 保证安全（和第四章 planner.py 同样的原理）
"""

import ast
import operator


# 只允许这些数学操作 — 白名单模式，防止任意代码执行
ALLOWED_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(node):
    """安全地计算表达式树"""
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    elif isinstance(node, ast.Constant):
        return node.value
    elif isinstance(node, ast.BinOp):
        op = ALLOWED_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"不支持的操作: {type(node.op).__name__}")
        return op(_safe_eval(node.left), _safe_eval(node.right))
    elif isinstance(node, ast.UnaryOp):
        op = ALLOWED_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"不支持的操作: {type(node.op).__name__}")
        return op(_safe_eval(node.operand))
    else:
        raise ValueError(f"不支持的表达式: {type(node).__name__}")


def calculator_tool(expression: str) -> str:
    """安全计算数学表达式

    示例:
        calculator_tool("2 + 3 * 4")   → "14"
        calculator_tool("2 ** 10")      → "1024"
    """
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        result = _safe_eval(tree)
        return str(result)
    except Exception as e:
        return f"计算错误: {e}"
