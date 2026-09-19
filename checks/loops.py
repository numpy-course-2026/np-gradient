"""Проверка правила курса: никаких Python-циклов по элементам массива.

    python tests/loops.py файл.py функция [функция:N ...]

В перечисленных функциях запрещены while, генераторы списков/словарей/множеств,
генераторные выражения, np.vectorize, np.frompyfunc, np.apply_along_axis и map.
Цикл for разрешён, только если явно указано, сколько их можно: «функция:1» —
для цикла по итерациям, эпохам или батчам, а не по элементам.
Код возврата 1 и список нарушений с номерами строк — если правило нарушено.
"""
import ast
import sys
from pathlib import Path

SLOW_CALLS = {"vectorize", "frompyfunc", "apply_along_axis", "map"}
COMPREHENSIONS = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)


def called_name(node):
    f = node.func
    if isinstance(f, ast.Attribute):
        return f.attr
    if isinstance(f, ast.Name):
        return f.id
    return None


def violations(func, allowed_for):
    found, fors = [], 0
    for node in ast.walk(func):
        if isinstance(node, (ast.For, ast.AsyncFor)):
            fors += 1
            if fors > allowed_for:
                found.append((node.lineno, "цикл for"))
        elif isinstance(node, ast.While):
            found.append((node.lineno, "цикл while"))
        elif isinstance(node, COMPREHENSIONS):
            found.append((node.lineno, "генератор/включение — это тоже цикл"))
        elif isinstance(node, ast.Call) and called_name(node) in SLOW_CALLS:
            found.append((node.lineno, f"{called_name(node)} — цикл на Python под капотом"))
    return found


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    path = Path(sys.argv[1])
    tree = ast.parse(path.read_text(encoding="utf-8"))
    funcs = {n.name: n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    bad = False
    for spec in sys.argv[2:]:
        name, _, n = spec.partition(":")
        func = funcs.get(name)
        if func is None:
            print(f"{path}: нет функции {name}")
            bad = True
            continue
        for line, what in violations(func, int(n or 0)):
            print(f"{path}:{line}: {name}: {what}")
            bad = True
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
