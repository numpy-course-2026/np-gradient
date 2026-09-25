"""Проверка правила курса: никаких Python-циклов по элементам массива.

    python tests/loops.py файл.py функция [Класс.метод:N ...]

В перечисленных функциях запрещены while, np.vectorize, np.frompyfunc,
np.apply_along_axis и map. Циклы for и генераторы (списков, словарей, множеств,
генераторные выражения) разрешены, только если явно указано, сколько их можно:
«функция:1» — для цикла по итерациям, эпохам, батчам или слоям, а не по элементам.
Метод класса называют либо коротко («forward»), либо через класс («MLP.forward»):
короткое имя действует на все одноимённые методы файла.
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


def violations(func, allowed):
    found, loops = [], 0
    for node in ast.walk(func):
        if isinstance(node, (ast.For, ast.AsyncFor, *COMPREHENSIONS)):
            loops += 1
            if loops > allowed:
                what = "цикл for" if isinstance(node, (ast.For, ast.AsyncFor)) else "генератор — это тоже цикл"
                found.append((node.lineno, what))
        elif isinstance(node, ast.While):
            found.append((node.lineno, "цикл while"))
        elif isinstance(node, ast.Call) and called_name(node) in SLOW_CALLS:
            found.append((node.lineno, f"{called_name(node)} — цикл на Python под капотом"))
    return found


def collect(tree, prefix=""):
    """Функции файла как пары (полное имя, узел): «f», «Класс.метод»."""
    out = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.append((prefix + node.name, node))
            out += collect(node, prefix + node.name + ".")
        elif isinstance(node, ast.ClassDef):
            out += collect(node, prefix + node.name + ".")
    return out


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    path = Path(sys.argv[1])
    tree = ast.parse(path.read_text(encoding="utf-8"))
    funcs = collect(tree)
    bad = False
    for spec in sys.argv[2:]:
        name, _, n = spec.partition(":")
        found = [(full, f) for full, f in funcs if full == name or full.endswith("." + name)]
        if not found:
            print(f"{path}: нет функции {name}")
            bad = True
            continue
        for full, func in found:
            for line, what in violations(func, int(n or 0)):
                print(f"{path}:{line}: {full}: {what}")
                bad = True
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
