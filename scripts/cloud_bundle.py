"""Check internal static imports, including deferred branches, before paid deployment."""

import ast
from pathlib import PurePosixPath


def validate_bundle(files):
    for name, source in files.items():
        if not name.endswith(".py"):
            continue
        for node in ast.walk(ast.parse(source, filename=name)):
            if not isinstance(node, ast.ImportFrom) or not node.module:
                continue
            if node.level:
                parent = list(PurePosixPath(name).parent.parts)
                if node.level > len(parent):
                    raise ValueError("Relative import escapes bundle package")
                path = (
                    "/".join(parent[: len(parent) - node.level + 1] + node.module.split("."))
                    + ".py"
                )
            elif node.module.startswith("houndoracle."):
                path = node.module.replace(".", "/") + ".py"
            else:
                continue
            if path not in files:
                raise ValueError(f"Missing bundled dependency {path} imported by {name}")
