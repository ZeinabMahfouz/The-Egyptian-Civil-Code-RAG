"""Rubric point 1: `pip install -e .` alone must install everything the
package imports. requirements.txt is installed first in CI and Docker, so a
missing entry in pyproject.toml would only break for someone who follows
the plain packaging route -- this test catches it everywhere."""

import ast
import sys
import tomllib
from pathlib import Path

REPO = Path(__file__).parent.parent
PKG = REPO / "src" / "egyptian_civil_code_rag"

# import name -> distribution name, where they differ
DIST = {
    "yaml": "pyyaml",
    "sentence_transformers": "sentence-transformers",
    "qdrant_client": "qdrant-client",
    "prometheus_client": "prometheus-client",
    "guardrails": "guardrails-ai",
}
# optional: service.py runs in its own BentoML environment; guardrails is a
# try/except import in pii.py
OPTIONAL = {"bentoml": "serve", "guardrails": "guardrails"}


def third_party_imports() -> set[str]:
    names = set()
    for f in PKG.glob("*.py"):
        for node in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                names |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names.add(node.module.split(".")[0])
    stdlib = set(sys.stdlib_module_names)
    return {n for n in names if n not in stdlib and n != "egyptian_civil_code_rag"}


def declared(spec_list) -> set[str]:
    out = set()
    for spec in spec_list:
        name = spec
        for sep in "<>=!~[; ":
            name = name.split(sep)[0]
        out.add(name.strip().lower())
    return out


def test_every_import_is_declared_in_pyproject():
    project = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    core = declared(project["dependencies"])
    extras = {k: declared(v) for k, v in project.get("optional-dependencies", {}).items()}
    missing = []
    for mod in sorted(third_party_imports()):
        dist = DIST.get(mod, mod).lower()
        if mod in OPTIONAL:
            if dist not in extras.get(OPTIONAL[mod], set()):
                missing.append(f"{mod} (extra [{OPTIONAL[mod]}])")
        elif dist not in core:
            missing.append(mod)
    assert not missing, f"imported by the package but not in pyproject.toml: {missing}"
