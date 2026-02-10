"""Generate the code reference pages and navigation."""

import ast
from pathlib import Path

import mkdocs_gen_files


nav = mkdocs_gen_files.Nav()
root = Path(__file__).parent.parent
src = root / "src"


def extract_module_docstring(filepath) -> str | None:
    """Extract the module-level docstring from a Python file."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read())

        return ast.get_docstring(tree)
    except:
        return None


for path in sorted(src.rglob("*.py")):
    module_path = path.relative_to(src).with_suffix("")
    doc_path = path.relative_to(src).with_suffix(".md")
    full_doc_path = Path("reference", doc_path)

    parts = tuple(module_path.parts)

    if parts[-1] == "__init__":
        parts = parts[:-1]
        doc_path = doc_path.with_name("index.md")
        full_doc_path = full_doc_path.with_name("index.md")
    elif parts[-1] == "__main__":
        continue

    nav[parts] = doc_path.as_posix()
    ident = ".".join(parts)

    # Extract module docstring
    module_docstring = extract_module_docstring(path)

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        # Write module docstring if it exists
        if module_docstring:
            fd.write(f"{module_docstring}\n\n")

        else:
            # Write the mkdocstrings directive for the actual API documentation
            fd.write(f"::: {ident}\n")

    mkdocs_gen_files.set_edit_path(full_doc_path, path.relative_to(root))


with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
