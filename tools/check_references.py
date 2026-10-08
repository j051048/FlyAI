"""Check configuration and documentation reference integrity.

Validates that:
1. File paths mentioned in .github/workflows/*.yml actually exist.
2. Local Markdown links in every tracked/new repository Markdown file exist.
   This includes sidecar/, phase0/ and provenance documents, while excluding
   ignored private workspaces. Code fences and inline-code examples are skipped.

Exit code 0 on success, 1 if any broken references are found.
"""
import os
import re
import sys
import subprocess
from pathlib import Path
from urllib.parse import unquote

REPO_ROOT = Path(__file__).resolve().parent.parent

# Allowed placeholder patterns in docs and workflows
IGNORE_PATH_PATTERNS = [
    r"<[^>]+>",             # placeholders like <run_id>, <id>.json
    r"\${{.*}}",            # GitHub Actions expressions
    r"\*",                  # wildcards
    r"https?://",           # external URLs
    r"^#",                  # internal markdown anchors
    r"mailto:",             # mailto links
]

# Explicit exempt paths (e.g. theoretical artifacts, planned paths)
EXEMPT_PATHS = {
    "docs/receipts/<id>.json",
    "docs/receipts/<run_id>.json",
    "receipts/<run_id>.json",
}


def is_ignored(path_str: str) -> bool:
    for pat in IGNORE_PATH_PATTERNS:
        if re.search(pat, path_str):
            return True
    return path_str in EXEMPT_PATHS


def check_workflows(errors: list):
    """Check internal path references inside workflow files."""
    wf_dir = REPO_ROOT / ".github" / "workflows"
    if not wf_dir.exists():
        return

    path_regex = re.compile(r'(?:[\s\'"]|^)((?:docs|phase0|shard|sidecar|tests|tools)/[A-Za-z0-9_\-\./]+)')

    for yml in wf_dir.glob("*.y*ml"):
        lines = yml.read_text(encoding="utf-8", errors="ignore").splitlines()
        for idx, line in enumerate(lines, 1):
            for match in path_regex.finditer(line):
                candidate = match.group(1).rstrip(".,;:'\"")
                if is_ignored(candidate):
                    continue
                # strip section specifiers like §5 or #anchor
                clean_path = re.split(r"[#§]", candidate)[0].strip()
                target = REPO_ROOT / clean_path
                if not target.exists():
                    errors.append(f"{yml.relative_to(REPO_ROOT)}:{idx}: broken reference -> {candidate}")


def markdown_files():
    """Use Git's scope when available; source archives have a bounded fallback."""
    try:
        result = subprocess.run(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard", "--", "*.md"],
                                cwd=REPO_ROOT, capture_output=True, check=True)
        return sorted({REPO_ROOT / p for p in result.stdout.decode("utf-8").split("\0") if p and (REPO_ROOT / p).is_file()})
    except (OSError, subprocess.CalledProcessError):
        excluded = {".git", ".venv", "venv", "models", "weights", "scratchpad", ".claude", "__pycache__"}
        return sorted(p for p in REPO_ROOT.rglob("*.md") if not excluded.intersection(p.relative_to(REPO_ROOT).parts))


def check_markdown_links(errors: list):
    """Resolve links relative to their document, just as repository viewers do."""
    md_files = markdown_files()
    link_regex = re.compile(r'\[([^\]]*)\]\((<[^>]+>|[^)]+)\)')

    for md in md_files:
        content = md.read_text(encoding="utf-8", errors="ignore")
        lines = content.splitlines()
        fence = None
        for idx, line in enumerate(lines, 1):
            marker = re.match(r"^\s*(`{3,}|~{3,})", line)
            if marker:
                value = marker.group(1)
                if fence is None:
                    fence = value
                elif value[0] == fence[0] and len(value) >= len(fence):
                    fence = None
                continue
            if fence is not None:
                continue
            # Strip inline code blocks (`...`) so code like `array[i](val)` is not misparsed as a markdown link
            scrubbed_line = re.sub(r'`[^`]*`', '', line)
            for match in link_regex.finditer(scrubbed_line):
                dest = match.group(2).strip()
                # remove optional title inside () e.g. "path 'title'"
                dest = dest[1:-1] if dest.startswith("<") and dest.endswith(">") else dest.split()[0]
                if is_ignored(dest):
                    continue
                # strip anchor
                dest_path = unquote(dest.split("#")[0])
                if not dest_path:
                    continue
                if is_ignored(dest_path):
                    continue

                if dest_path.startswith("/"):
                    resolved = REPO_ROOT / dest_path.lstrip("/")
                else:
                    resolved = (md.parent / dest_path).resolve()

                if not resolved.exists():
                    errors.append(f"{md.relative_to(REPO_ROOT)}:{idx}: broken link -> {dest}")


def main():
    errors = []
    check_workflows(errors)
    check_markdown_links(errors)

    if errors:
        print(f"FAILED: Found {len(errors)} broken reference(s):")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)
    else:
        print("OK: All workflow and documentation references are valid.")
        sys.exit(0)


if __name__ == "__main__":
    main()
