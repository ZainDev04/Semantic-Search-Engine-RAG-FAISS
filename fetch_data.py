"""
Pulls real commit data out of a cloned git repository and saves it as a
flat JSONL dataset for the search project.

The corpus is the commit history of huggingface/datasets. Commit messages
are short, technical English written by working engineers, and every
commit comes with two related texts (a one-line title and a longer body),
which is what the label-free evaluation in evaluate.py relies on.

The clone uses --filter=blob:none --no-checkout, so only commit and tree
metadata is fetched, not file contents; it takes seconds on a large repo.

Cleaning applied to every record (see clean_title / clean_body):
  * the "(#1234)" squash-merge suffix GitHub appends to PR titles is
    removed, since it is metadata and never appears in the body;
  * git trailer lines (Co-authored-by:, Signed-off-by:, ...) are dropped
    from the body, so a body made only of trailers becomes empty;
  * commits whose cleaned title+body duplicate an earlier one are skipped
    (git log --all sees cherry-picks and release branches as separate
    commits, which put identical documents in the index).
"""

import json
import re
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/huggingface/datasets.git"
CLONE_DIR = Path(__file__).parent / "data_src" / "repo"
OUT_PATH = Path(__file__).parent / "data" / "commits.jsonl"

PR_SUFFIX = re.compile(r"\s*\(#\d+\)\s*$")
TRAILER = re.compile(r"^(co-authored-by|signed-off-by|reviewed-by|acked-by|tested-by|cc):", re.I)

RECORD_SEP = "\x1e"
FIELD_SEP = "\x1f"


def clone_if_needed() -> None:
    if (CLONE_DIR / ".git").exists():
        print(f"Repo already cloned at {CLONE_DIR}")
        return
    CLONE_DIR.parent.mkdir(parents=True, exist_ok=True)
    print(f"Cloning {REPO_URL} (metadata only, no file contents)...")
    subprocess.run(
        [
            "git", "clone", "--filter=blob:none", "--no-checkout", "--quiet",
            REPO_URL, str(CLONE_DIR),
        ],
        check=True,
    )


def clean_title(title: str) -> str:
    return PR_SUFFIX.sub("", title).strip()


def clean_body(body: str) -> str:
    lines = [ln for ln in body.splitlines() if not TRAILER.match(ln.strip())]
    return "\n".join(lines).strip()


def extract_commits(limit: int = 2500) -> list[dict]:
    fmt = f"{RECORD_SEP}%H{FIELD_SEP}%an{FIELD_SEP}%ad{FIELD_SEP}%s{FIELD_SEP}%b"
    raw = subprocess.run(
        ["git", "log", "--all", "--no-merges", f"--pretty=format:{fmt}", "--date=short"],
        cwd=CLONE_DIR,
        check=True,
        capture_output=True,
        encoding="utf-8",  # git emits UTF-8; the platform default (cp1252 on Windows) is not enough
        errors="replace",
    ).stdout

    records = []
    seen: set[str] = set()
    for chunk in raw.split(RECORD_SEP):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = chunk.split(FIELD_SEP)
        if len(parts) < 4:
            continue
        commit_hash, author, date, subject = parts[0], parts[1], parts[2], parts[3]
        body = clean_body(parts[4]) if len(parts) > 4 else ""
        subject = clean_title(subject)
        if not subject:
            continue
        # `git log --all` walks every branch, so cherry-picked and release-branch
        # commits show up more than once with different hashes. Keep the first.
        key = subject + "\n" + body
        if key in seen:
            continue
        seen.add(key)
        records.append(
            {
                "id": commit_hash[:10],
                "title": subject,
                "body": body,
                "text": (subject + "\n" + body).strip(),
                "author": author,
                "date": date,
            }
        )
        if len(records) >= limit:
            break
    return records


def main() -> None:
    clone_if_needed()
    records = extract_commits(limit=2500)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} records to {OUT_PATH}")
    if records:
        print("Sample record:")
        print(json.dumps(records[0], indent=2, ensure_ascii=False)[:500])


if __name__ == "__main__":
    sys.exit(main())
