#!/usr/bin/env python3
"""
build_anon_zip.py

Builds an anonymized code ZIP for ACCV 2026 supplementary submission.

Usage (run from ~/lite/):
    python build_anon_zip.py

Output:
    ~/lite/msfptrack_code_anon.zip
"""

import os
import re
import shutil
import zipfile
from pathlib import Path

# ── Source and destination ────────────────────────────────────────────────────
SRC  = Path.home() / "lite" / "litepp"
DEST = Path.home() / "lite" / "msfptrack_anon"
ZIP_OUT = Path.home() / "lite" / "msfptrack_code_anon.zip"

# ── Files / dirs to exclude entirely ─────────────────────────────────────────
EXCLUDE_DIRS = {
    "__pycache__", ".git", "*.egg-info", "litepp.egg-info",
    "docs",           # acknowledgments.md with identifying info
}
EXCLUDE_FILES = {
    "acknowledgments.md",
    "yolov8m.pt", "yolov8n.pt", "yolov8s.pt", "yolov8l.pt", "yolov8x.pt",
    "*.pt", "*.pth", "*.ckpt",   # no model weights
    # experiment scripts that use synthetic/fake data
    "gen_figs.py",
    "w3_exp.py",
    "generate_architecture_diagram.py",
    "generate_mot_visualizations.py",   # synthetic data only
    "generate_paper_visualizations.py", # synthetic data only
    "generate_speed_accuracy_plot.py",  # synthetic data only
}
EXCLUDE_EXTENSIONS = {".pyc", ".pyo", ".log", ".tmp", ".DS_Store"}

# ── Text replacements (applied to every .py / .md / .yaml / .txt file) ───────
# Order matters: more specific first.
REPLACEMENTS = [
    # Personal identifiers
    (r"muhiddin1979@inha\.edu",          "anonymous@anonymous.edu"),
    (r"muhiddin1979@[^\s\"']+",          "anonymous@anonymous.edu"),
    (r"\bmuxiddin19\b",                  "anonymous"),
    (r"\bmuxiddin\b",                    "anonymous"),
    (r"\bmuhiddin\b",                    "anonymous"),
    # Institution
    (r"Inha University",                 "[Anonymous Institution]"),
    (r"\bInha\b",                        "[Anonymous]"),
    (r"\bKorea\b",                       "[Anonymous Country]"),
    # Our venue: ECCV 2026 → ACCV 2026 (only for our own paper)
    (r"European Conference on Computer Vision \(ECCV\),\s*\n?\s*year=\{2026\}",
     "Asian Conference on Computer Vision (ACCV),\n  year={2026}"),
    (r"eccv2026",                        "accv2026"),
    (r"ECCV 2026",                       "ACCV 2026"),
    # GitHub links to personal repo
    (r"https://github\.com/muxiddin19/[^\s\"'\)]+",
     "https://github.com/[anonymous]/msfp-track"),
    (r"https://github\.com/muhiddin[^\s\"'\)]+",
     "https://github.com/[anonymous]/msfp-track"),
    # Home directory paths
    (r"/home/muhiddin/",                 "/path/to/"),
    (r"/home/muxiddin19/",               "/path/to/"),
    # Paper ID comment if present
    (r"Paper ID:?\s*557",                "Paper ID: [anonymous]"),
    # Author name in comments/strings
    (r"@author\s+muhiddin[^\n]*",        "@author Anonymous"),
    (r"@author\s+muxiddin[^\n]*",        "@author Anonymous"),
    # Author block in README bibtex
    (r"author=\{[Aa]non[^\}]*\}",        "author={Anonymous}"),
    (r"author=\{Muhiddin[^\}]*\}",       "author={Anonymous}"),
    (r"author=\{Muxiddin[^\}]*\}",       "author={Anonymous}"),
]

# Compile patterns once
COMPILED = [(re.compile(pat, re.IGNORECASE), repl) for pat, repl in REPLACEMENTS]


def should_exclude(path: Path) -> bool:
    """Return True if this path should be skipped."""
    # Check extension
    if path.suffix in EXCLUDE_EXTENSIONS:
        return True
    # Check file name
    for excl in EXCLUDE_FILES:
        if excl.startswith("*"):
            if path.suffix == excl[1:]:
                return True
        elif path.name == excl:
            return True
    # Check any parent dir name
    for part in path.parts:
        if part in EXCLUDE_DIRS:
            return True
        if part.endswith(".egg-info"):
            return True
    return False


def anonymize_text(text: str) -> str:
    for pattern, replacement in COMPILED:
        text = pattern.sub(replacement, text)
    return text


def process_file(src_file: Path, dest_file: Path):
    dest_file.parent.mkdir(parents=True, exist_ok=True)

    text_extensions = {".py", ".md", ".yaml", ".yml", ".txt", ".cfg",
                       ".ini", ".sh", ".rst", ".toml", ".json"}

    if src_file.suffix.lower() in text_extensions:
        try:
            text = src_file.read_text(encoding="utf-8", errors="replace")
            cleaned = anonymize_text(text)
            dest_file.write_text(cleaned, encoding="utf-8")
        except Exception as e:
            print(f"  [WARN] Could not process {src_file.name}: {e}")
            shutil.copy2(src_file, dest_file)
    else:
        shutil.copy2(src_file, dest_file)


def copy_script_itself(dest: Path):
    """Add the comparison video script to experiments/."""
    video_script = Path.home() / "lite" / "litepp" / "experiments" / \
                   "generate_comparison_videos.py"
    if video_script.exists():
        dst = dest / "experiments" / "generate_comparison_videos.py"
        dst.parent.mkdir(parents=True, exist_ok=True)
        text = video_script.read_text(encoding="utf-8", errors="replace")
        (dest / "experiments" / "generate_comparison_videos.py").write_text(
            anonymize_text(text), encoding="utf-8"
        )
        print("  + added experiments/generate_comparison_videos.py")


def build():
    print(f"Source : {SRC}")
    print(f"Staging: {DEST}")
    print(f"Output : {ZIP_OUT}")
    print()

    # Clean staging dir
    if DEST.exists():
        shutil.rmtree(DEST)

    skipped = []
    processed = []

    for src_file in SRC.rglob("*"):
        if src_file.is_dir():
            continue
        rel = src_file.relative_to(SRC)
        if should_exclude(src_file):
            skipped.append(str(rel))
            continue
        dest_file = DEST / rel
        process_file(src_file, dest_file)
        processed.append(str(rel))

    # Add comparison video script (newly created, may not be in litepp/ yet)
    copy_script_itself(DEST)

    print(f"Processed : {len(processed)} files")
    print(f"Skipped   : {len(skipped)} files")

    # ── Verification pass: check no identifiers remain ────────────────────────
    print("\nVerification scan for remaining identifiers...")
    DANGER = ["muhiddin", "muxiddin", "Inha University", "muxiddin19",
              "/home/muhiddin", "muhiddin1979"]
    found_issues = []
    for dest_file in DEST.rglob("*"):
        if dest_file.is_dir():
            continue
        if dest_file.suffix.lower() not in {".py", ".md", ".yaml", ".txt", ".rst"}:
            continue
        text = dest_file.read_text(encoding="utf-8", errors="replace")
        for term in DANGER:
            if term.lower() in text.lower():
                lines = [i+1 for i, l in enumerate(text.splitlines())
                         if term.lower() in l.lower()]
                found_issues.append(f"  [{dest_file.relative_to(DEST)}] "
                                    f"contains '{term}' at lines {lines}")

    if found_issues:
        print("  WARNING — identifiers still present:")
        for issue in found_issues:
            print(issue)
    else:
        print("  Clean — no personal identifiers found.")

    # ── Build ZIP ─────────────────────────────────────────────────────────────
    print(f"\nBuilding ZIP...")
    if ZIP_OUT.exists():
        ZIP_OUT.unlink()

    with zipfile.ZipFile(ZIP_OUT, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in DEST.rglob("*"):
            if f.is_file():
                zf.write(f, f"msfptrack_code/{f.relative_to(DEST)}")

    size_mb = ZIP_OUT.stat().st_size / 1024 / 1024
    print(f"ZIP size: {size_mb:.1f} MB  →  {ZIP_OUT}")
    print("\nDone.")


if __name__ == "__main__":
    build()
