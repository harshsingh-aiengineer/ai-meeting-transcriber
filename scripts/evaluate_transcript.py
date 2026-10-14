"""Compare a hypothesis transcript with a manual reference.

Example:
    python scripts/evaluate_transcript.py --reference notes.txt --hypothesis output.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.utils.evaluation import score_transcripts


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Score a hypothesis transcript against a manual reference."
    )
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--hypothesis", required=True, type=Path)
    args = parser.parse_args()
    reference = args.reference.read_text(encoding="utf-8")
    hypothesis = args.hypothesis.read_text(encoding="utf-8")
    scores = score_transcripts(reference, hypothesis)
    print(f"WER: {scores.wer:.4f}")
    print(f"CER: {scores.cer:.4f}")
    print(f"omissions: {scores.deletions} ({scores.omission_rate:.4f})")
    print(f"insertions: {scores.insertions} ({scores.insertion_rate:.4f})")
    print(f"substitutions: {scores.substitutions}")
    print(f"reference_words: {scores.reference_words}")
    print(f"hypothesis_words: {scores.hypothesis_words}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
