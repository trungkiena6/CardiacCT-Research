"""Print an official test case list, or create a new UTF-8 list without overwrite."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from coronary_tree.evaluate.export_contract import load_scopes  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("imagecas", "imagecas-x"), required=True)
    parser.add_argument("--split", type=int, choices=(1, 2, 3, 4))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if (args.dataset == "imagecas") != (args.split is not None):
        parser.error("--split is required for ImageCAS and must be omitted for ImageCAS-X")
    try:
        splits, lumen = load_scopes(
            REPO / "configs/imagecas_v2_split_manifest.csv",
            REPO / "configs/imagecasx_filelists/test.txt",
        )
        text = "\n".join(sorted(splits[args.split] if args.split else lumen)) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8", newline="\n") as handle:
                handle.write(text)
        else:
            print(text, end="")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
