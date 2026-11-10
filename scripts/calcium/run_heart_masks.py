"""Sinh mask tim (TotalSegmentator total --fast) cho bo kiem chuan nongated.

Dung de gioi han cham diem Agatston trong vung tim: model CACSeg train tren FOV
gated chua tung thay xuong suon/cot song cua CT nguc.
  python run_heart_masks.py --eval-root ~/cacseg-work/nongated_eval
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--eval-root",
        type=Path,
        default=None,
        help="layout cũ: <root>/images/nong_*_0000.nii.gz -> <root>/heart",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=None,
        help="hoặc: thư mục *_0000.nii.gz bất kỳ (vd imagesTr của nnU-Net)",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--device", default="gpu", choices=["gpu", "cpu"])
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    from totalsegmentator.python_api import totalsegmentator

    if args.images_dir is not None:
        if args.out_dir is None:
            parser.error("--images-dir can kem --out-dir")
        images = sorted(args.images_dir.glob("*_0000.nii.gz"))
        heart_dir = args.out_dir
    else:
        if args.eval_root is None:
            parser.error("can --eval-root hoac --images-dir")
        images = sorted((args.eval_root / "images").glob("nong_*_0000.nii.gz"))
        heart_dir = args.eval_root / "heart"
    if args.limit:
        images = images[: args.limit]
    heart_dir.mkdir(parents=True, exist_ok=True)
    done = failed = 0
    for img in images:
        cid = img.name[: -len("_0000.nii.gz")]
        dst = heart_dir / f"{cid}.nii.gz"
        if dst.exists():
            done += 1
            continue
        try:
            with tempfile.TemporaryDirectory(dir=heart_dir) as tmp:
                totalsegmentator(
                    str(img),
                    tmp,
                    fast=True,
                    task="total",
                    roi_subset=["heart"],
                    quiet=True,
                    device=args.device,
                )
                produced = Path(tmp) / "heart.nii.gz"
                if not produced.is_file():
                    raise RuntimeError("khong thay heart.nii.gz")
                shutil.move(str(produced), dst)
            done += 1
            if done % 20 == 0:
                print(f"da xong {done}/{len(images)}", flush=True)
        except Exception as exc:
            print(f"[!] {cid}: {type(exc).__name__}: {exc}", flush=True)
            failed += 1
    print(f"HOAN TAT MASK TIM: {done} ca, loi {failed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
