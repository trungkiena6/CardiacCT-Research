"""MultiD4CAD full CCTA volumes (EAT folder): stored as HU + 1024 (float32, min 0). The NIfTI header claims RAS, but the stored
array axes are actually (anterior->posterior, right->left, superior->inferior) = "PLI" (checked on anatomy: sternum/apex at low i,
liver at low j, trachea/arch at low k). Earlier flip-only attempts left the array transposed; this v3 transform was used for the
recorded pipeline outputs. A controlled comparison attributing failure rates to orientation correction was not performed.

This writes HU int16 .nii.gz copies in the ImageCAS/nnU-Net LAS orientation (x -> left, y -> anterior, z -> superior):
    new = O.transpose(1, 0, 2)[:, ::-1, ::-1]      # x = j (R->L), y = reversed i (P->A), z = reversed k (I->S)
with affine diag(-s, +s, +sz) (LAS) to stage/images/<pid>_0000.nii.gz, and the radiologist EAT masks transformed identically to
stage/eat_ref/<pid>.nii.gz. P113's mask file is truncated in the dataset (no reference mask written).

Copied from the study workspace's convert_hu_v3.py on 17 October 2026. Its
image/reference transform, intensity conversion and zero-origin LAS affine are
retained. CLI paths and a truncated-mask check were added for release preparation.
This is a cohort-specific correction, not generic NIfTI canonicalization. The
output affine defines a corrected local frame; it does not preserve a trustworthy
original patient-space origin. Use matching converted images and references.
Existing output files are skipped; use a fresh output root for a new conversion.

Example:
  python scripts/convert_multid4cad.py --input-root <EAT-patient-folder> --output-root <stage-folder> --workers 2
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np


def to_las(arr):
    return np.ascontiguousarray(arr.transpose(1, 0, 2)[:, ::-1, ::-1])


def las_affine(zooms):
    si, sj, sk = [float(z) for z in zooms[:3]]
    aff = np.eye(4)
    aff[0, 0], aff[1, 1], aff[2, 2] = -sj, si, sk
    return aff


def one(args):
    d, S, R = args
    pid = d.name
    oi, om = S / f"{pid}_0000.nii.gz", R / f"{pid}.nii.gz"
    im = nib.load(str(d / "epicardialDatasetNIFTI.nii"))
    a = np.asanyarray(im.dataobj).astype(np.float32)
    assert a.min() >= 0, (pid, a.min())
    try:
        mask_path = d / "epicardialDatasetNIFTI_mask.nii"
        mask_image = nib.load(str(mask_path))
        expected_bytes = (
            int(mask_image.dataobj.offset)
            + int(np.prod(mask_image.shape)) * mask_image.get_data_dtype().itemsize
        )
        if mask_path.stat().st_size < expected_bytes:
            raise ValueError("Reference NIfTI payload is truncated")
        mk = np.asanyarray(mask_image.dataobj) > 0
        assert a.shape == mk.shape, (pid, a.shape, mk.shape)
    except Exception as e:  # noqa: BLE001
        print(pid, "mask_missing:", repr(e)[:120], flush=True)
        mk = None
    aff = las_affine(im.header.get_zooms())
    if not oi.exists():
        img = nib.Nifti1Image(
            to_las(np.clip(np.round(a - 1024.0), -1024, 3071).astype(np.int16)), aff
        )
        img.header.set_data_dtype(np.int16)
        img.set_qform(aff, code=1)
        img.set_sform(aff, code=1)
        nib.save(img, str(oi))
    if mk is not None and not om.exists():
        m = nib.Nifti1Image(to_las(mk.astype(np.uint8)), aff)
        m.set_qform(aff, code=1)
        m.set_sform(aff, code=1)
        nib.save(m, str(om))
    return pid, [int(s) for s in a.shape]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="Folder containing P001 ... P118 patient directories",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Fresh stage folder; writes images/ and eat_ref/",
    )
    parser.add_argument("--workers", type=int, default=2)
    options = parser.parse_args()
    if not options.input_root.is_dir():
        parser.error("--input-root must exist")
    if options.workers < 1:
        parser.error("--workers must be positive")
    patients = sorted(p for p in options.input_root.glob("P*") if p.is_dir())
    if not patients:
        parser.error("No P* patient directories were found")
    S, R = options.output_root / "images", options.output_root / "eat_ref"
    S.mkdir(parents=True, exist_ok=True)
    R.mkdir(parents=True, exist_ok=True)
    shapes = {}
    with ProcessPoolExecutor(options.workers) as ex:
        for i, (pid, shape) in enumerate(ex.map(one, ((d, S, R) for d in patients)), 1):
            shapes[pid] = shape
            if i % 20 == 0:
                print(i, pid, flush=True)
    (S.parent / "original_shapes.json").write_text(json.dumps(shapes, indent=1))
    print(
        "CONVERT-DONE",
        len(list(S.glob("*_0000.nii.gz"))),
        "images,",
        len(list(R.glob("*.nii.gz"))),
        "reference masks",
        flush=True,
    )
