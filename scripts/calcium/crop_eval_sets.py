# Crop val_all/test_all theo bbox tim + 20 mm (cung quy tac voi Dataset911/912) -> val_crop, test_crop (+ kenh ung vien HU>=130); luu crop_boxes.json de dan du doan ve khung goc khi cham.
import json
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_cacseg_variants import crop, heart_bbox  # noqa: E402

W = Path("/workspace/cacseg-work")
H = W / "heart_flat"
boxes = {}
for name in ("val", "test"):
    src = W / f"{name}_all"
    o1 = W / f"{name}_crop"
    o2 = W / f"{name}_crop_cand"
    o1.mkdir(exist_ok=True)
    o2.mkdir(exist_ok=True)
    for img_path in sorted(src.glob("*_0000.nii.gz")):
        stem = img_path.name[:-12]
        hp = H / f"{stem}.nii.gz"
        if not hp.exists():
            print("no heart mask", stem)
            continue
        box = heart_bbox(sitk.ReadImage(str(hp)), 20.0)
        if box is None:
            print("empty heart", stem)
            continue
        lo, size = box
        c = crop(sitk.ReadImage(str(img_path)), lo, size)
        sitk.WriteImage(c, str(o1 / img_path.name))
        sitk.WriteImage(c, str(o2 / img_path.name))
        cand = sitk.GetImageFromArray((sitk.GetArrayFromImage(c) >= 130).astype(np.uint8))
        cand.CopyInformation(c)
        sitk.WriteImage(cand, str(o2 / f"{stem}_0001.nii.gz"))
        boxes[stem] = {"set": name, "lo": [int(v) for v in lo], "size": [int(v) for v in size]}
json.dump(boxes, open(W / "crop_boxes.json", "w"), indent=0)
print("CROP-EVAL-DONE", len(boxes))
