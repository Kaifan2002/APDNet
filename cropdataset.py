import os
import cv2
import math

CROP_SIZE = 512
OVERLAP = 0.2
STRIDE = int(CROP_SIZE * (1 - OVERLAP))  # 410


def sliding_crop_coords(h, w, crop, stride):
    ys = list(range(0, h - crop + 1, stride))
    xs = list(range(0, w - crop + 1, stride))

    if ys[-1] != h - crop:
        ys.append(h - crop)
    if xs[-1] != w - crop:
        xs.append(w - crop)

    return ys, xs


def process_pair(input_path, gt_path,
                 out_input_dir, out_gt_dir):
    name = os.path.basename(input_path)
    base = os.path.splitext(name)[0]

    inp = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    gt = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)

    if inp is None or gt is None:
        print(f"❌ Failed to read: {name}")
        return

    assert inp.shape[:2] == gt.shape[:2], \
        f"Size mismatch: {name}"

    h, w = inp.shape[:2]
    ys, xs = sliding_crop_coords(h, w, CROP_SIZE, STRIDE)

    idx = 0
    for y in ys:
        for x in xs:
            inp_patch = inp[y:y + CROP_SIZE, x:x + CROP_SIZE]
            gt_patch = gt[y:y + CROP_SIZE, x:x + CROP_SIZE]

            inp_out = f"{base}_{idx:04d}.png"
            gt_out = f"{base}_{idx:04d}.png"

            cv2.imwrite(os.path.join(out_input_dir, inp_out), inp_patch)
            cv2.imwrite(os.path.join(out_gt_dir, gt_out), gt_patch)

            idx += 1


def crop_dataset(input_dir, gt_dir,
                 out_input_dir, out_gt_dir):

    os.makedirs(out_input_dir, exist_ok=True)
    os.makedirs(out_gt_dir, exist_ok=True)

    names = sorted(os.listdir(input_dir))

    for name in names:
        if not name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            continue

        input_path = os.path.join(input_dir, name)
        gt_path = os.path.join(gt_dir, name)

        if not os.path.exists(gt_path):
            print(f"⚠️ Missing GT: {name}")
            continue

        process_pair(input_path, gt_path,
                     out_input_dir, out_gt_dir)


if __name__ == "__main__":
    crop_dataset(
        input_dir="/home/temp/NTIRE2026_DENOISE/jdllie_val_in",
        gt_dir="/home/temp/NTIRE2026_DENOISE/jdllie_val_gt",
        out_input_dir="/home/temp/NTIRE2026_DENOISE/val_in_512",
        out_gt_dir="/home/temp/NTIRE2026_DENOISE/val_gt_512"
    )
