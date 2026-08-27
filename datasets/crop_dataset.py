import os
import random
import shutil
import json
import argparse
from sklearn.model_selection import train_test_split


def make_dir(path):
    """
    创建输出目录
    """
    for split in ["train", "val"]:
        os.makedirs(
            os.path.join(path, split, "GT"),
            exist_ok=True
        )

        os.makedirs(
            os.path.join(path, split, "LQ"),
            exist_ok=True
        )



def make_task_dir(output_path, split, task):
    """
    创建某个任务的输出目录：
    output_path/{train,val}/{task}/{LQ,GT}
    """
    os.makedirs(os.path.join(output_path, split, task, "LQ"), exist_ok=True)
    os.makedirs(os.path.join(output_path, split, task, "GT"), exist_ok=True)


def collect_samples(task_path):
    """
    收集一个任务下的GT-LQ配对

    task_path
    ├── GT
    └── LQ
    """

    gt_dir = os.path.join(task_path, "GT")
    lq_dir = os.path.join(task_path, "LQ")

    if not os.path.exists(gt_dir):
        raise FileNotFoundError(f"GT目录不存在: {gt_dir}")

    if not os.path.exists(lq_dir):
        raise FileNotFoundError(f"LQ目录不存在: {lq_dir}")

    pairs = []

    for img_name in os.listdir(gt_dir):

        gt_path = os.path.join(gt_dir, img_name)
        lq_path = os.path.join(lq_dir, img_name)

        if (
            os.path.isfile(gt_path)
            and os.path.exists(lq_path)
        ):
            pairs.append((lq_path, gt_path))

    return pairs


def main(args):

    SAMPLE_RATIO = args.sample_ratio
    TEST_RATIO = args.test_ratio
    RANDOM_SEED = args.random_seed

    random.seed(RANDOM_SEED)

    make_dir(args.output_path)

    train_meta = []
    val_meta = []

    # 遍历所有任务
    for task in sorted(os.listdir(args.root)):

        task_path = os.path.join(args.root, task)

        if not os.path.isdir(task_path):
            continue

        pairs = collect_samples(task_path)

        print(f"[{task}] "f"原始样本数: {len(pairs)}")

        if len(pairs) == 0:
            continue

        # ------------------
        # 抽样
        # ------------------
        sample_num = max(1, int(len(pairs) * SAMPLE_RATIO))
        pairs = random.sample(pairs, sample_num)
        print(f"[{task}] " f"抽样后样本数: {len(pairs)}")


        train_pairs, val_pairs = train_test_split(pairs, test_size=TEST_RATIO, random_state=RANDOM_SEED)
        print(f"[{task}] " f"Train: {len(train_pairs)}, " f"Val: {len(val_pairs)}")

        # ==================
        # Train
        # ==================
        for inp, gt in train_pairs:
            ext = os.path.splitext(inp)[1]
            input_filename = os.path.basename(inp).split('.')[0]
            name = (
                f"{task}_"
                f"{input_filename}"
                f"{ext}"
            )
            make_task_dir(args.output_path, "train", task)

            train_lq_path = os.path.join(
                args.output_path,
                "train",
                "LQ",
                name
            )

            train_gt_path = os.path.join(
                args.output_path,
                "train",
                "GT",
                name
            )

            shutil.copy2(inp, train_lq_path)
            shutil.copy2(gt, train_gt_path)

            train_meta.append(
                {
                    "lq": f"train/LQ/{name}",
                    "gt": f"train/GT/{name}",
                    "task": task,
                }
            )

        # ==================
        # Val
        # ==================
        for inp, gt in val_pairs:
            input_filename = os.path.basename(inp).split('.')[0]
            ext = os.path.splitext(inp)[1]

            name = (
                f"{task}_"
                f"{input_filename}"
                f"{ext}"
            )
            make_task_dir(args.output_path, "val", task)


            val_lq_path = os.path.join(
                args.output_path,
                "val",
                "LQ",
                name
            )

            val_gt_path = os.path.join(
                args.output_path,
                "val",
                "GT",
                name
            )

            shutil.copy2(inp, val_lq_path)
            shutil.copy2(gt, val_gt_path)

            val_meta.append(
                {
                    "lq": f"val/LQ/{name}",
                    "gt": f"val/GT/{name}",
                    "task": task,
                }
            )
    print("=" * 60)
    print(f"Train Samples : {len(train_meta)}")
    print(f"Val Samples   : {len(val_meta)}")
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root",type=str, default="/home/temp/kaifan/Train")
    parser.add_argument( "--output_path", type=str, default="/home/qiaokaifan/datasets/lovif_crop", help="输出目录")
    parser.add_argument( "--sample_ratio", type=float,default=0.05, help="每个任务抽样比例")
    parser.add_argument( "--test_ratio", type=float, default=0.2, help="验证集比例")
    parser.add_argument("--random_seed", type=int, default=42)
    args = parser.parse_args()

    main(args)