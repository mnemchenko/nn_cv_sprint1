"""Convert COCO annotations to YOLO format (one .txt per image).

COCO bbox: [x_min, y_min, width, height] (absolute pixels)
YOLO bbox:  class_id x_center y_center width height (normalized 0-1)

The dataset has category_id=0 as a parent 'minecraft-mobs' category,
so we remap IDs 1-17 -> 0-16 for YOLO.
"""

import json
import os
from pathlib import Path


CATEGORY_REMAP = {i: i - 1 for i in range(1, 18)}


def convert_split(ann_path: str, img_dir: str, label_dir: str) -> None:
    os.makedirs(label_dir, exist_ok=True)

    with open(ann_path, 'r') as f:
        data = json.load(f)

    img_id_to_info = {img['id']: img for img in data['images']}

    img_id_to_anns: dict[int, list] = {}
    for ann in data['annotations']:
        cat_id = ann['category_id']
        if cat_id not in CATEGORY_REMAP:
            continue
        img_id_to_anns.setdefault(ann['image_id'], []).append(ann)

    for img_info in data['images']:
        img_id = img_info['id']
        w_img = img_info['width']
        h_img = img_info['height']
        stem = Path(img_info['file_name']).stem
        label_path = os.path.join(label_dir, f'{stem}.txt')

        anns = img_id_to_anns.get(img_id, [])
        lines = []
        for ann in anns:
            cat_id = CATEGORY_REMAP[ann['category_id']]
            x_min, y_min, bw, bh = ann['bbox']
            x_center = (x_min + bw / 2) / w_img
            y_center = (y_min + bh / 2) / h_img
            nw = bw / w_img
            nh = bh / h_img
            x_center = max(0, min(1, x_center))
            y_center = max(0, min(1, y_center))
            nw = max(0, min(1, nw))
            nh = max(0, min(1, nh))
            lines.append(f'{cat_id} {x_center:.6f} {y_center:.6f} {nw:.6f} {nh:.6f}')

        with open(label_path, 'w') as f:
            f.write('\n'.join(lines))

    print(f'{ann_path}: {len(data["images"])} images -> {label_dir}')


if __name__ == '__main__':
    base = 'datasets/minecraft'
    for split, ann_name in [('train', 'train_annotations.json'),
                            ('val', 'val_annotations.json'),
                            ('test', 'test_annotations.json')]:
        convert_split(
            ann_path=os.path.join(base, 'annotations', ann_name),
            img_dir=os.path.join(base, split),
            label_dir=os.path.join(base, 'labels', split),
        )
