"""
Objective, low-level image features for all 10000 images in the N=10000
manifest: deterministic Resize-256-LANCZOS + CenterCrop-256, 8 features
computed with PIL + numpy + scipy only (no learned models).

Output: results/image_features.csv
"""
import csv
import io
import json
import os

import numpy as np
from PIL import Image
from scipy import ndimage

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(ROOT, 'results')

EDGE_THRESHOLD = 0.1


def load_canonical(path):
    with Image.open(path) as im:
        im = im.convert('RGB')
        w, h = im.size
        if w < h:
            new_w, new_h = 256, round(256 * h / w)
        else:
            new_h, new_w = 256, round(256 * w / h)
        im = im.resize((new_w, new_h), Image.LANCZOS)
        left = (new_w - 256) // 2
        top = (new_h - 256) // 2
        im = im.crop((left, top, left + 256, top + 256))
        return im


def compute_features(im: Image.Image) -> dict:
    rgb = np.asarray(im, dtype=np.float64)
    gray = np.asarray(im.convert('L'), dtype=np.float64) / 255.0

    hist, _ = np.histogram((gray * 255).astype(np.uint8), bins=256, range=(0, 255))
    p = hist / hist.sum()
    p = p[p > 0]
    entropy = float(-np.sum(p * np.log2(p)))

    sx = ndimage.sobel(gray, axis=1)
    sy = ndimage.sobel(gray, axis=0)
    mag = np.hypot(sx, sy)
    edge_density = float(np.mean(mag > EDGE_THRESHOLD))

    lap = ndimage.laplace(gray)
    sharpness = float(lap.var())

    luminance_contrast = float(gray.std())

    hsv = np.asarray(im.convert('HSV'), dtype=np.float64)
    mean_saturation = float(hsv[:, :, 1].mean() / 255.0)

    R, G, B = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    rg = R - G
    yb = 0.5 * (R + G) - B
    colorfulness = float(np.sqrt(rg.std() ** 2 + yb.std() ** 2) +
                          0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))

    c0, c1 = 64, 192
    center = gray[c0:c1, c0:c1]
    border_mask = np.ones_like(gray, dtype=bool)
    border_mask[c0:c1, c0:c1] = False
    center_vs_border = float(abs(center.mean() - gray[border_mask].mean()))

    buf = io.BytesIO()
    im.save(buf, format='JPEG', quality=90)
    jpeg_bytes_per_pixel = float(len(buf.getvalue()) / (256 * 256))

    return {
        'grayscale_entropy': entropy,
        'edge_density': edge_density,
        'laplacian_sharpness': sharpness,
        'luminance_contrast': luminance_contrast,
        'mean_saturation': mean_saturation,
        'colorfulness': colorfulness,
        'center_vs_border_contrast': center_vs_border,
        'jpeg_bytes_per_pixel': jpeg_bytes_per_pixel,
    }


def main():
    with open(os.path.join(RESULTS, 'dataset_manifest_N10000.json'), encoding='utf-8') as f:
        manifest = json.load(f)
    images = manifest['images']
    assert len(images) == 10000

    feature_names = None
    rows = []
    for im_rec in images:
        im = load_canonical(os.path.join(ROOT, im_rec['local_path']))
        feats = compute_features(im)
        if feature_names is None:
            feature_names = list(feats.keys())
        row = {'image_id': im_rec['image_id'], 'class_id': im_rec['class_id'],
               'image_filename': im_rec['filename'], 'is_old_subset': im_rec['is_old_subset']}
        row.update(feats)
        rows.append(row)
        if (im_rec['image_id'] + 1) % 1000 == 0:
            print(f'  {im_rec["image_id"]+1}/10000 images processed')

    out_path = os.path.join(RESULTS, 'image_features.csv')
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['image_id', 'class_id', 'image_filename', 'is_old_subset'] + feature_names)
        w.writeheader()
        w.writerows(rows)

    arr = np.array([[r[fn] for fn in feature_names] for r in rows])
    assert not np.isnan(arr).any(), 'NaN in computed features!'
    assert arr.shape == (10000, len(feature_names))
    print(f'\nwrote {out_path} (10000 rows, {len(feature_names)} features, no NaN)')


if __name__ == '__main__':
    main()
