import os
import json

import torch
import numpy as np
import pandas as pd
import open_clip
import pydicom

from PIL import Image

from tqdm import tqdm
from torchmetrics import Accuracy, F1Score

# Constants and Configuration (all paths overridable via env vars)
DIR_IMAGES = os.environ.get("DIR_IMAGES", "<path_to_images>")
PATH_CSV = os.environ.get("PATH_CSV", "../processed_iter_1.csv")
PATH_FETALCLIP_WEIGHT = os.environ.get("PATH_FETALCLIP_WEIGHT", "../FetalCLIP_weights.pt")
PATH_FETALCLIP_CONFIG = os.environ.get("PATH_FETALCLIP_CONFIG", "./fetal_clip_config.json")
PATH_TEXT_PROMPTS = os.environ.get("PATH_TEXT_PROMPTS", "./classes.json")
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "16"))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "4"))
IMAGE_EXT = os.environ.get("IMAGE_EXT", "")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Load model configuration
with open(PATH_FETALCLIP_CONFIG, "r") as file:
    config_fetalclip = json.load(file)
open_clip.factory._MODEL_CONFIGS["FetalCLIP"] = config_fetalclip


def load_text_prompts(path):
    """
    Load classes.json and separate evaluable classes (with prompt dicts)
    from 'other' classes (plain integer values).

    Returns:
        text_prompts:    dict {class_name: [prompt_strings]} — only evaluable classes
        id_to_classname: dict {class_id: class_name} — maps numeric id → class key
        other_class_ids: set of class_ids that belong to 'other' categories
    """
    with open(path, 'r') as f:
        raw = json.load(f)

    text_prompts = {}
    id_to_classname = {}
    other_class_ids = set()

    for key, val in raw.items():
        if isinstance(val, dict):
            text_prompts[key] = val['prompts']
            id_to_classname[val['id']] = key
        else:
            other_class_ids.add(val)

    return text_prompts, id_to_classname, other_class_ids


class DatasetProcessed(torch.utils.data.Dataset):
    """Dataset backed by processed_iter_1.csv (columns: class, image, class_idx)."""

    def __init__(self, dir_images, path_csv, preprocess,
                 exclude_class_ids=None, image_ext=".png"):
        self.root = dir_images
        self.preprocess = preprocess
        self.image_ext = image_ext

        if exclude_class_ids is None:
            exclude_class_ids = set()

        df = pd.read_csv(path_csv)

        # Drop rows whose class_idx falls into "other" categories
        df = df[~df['class_idx'].isin(exclude_class_ids)]

        self.data = []
        for _, row in df.iterrows():
            self.data.append({
                'img': os.path.join(self.root, f"{row['image']}{self.image_ext}"),
                'class_name': row['class'],
                'class_idx': int(row['class_idx']),
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = load_dicom_as_pil(self.data[index]['img'])
        img = make_image_square_with_zero_padding(img)
        img = self.preprocess(img)
        class_idx = self.data[index]['class_idx']

        return img, class_idx, self.data[index]['img']


def load_dicom_as_pil(path):
    """Read a DICOM file and return an RGB PIL Image."""
    ds = pydicom.dcmread(path)
    arr = ds.pixel_array  # numpy array

    # Apply modality LUT / VOI LUT if available
    if hasattr(pydicom.pixel_data_handlers, 'apply_voi_lut'):
        from pydicom.pixel_data_handlers.util import apply_voi_lut
        arr = apply_voi_lut(arr, ds)

    # Normalize to 0-255 uint8
    arr = arr.astype(np.float32)
    arr -= arr.min()
    denom = arr.max()
    if denom > 0:
        arr = arr / denom
    arr = (arr * 255).astype(np.uint8)

    # Handle PhotometricInterpretation
    photometric = getattr(ds, 'PhotometricInterpretation', 'MONOCHROME2')
    if photometric == 'MONOCHROME1':
        arr = 255 - arr

    # Convert to PIL RGB
    if arr.ndim == 2:
        img = Image.fromarray(arr, mode='L').convert('RGB')
    elif arr.ndim == 3 and arr.shape[2] == 3:
        img = Image.fromarray(arr, mode='RGB')
    elif arr.ndim == 3 and arr.shape[2] == 4:
        img = Image.fromarray(arr, mode='RGBA').convert('RGB')
    else:
        # Fallback: take first channel
        img = Image.fromarray(arr[:, :, 0] if arr.ndim == 3 else arr, mode='L').convert('RGB')

    return img


def make_image_square_with_zero_padding(image):
    width, height = image.size

    max_side = max(width, height)

    if image.mode == 'RGBA':
        image = image.convert('RGB')

    if image.mode == "RGB":
        padding_color = (0, 0, 0)
    elif image.mode == "L":
        padding_color = 0
    else:
        padding_color = 0

    new_image = Image.new(image.mode, (max_side, max_side), padding_color)

    padding_left = (max_side - width) // 2
    padding_top = (max_side - height) // 2

    new_image.paste(image, (padding_left, padding_top))

    return new_image

def main(checkpoint):
    # Load text prompts; id_to_classname maps class_idx → classes.json key
    # (avoids string-matching issues between CSV class names and classes.json keys)
    text_prompts, id_to_classname, other_class_ids = load_text_prompts(PATH_TEXT_PROMPTS)

    # Sequential index ↔ class name (order follows text_prompts iteration order)
    planename_to_index = {key: i for i, key in enumerate(text_prompts.keys())}
    index_to_planename = {val: key for key, val in planename_to_index.items()}

    # Map original class_idx (from CSV / classes.json id) → sequential index
    classidx_to_seqidx = {
        cls_id: planename_to_index[cls_name]
        for cls_id, cls_name in id_to_classname.items()
    }

    model, _, preprocess = open_clip.create_model_and_transforms("FetalCLIP", pretrained=checkpoint)
    tokenizer = open_clip.get_tokenizer("FetalCLIP")

    ds = DatasetProcessed(
        DIR_IMAGES, PATH_CSV, preprocess,
        exclude_class_ids=other_class_ids, image_ext=IMAGE_EXT,
    )
    dl = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model.eval()
    model.to(device)

    # Encode text features for each evaluable class
    list_text_features = []
    with torch.no_grad(), torch.cuda.amp.autocast():
        for plane, prompts in tqdm(text_prompts.items()):
            prompts = tokenizer(prompts).to(device)
            text_features = model.encode_text(prompts)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            text_features = text_features.mean(dim=0).unsqueeze(0)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            list_text_features.append(text_features)

    # Classify images
    list_paths = []
    list_gt_classidx = []
    list_probs = []
    for imgs, class_idxs, paths in tqdm(dl):
        with torch.no_grad(), torch.cuda.amp.autocast():
            imgs = imgs.to(device)
            image_features = model.encode_image(imgs)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            list_text_logits = []
            for text_features in list_text_features:
                text_logits = (100.0 * image_features @ text_features.T).mean(dim=-1)[:, None]
                list_text_logits.append(text_logits)
            text_probs = torch.cat(list_text_logits, dim=1).softmax(dim=-1)

            list_paths.extend(paths)
            list_gt_classidx.extend(class_idxs.tolist())
            list_probs.append(text_probs.cpu())

    probs = torch.cat(list_probs, dim=0).detach().cpu()
    preds = torch.argmax(probs, dim=1)

    list_pred = [index_to_planename[pred.item()] for pred in preds]

    # Map ground-truth class_idx → sequential index used by the classifier
    targets = torch.tensor([classidx_to_seqidx[idx] for idx in list_gt_classidx])

    num_classes = len(planename_to_index)
    acc = Accuracy(task="multiclass", num_classes=num_classes, top_k=1, average='none')(probs, targets)
    f1 = F1Score(task="multiclass", num_classes=num_classes, top_k=1, average='none')(probs, targets)

    acc_top2 = Accuracy(task="multiclass", num_classes=num_classes, top_k=2, average='none')(probs, targets)
    f1_top2 = F1Score(task="multiclass", num_classes=num_classes, top_k=2, average='none')(probs, targets)

    acc_top3 = Accuracy(task="multiclass", num_classes=num_classes, top_k=3, average='none')(probs, targets)
    f1_top3 = F1Score(task="multiclass", num_classes=num_classes, top_k=3, average='none')(probs, targets)

    list_targets = sorted(targets.unique().tolist())
    acc = retrieve_based_on_indexes(acc, list_targets)
    f1 = retrieve_based_on_indexes(f1, list_targets)
    acc_top2 = retrieve_based_on_indexes(acc_top2, list_targets)
    f1_top2 = retrieve_based_on_indexes(f1_top2, list_targets)
    acc_top3 = retrieve_based_on_indexes(acc_top3, list_targets)
    f1_top3 = retrieve_based_on_indexes(f1_top3, list_targets)
    list_classes = [index_to_planename[x] for x in list_targets]

    print('Classes')
    print(list_classes)
    print('')
    print(f'acc: {np.mean(acc):.4f} | {[f"{x:.4f}" for x in acc]}')
    print(f'f1 : {np.mean(f1):.4f} | {[f"{x:.4f}" for x in f1]}')
    print('')
    print(f'acc_top2: {np.mean(acc_top2):.4f} | {[f"{x:.4f}" for x in acc_top2]}')
    print('')
    print(f'acc_top3: {np.mean(acc_top3):.4f} | {[f"{x:.4f}" for x in acc_top3]}')

    data = {
            'planename_to_index': planename_to_index,
            'targets': targets.detach().cpu(),
            'probs': probs.detach().cpu(),
            'paths': list_paths,
            'list_classes': list_classes,
            'acc': acc,
            'f1': f1,
            'acc_top2': acc_top2,
            'acc_top3': acc_top3,
        }
    list_data = [{
        'model': "FetalCLIP",
        'f1': np.mean(data['f1']),
        'acc': np.mean(data['acc']),
        'acc_top2': np.mean(data['acc_top2']),
        'acc_top3': np.mean(data['acc_top3']),
        **{
            f'f1_{key}': val for key, val in zip(data['list_classes'], data['f1'])
        },
        **{
            f'acc_{key}': val for key, val in zip(data['list_classes'], data['acc'])
        },
        **{
            f'acc_top2_{key}': val for key, val in zip(data['list_classes'], data['acc_top2'])
        },
        **{
            f'acc_top3_{key}': val for key, val in zip(data['list_classes'], data['acc_top3'])
        },
    }]

    df = pd.DataFrame(list_data)
    df.to_csv('test_results.csv', index=False)

    list_gt_names = [id_to_classname[idx] for idx in list_gt_classidx]
    with open("test_prediction.json", "w") as f:
        json.dump(
            {"prediction": list_pred, "label": list_gt_names},
            f, indent=2, ensure_ascii=False,
        )

def retrieve_based_on_indexes(vals, list_indexes):
    return [x.detach().item() for i, x in enumerate(vals) if i in list_indexes]

if __name__ == '__main__':
    main(PATH_FETALCLIP_WEIGHT)