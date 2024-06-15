import os

import torch
import torch.utils.data
from PIL import Image
import sys
import numpy as np
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

from maskrcnn_benchmark.structures.bounding_box import BoxList
import random
from ACVCGenerator import ACVCGenerator

class MultiWeatherDataset(torch.utils.data.Dataset):


    CLASSES = ('__background__',  # always index 0
                         'bus','bike','car','motor','person','rider','truck')

    def __init__(self, data_dir, split, use_difficult=False, transforms=None):
        self.root = data_dir
        self.image_set = split
        self.keep_difficult = use_difficult
        self.transforms = transforms
        self.acvc = ACVCGenerator()
        self.n_augs = 1
        self._annopath = os.path.join(self.root, "Annotations", "%s.xml")
        self._imgpath = os.path.join(self.root, "JPEGImages", "%s.jpg")
        self._imgsetpath = os.path.join(self.root, "ImageSets", "Main", "%s.txt")
        with open(self._imgsetpath % self.image_set) as f:
            self.ids = f.readlines()
        self.ids = [x.strip("\n") for x in self.ids]
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}

        cls = self.CLASSES
        self.class_to_ind = dict(zip(cls, range(len(cls))))
        self.categories = dict(zip(range(len(cls)), cls))
        self.corruption_func = [
            "defocus_blur",
            "glass_blur",
            "gaussian_blur",
            "motion_blur",
            "speckle_noise",
            "shot_noise",
            "impulse_noise",
            "gaussian_noise",
            "jpeg_compression",
            "pixelate",
            "elastic_transform",
            "brightness",
            "saturate",
            "contrast",
            "high_pass_filter",
            "phase_scaling"
        ]
        
    def __getitem__(self, index):
        is_train = 'train' in self.image_set
        img_id = self.ids[index]
        img = Image.open(self._imgpath % img_id).convert("RGB")
        if is_train:
            augs = self.corruption(np.copy(img))
        target_orig = self.get_groundtruth(index)
        target_orig = target_orig.clip_to_image(remove_empty=True)

        target_orig = self.get_groundtruth(index)
        target_orig = target_orig.clip_to_image(remove_empty=True)
        if self.transforms is not None:
            img, target = self.transforms(img, target_orig)
        images = [img]
        targets = [target]
        if self.transforms is not None and is_train:
            for aug in augs:
                aug, target_ = self.transforms(aug, target_orig)
                images.append(aug)
                targets.append(target_)
        if is_train:
            img = torch.stack(images)
            target = targets
        return img, target, index

    def __len__(self):
        return len(self.ids)

    def corruption(self, img):
        crs = random.sample(self.corruption_func, self.n_augs)
        images = []
        for c in crs:
            s = random.randint(1, 5)
            aug_img = self.acvc.apply_corruption(img, c, s).convert("RGB")
            images.append(aug_img)
        return images
    def get_groundtruth(self, index):
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        anno = self._preprocess_annotation(anno, img_id)
        
        height, width = anno["im_info"]
        target = BoxList(anno["boxes"], (width, height), mode="xyxy")
        target.add_field("labels", anno["labels"])
        target.add_field("difficult", anno["difficult"])
        return target

    def _preprocess_annotation(self, target, img_id=None):
        boxes = []
        gt_classes = []
        difficult_boxes = []
        TO_REMOVE = 1

        for obj in target.iter("object"):
            diffc = obj.find('difficult')
            difficult = 0 if diffc == None else int(diffc.text)
            # difficult = int(obj.find("difficult").text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find("name").text.lower().strip()
            bb = obj.find("bndbox")
            # Make pixel indexes 0-based
            # Refer to "https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/datasets/pascal_voc.py#L208-L211"
            box = [
                bb.find("xmin").text,
                bb.find("ymin").text,
                bb.find("xmax").text,
                bb.find("ymax").text,
            ]
            bndbox = tuple(
                map(lambda x: x - TO_REMOVE, list(map(float, box)))
                # map(lambda x: x - TO_REMOVE, list(map(int, list(map(float, box)) )) )
            )

            boxes.append(bndbox)
            gt_classes.append(self.class_to_ind[name])
            difficult_boxes.append(difficult)
        size = target.find("size")
        if size is not None:
            im_info = tuple(map(int, (size.find("height").text, size.find("width").text)))
        else:
            im_info = (720, 1280)

        res = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(gt_classes),
            "difficult": torch.tensor(difficult_boxes),
            "im_info": im_info,
        }
        return res

    def get_img_info(self, index):
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        size = anno.find("size")
        if size is not None:
            im_info = tuple(map(int, (size.find("height").text, size.find("width").text)))
        else:
            im_info = (720, 1280)
        return {"height": im_info[0], "width": im_info[1]}

    def map_class_id_to_class_name(self, class_id):
        return self.CLASSES[class_id]