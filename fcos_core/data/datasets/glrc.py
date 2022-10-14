import os

import torch
import torch.utils.data
from PIL import Image
import sys
import cv2
import numpy as np

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET


from fcos_core.structures.bounding_box import BoxList

import logging
from fcos_core.config import cfg

class GLRCDataset(torch.utils.data.Dataset):

    CLASSES = (
        "__background__ ",
        "adenomatous",
        "hyperplastic",
    )

    def __init__(self, data_dir, split, use_difficult=False, transforms=None, is_source= True):
        self.root = data_dir
        self.image_set = split
        self.keep_difficult = use_difficult
        self.transforms = transforms

        if cfg.SOLVER.IDN:
            self._annopath = os.path.join(self.root, "Noisy_Annotations_seed_1997_idn_"+str(cfg.SOLVER.IDN), "%s.xml")
        else:
            if not cfg.SOLVER.ANNOTATIONS:
                self._annopath = os.path.join(self.root, "Annotations", "%s.xml")
            else:
                self._annopath = os.path.join(self.root, "Noisy_Annotations_seed_1997_per_"+str(cfg.SOLVER.ANNOTATIONS)+"00000", "%s.xml")
        self._imgpath = os.path.join(self.root, "JPEGImages", "%s.jpg")
        self._imgsetpath = os.path.join(self.root, "ImageSets", "Main", "%s.txt")

        with open(self._imgsetpath % self.image_set) as f:
            ori_ids = f.readlines()
        ori_ids = [x.strip("\n") for x in ori_ids]
        
        logger = logging.getLogger(__name__)
        # Remove negative images:

        cls = GLRCDataset.CLASSES
        self.class_count = {k: 0 for k in cls}
        del self.class_count["__background__ "]
        
        logger.info("Removing negative images ... ==>")
        ids = []
        for id in ori_ids:
            anno = ET.parse(self._annopath % id).getroot()
            ress = self._preprocess_annotation(anno)
            if ress:
                ids.append(id)
                ######### Count class objects
                for i in ress["labels"]:
                    self.class_count[self.map_class_id_to_class_name(int(i))] += 1

        logger.info('{} ==> {}'.format(len(ori_ids), len(ids)))
        for itm in self.class_count.items():
            logger.info('| {} : {} |'.format(itm[0], itm[1]))
        self.ids = ids

        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}

        self.class_to_ind = dict(zip(cls, range(len(cls))))

        self.is_source = is_source

    def __getitem__(self, index):
        img_id = self.ids[index]
        img = Image.open(self._imgpath % img_id).convert("RGB")

        target = self.get_groundtruth(index)
        target = target.clip_to_image(remove_empty=True)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target, index, img_id

    def __len__(self):
        return len(self.ids)

    def get_groundtruth(self, index):
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        anno = self._preprocess_annotation(anno)

        height, width = anno["im_info"]
        target = BoxList(anno["boxes"], (width, height), mode="xyxy")
        target.add_field("labels", anno["labels"])
        target.add_field("difficult", anno["difficult"])
        domain_labels = torch.ones_like(anno["labels"], dtype=torch.uint8) if self.is_source else torch.zeros_like(anno["labels"], dtype=torch.uint8)
        target.add_field("is_source", domain_labels)

        return target

    def _preprocess_annotation(self, target):
        boxes = []
        gt_classes = []
        difficult_boxes = []
        TO_REMOVE = 1
        cls = GLRCDataset.CLASSES
        class_to_ind = dict(zip(cls, range(len(cls))))
        
        # if "object" in target:
        for obj in target.iter("object"):
            difficult = int(obj.find("difficult").text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find("name").text.lower().strip()
            # ignore if not car
            if not name in cls:
                continue
            bb = obj.find("bndbox")
            # Make pixel indexes 0-based
            # Refer to "https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/datasets/pascal_voc.py#L208-L211"
            box = [
                float(bb.find("xmin").text), 
                float(bb.find("ymin").text), 
                float(bb.find("xmax").text), 
                float(bb.find("ymax").text),
            ]
            bndbox = tuple(
                map(lambda x: x - TO_REMOVE, list(map(int, box)))
            )

            boxes.append(bndbox)
            gt_classes.append(class_to_ind[name])
            difficult_boxes.append(difficult)

        size = target.find("size")
        im_info = tuple(map(int, (size.find("height").text, size.find("width").text)))

        if boxes == []:
            return None
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
        im_info = tuple(map(int, (size.find("height").text, size.find("width").text)))
        return {"height": im_info[0], "width": im_info[1]}

    def get_img_infov2(self, index):
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        size = anno.find("size")
        im_info = tuple(map(int, (size.find("height").text, size.find("width").text)))
        return {"height": im_info[0], "width": im_info[1], "name": img_id}

    def map_class_id_to_class_name(self, class_id):
        return GLRCDataset.CLASSES[class_id]

    def get_img_for_vis(self, index):
        img_id = self.ids[index]
        img = cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)[:, :, ::-1]
        img = np.asarray(img).clip(0, 255).astype(np.uint8)
        return img_id, img
        
    def map_class_id_to_class_color(self):
        colors = ["cyan", "r", "yellow", "b", 
                "slateblue", "darkgreen", "m", 
                "lime", "r", "goldenrod", 
                "peru", "purple", "indigo", 
                "fuchsia", "tan",
                "b", "gray", "yellow"]
        return colors