import pandas as pd
import torch
from detectron2.structures import Boxes, pairwise_iou

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
import cv2
import json
from detectron2 import model_zoo
import numpy as np
import os, json, cv2, random

# detectron2 utilities
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2.structures import BoxMode

import torch, torchvision
print(torch.__version__, torch.cuda.is_available())
import seaborn as sn
import matplotlib.pyplot as plt


def coco_bbox_to_coordinates(bbox):
    out = bbox.copy().astype(float)
    out[:, 2] = bbox[:, 0] + bbox[:, 2]
    out[:, 3] = bbox[:, 1] + bbox[:, 3]
    return out

def conf_matrix_calc(labels, detections, n_classes, conf_thresh, iou_thresh):
    confusion_matrix = np.zeros([n_classes + 1, n_classes + 1])
    l_classes = np.array(labels)[:, 0].astype(int)
    l_bboxs = coco_bbox_to_coordinates((np.array(labels)[:, 1:]))
    d_confs = np.array(detections)[:, 4]
    d_bboxs = (np.array(detections)[:, :4])
    d_classes = np.array(detections)[:, -1].astype(int)
    detections = detections[np.where(d_confs > conf_thresh)]
    labels_detected = np.zeros(len(labels))
    detections_matched = np.zeros(len(detections))
    for l_idx, (l_class, l_bbox) in enumerate(zip(l_classes, l_bboxs)):
        for d_idx, (d_bbox, d_class) in enumerate(zip(d_bboxs, d_classes)):
            iou = pairwise_iou(Boxes(torch.from_numpy(np.array([l_bbox]))), Boxes(torch.from_numpy(np.array([d_bbox]))))
            if iou >= iou_thresh:
                confusion_matrix[l_class, d_class] += 1
                labels_detected[l_idx] = 1
                detections_matched[d_idx] = 1
    for i in np.where(labels_detected == 0)[0]:
        confusion_matrix[l_classes[i], -1] += 1
    for i in np.where(detections_matched == 0)[0]:
        confusion_matrix[-1, d_classes[i]] += 1
    return confusion_matrix

def get_supermarket_dicts(annotation_file_path, image_list=False):
    json_file = annotation_file_path
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    
    for image_record in imgs_anns["images"]:
        image_id = image_record['id']
        if not image_list or image_id in image_list:
            record = {}
            file_name = os.path.join("../supermarket_dataset/images/", image_record["file_name"])
            record["file_name"] = file_name
            record["image_id"] = image_record["id"]
            record["height"] = image_record["height"]
            record["width"] = image_record["width"]
            
            annots = []
            for segmentation_orig in imgs_anns["annotations"]:
                segment_record = {}
                if segmentation_orig["image_id"] == record["image_id"]:
                    segment_record["bbox"] = segmentation_orig["bbox"]
                    segment_record["bbox_mode"] = BoxMode.XYWH_ABS
                    segment_record["segmentation"] = segmentation_orig["segmentation"]
                    segment_record["category_id"] = segmentation_orig["category_id"] - 1
                if segment_record:
                    annots.append(segment_record)
            record["annotations"] = annots
            dataset_dicts.append(record)
    print(f"converting {len(dataset_dicts)} images registering", flush=True)
    return dataset_dicts

# Creating a list of categories to register to detectron2
def class_list(annotation_file_path):
    with open(annotation_file_path) as f:
        imgs_anns = json.load(f)
    cat_list = []
    for category in imgs_anns["categories"]:
        cat_list.append(category["name"])
    return cat_list


cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.WEIGHTS = "outputsubset1_benchmark/model_final.pth"
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 60 
predictor = DefaultPredictor(cfg)


DatasetCatalog.register("supermarket_test", lambda: get_supermarket_dicts("../supermarket_dataset/alteredannotations/D2S_test_newsplit.json"))
list_classes = class_list("../supermarket_dataset/alteredannotations/D2S_test_newsplit.json")
MetadataCatalog.get("supermarket_test").set(thing_classes=list_classes)

dataset_dicts_validation = DatasetCatalog.get("supermarket_test")

n_classes = 60
confusion_matrix = np.zeros([n_classes + 1, n_classes + 1])
for d in dataset_dicts_validation:
    img = cv2.imread(d["file_name"])
    outputs = predictor(img)
    labels = list()
    detections = list()
    for coord, conf, cls, ann in zip(
        outputs["instances"].get("pred_boxes").tensor.cpu().numpy(),
        outputs["instances"].get("scores").cpu().numpy(),
        outputs["instances"].get("pred_classes").cpu().numpy(),
        d["annotations"]
    ):
        labels.append([ann["category_id"]] + ann["bbox"])
        detections.append(list(coord) + [conf] + [cls])
    confusion_matrix += conf_matrix_calc(np.array(labels), np.array(detections), n_classes, conf_thresh=0.5, iou_thresh=0.5)
matrix_indexes = list_classes + ["null"]
df = pd.DataFrame(confusion_matrix, columns=matrix_indexes, index=matrix_indexes)

df.to_csv("confusionmatrix_subset2.csv")

plt.figure(dpi = 600) 

# , cmap = "Reds"
sn.set(font_scale=0.5)
ax = sn.heatmap(df, annot=False, xticklabels = True, yticklabels=True, annot_kws={"size":8})
ax.set_xlabel("Predicted Class", fontdict={"size":14})
ax.set_ylabel("True Class", fontdict={"size":14})
#ax.set(xlabel = "Actual", ylabel="Predicted")

ax.figure.savefig("heatmapsubset1c.png", bbox_inches="tight")