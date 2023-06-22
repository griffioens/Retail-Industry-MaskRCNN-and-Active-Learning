import numpy as np
import os, json, cv2, random

# detectron2 utilities
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode

# Creating a list of categories to register to detectron2
def class_list(annotation_file_path):
    with open(annotation_file_path) as f:
        imgs_anns = json.load(f)
    cat_list = []
    for category in imgs_anns["categories"]:
        cat_list.append(category["name"])
    return cat_list

# Function to convert the annotation file to a dictionary as required by detectron2
def get_supermarket_dicts(annotation_file_path):
    json_file = annotation_file_path
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    
    for image_record in imgs_anns["images"]:
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
    return dataset_dicts

if __name__ == '__main__':
    # Calculating the number of images in the training set to determine epoch length
    with open("../supermarket_dataset/alteredannotations/D2S_training_newsplit.json") as f:
        counter = json.load(f)
    image_count = len(counter["images"])
    print(f"Image count in training set: {image_count}")

    # CONSTANTS
    EXPERIMENTNAME = "experiment19"
    EPOCHS = 50
    LEARNING_RATE = 0.001
    step_1 = int((image_count/2) * EPOCHS * 0.75)
    LEARNING_RATE_DECAY_STEP = (step_1,) #empty list if no decay, otherwise an iteratble with step numberw here to decay weight eg. (3000, ) will decay the weight at the 3000th iteration (with 2 images per iteration this is at image 6000)
    ROI_HEADS_LOSS_CALCULATION = 128

    # Create the directory where the results are stored
    outputdir = "output" + EXPERIMENTNAME
    os.makedirs(outputdir, exist_ok=True)


    # Registering the dataset
    DatasetCatalog.register("supermarket_train", lambda: get_supermarket_dicts("../supermarket_dataset/alteredannotations/D2S_training_newsplit.json"))

    # Registering the class list
    list_classes = class_list("../supermarket_dataset/alteredannotations/D2S_training_newsplit.json")
    MetadataCatalog.get("supermarket_train").set(thing_classes=list_classes)
    supermarket_metadata = MetadataCatalog.get("supermarket_train")


    #Setting up the config file
    from detectron2.engine import DefaultTrainer
    cfg = get_cfg()
    cfg.MODEL.DEVICE = "cuda"
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("supermarket_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = LEARNING_RATE  
    cfg.SOLVER.MAX_ITER = int((image_count/2) * EPOCHS)    
    #cfg.SOLVER.MAX_ITER = 100   
    cfg.SOLVER.STEPS = LEARNING_RATE_DECAY_STEP        
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = ROI_HEADS_LOSS_CALCULATION   
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 60 
    cfg.INPUT.MASK_FORMAT = "bitmask" # note to self: bitmask here because the mvtec annotations uses runlength encoding instead of keypoints
    cfg.OUTPUT_DIR = outputdir

    trainer = DefaultTrainer(cfg) 
    trainer.resume_or_load(resume=False)
    trainer.train()

    ######################################################
    ##### inference from here onwards #########
    #######################################################
    # Load modules for inference
    from detectron2.evaluation import COCOEvaluator, inference_on_dataset
    from detectron2.data import build_detection_test_loader


    # Load trained weights
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  

    # Register the test set
    DatasetCatalog.register("supermarket_validation", lambda: get_supermarket_dicts("../supermarket_dataset/alteredannotations/D2S_validation_newsplit.json"))
    list_classes = class_list("../supermarket_dataset/alteredannotations/D2S_validation_newsplit.json")
    MetadataCatalog.get("supermarket_validation").set(thing_classes=list_classes)

    for treshold in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = treshold   # set a custom testing threshold
        predictor = DefaultPredictor(cfg)
        evaluator = COCOEvaluator("supermarket_validation", cfg, False, output_dir=outputdir)
        val_loader = build_detection_test_loader(cfg, "supermarket_validation")
        outcome = inference_on_dataset(predictor.model, val_loader, evaluator)
        with open(os.path.join(outputdir, f"mAP{treshold}.json"), 'w') as f:
            json.dump(outcome, f)

























