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
import multiprocessing
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.engine import DefaultTrainer

##########################################


class IterationManager():
    def __init__(self, maximages, startamount, iterationamount) -> None:
        self.image_list = []
        self.unlabeled_image_list = []
        self.labeled_image_list = []
        self.keep_iterating = True
        self.max_images = maximages
        self.startamount = startamount
        self.iterationamount = iterationamount
        
    def loadcocolike(self, urllinks):
        for url in urllinks:
            with open(url, 'r') as f:
                d2s = json.load(f)
            for image in d2s['images']:
                self.image_list.append(image["id"])
        len_image_list = len(self.image_list)
        print(f"Total {len_image_list} image ids.")
    
    def initialsplit(self):
        nr_images = self.startamount
        
        if nr_images > len(self.image_list):
            nr_images = len(self.image_list)
        
        self.labeled_image_list = self.labeled_image_list + random.sample(self.image_list, nr_images)
        self.unlabeled_image_list = [imagenr for imagenr in self.image_list if imagenr not in self.labeled_image_list]

    def iteration(self):
        nr_images = self.iterationamount
        
        if nr_images > len(self.unlabeled_image_list):
            nr_images = len(self.unlabeled_image_list)
        
        self.labeled_image_list = self.labeled_image_list + random.sample(self.unlabeled_image_list, nr_images)
        self.unlabeled_image_list = [imagenr for imagenr in self.image_list if imagenr not in self.labeled_image_list]  

        if len(self.labeled_image_list) > self.max_images:
            self.keep_iterating = False 

# Creating a list of categories to register to detectron2
def class_list(annotation_file_path):
    with open(annotation_file_path) as f:
        imgs_anns = json.load(f)
    cat_list = []
    for category in imgs_anns["categories"]:
        cat_list.append(category["name"])
    return cat_list

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
    
def get_combined_dict(listing, image_list = False):
    final_list = []
    for dataset in listing:
        final_list = final_list + get_supermarket_dicts(dataset, image_list)
    return final_list

#########################################################################

def create_model_and_train(iterator, outputdir):
    # Calculating the number of images in the training set to determine epoch length
    image_count = len(iterator.labeled_image_list)
    print(f"Image count in training set: {image_count}")

    # CONSTANTS
    EPOCHS = 20
    LEARNING_RATE = 0.001
    step_1 = int((image_count/2) * EPOCHS * 0.75)
    LEARNING_RATE_DECAY_STEP = (step_1,) #empty list if no decay, otherwise an iteratble with step numberw here to decay weight eg. (3000, ) will decay the weight at the 3000th iteration (with 2 images per iteration this is at image 6000)
    ROI_HEADS_LOSS_CALCULATION = 512

    # Calculating the number of images in the training set to determine epoch length
    image_count = len(iterator.labeled_image_list)
    print(f"Expected count: {image_count}", flush=True)

    # Registering the dataset
    DatasetCatalog.register("supermarket_train", lambda: get_combined_dict(["../supermarket_dataset/annotations/annotations/D2S_training.json", "../supermarket_dataset/annotations/annotations/D2S_augmented.json"], iterator.labeled_image_list))
    actual_count = len(DatasetCatalog.get("supermarket_train"))
    print(f"actual count: {actual_count}", flush=True)
    assert image_count == actual_count, f"Length of loaded dataset is not as expected, actual {actual_count}, expected {image_count}"
    # Registering the class list
    list_classes = class_list("../supermarket_dataset/annotations/annotations/D2S_training.json")
    MetadataCatalog.get("supermarket_train").set(thing_classes=list_classes)
    supermarket_metadata = MetadataCatalog.get("supermarket_train")


    #Setting up the config file
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

        # Load trained weights
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  

    # Register the test set
    DatasetCatalog.register("supermarket_test", lambda: get_supermarket_dicts("../supermarket_dataset/alteredannotations/D2S_test_newsplit.json", False))
    list_classes = class_list("../supermarket_dataset/alteredannotations/D2S_test_newsplit.json")
    MetadataCatalog.get("supermarket_test").set(thing_classes=list_classes)


    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set a custom testing threshold
    predictor = DefaultPredictor(cfg)
    evaluator = COCOEvaluator("supermarket_test", cfg, False, output_dir=outputdir)
    val_loader = build_detection_test_loader(cfg, "supermarket_test")
    outcome = inference_on_dataset(predictor.model, val_loader, evaluator)
    with open(os.path.join(outputdir, f"{actual_count}iteration_mAP_treshold0_5.json"), 'w') as f:
        json.dump(outcome, f)
    

# Main loop
if __name__ == '__main__':
    EXPERIMENTNAME = "randomsubset2"
    outputdir = "output" + EXPERIMENTNAME
    os.makedirs(outputdir, exist_ok=True)

    # Preparing the iteration manager with random choice
    iterator = IterationManager(5000, 250, 250) #Starts with 100 random images in Dl, each iteration transfers 100 from Du to Dl up to a maximum of 2800 images
    iterator.loadcocolike(["../supermarket_dataset/annotations/annotations/D2S_training.json", "../supermarket_dataset/annotations/annotations/D2S_augmented.json"]) # load hte imagenumbers of the 2 datasets into memory
    iterator.initialsplit() #splits the imagelist from loadcocolike into a Du anda Dl list
    print(f"Length of Dl = {len(iterator.labeled_image_list)} ; Length of Du = {len(iterator.unlabeled_image_list)} ") #Length of labeled image list and unlabeled image list

    while iterator.keep_iterating:
        p = multiprocessing.Process(target=create_model_and_train, args=[iterator, outputdir])
        p.start()
        p.join()
        
        iterator.iteration() # transfer 100 images from Du to Dl through random choice (RANDOM)
        print(f"Length of Dl = {len(iterator.labeled_image_list)} ; Length of Du = {len(iterator.unlabeled_image_list)} ") # Print new lengths of labeled and unlabeled image list for confirmation



