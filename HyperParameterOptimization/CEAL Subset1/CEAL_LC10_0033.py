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
from matplotlib import image as im
from matplotlib import pyplot as plt
import torch, torchvision
from scipy.stats import entropy
from multiprocessing import set_start_method


##########################################


class IterationManager():
    def __init__(self, maximages, startamount, iterationamount, initial_treshold =  0.10, initial_decayrate = 0.0033) -> None:
        self.image_list = []
        self.unlabeled_image_list = []
        self.labeled_image_list = []
        self.keep_iterating = True
        self.max_images = maximages
        self.startamount = startamount
        self.iterationamount = iterationamount

        # CEAL additions
        self.pseudlabelnumber_list = []
        self.timeperiod = 0
        self.treshold = initial_treshold
        self.decayrate = initial_decayrate
        
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

    def leastconfidentiteration(self, sorted_list):
        nr_images = self.iterationamount
        
        if nr_images > len(self.unlabeled_image_list):
            nr_images = len(self.unlabeled_image_list)

        current_length = len(self.labeled_image_list)
            
        print(f"adding {len(sorted_list[:nr_images])} least confident images to dl")
        for imagecombi in sorted_list[:nr_images]:
            self.labeled_image_list.append(imagecombi[0])   
        self.unlabeled_image_list = [imagenr for imagenr in self.image_list if imagenr not in self.labeled_image_list]

        assert len(self.labeled_image_list) == current_length + nr_images  

        if len(self.labeled_image_list) > self.max_images:
            self.keep_iterating = False 

    def cealiteration(self, sorted_list, sorted_entropy_list):   
        nr_images = self.iterationamount
        
        if nr_images > len(self.unlabeled_image_list):
            nr_images = len(self.unlabeled_image_list)

        current_length = len(self.labeled_image_list)
            
        print(f"adding {len(sorted_list[:nr_images])} least confident images to dl")
        for imagecombi in sorted_list[:nr_images]:
            self.labeled_image_list.append(imagecombi[0])   
        self.unlabeled_image_list = [imagenr for imagenr in self.image_list if imagenr not in self.labeled_image_list]

        assert len(self.labeled_image_list) == current_length + nr_images  

        if len(self.labeled_image_list) > self.max_images:
            self.keep_iterating = False 

        #Formula 9 from ceal paper
        self.pseudlabelnumber_list = []
        if self.timeperiod==0:
            thresholdvalue = self.treshold
        else: 
            thresholdvalue = self.treshold - (self.timeperiod * self.decayrate)
        print(f"current treshold value = {thresholdvalue}")

        # Selection of pseudo labels
        for imagecombi in sorted_entropy_list:
            if imagecombi[1] < thresholdvalue:
                print(f"image {imagecombi[0]} has an entropy value of {imagecombi[1]} which is above tresholdvalue {thresholdvalue}", flush=True)
                self.pseudlabelnumber_list.append(imagecombi[0])
        nrpseudolabels = len(self.pseudlabelnumber_list)
        print(f"length of psuedolabellist {nrpseudolabels}, pseudlabels for images {self.pseudlabelnumber_list}")
        # Increase timeperiod by 1 for next treshold value calculation
        self.timeperiod += 1


# Creating a list of categories to register to detectron2
def class_list(annotation_file_path):
    with open(annotation_file_path) as f:
        imgs_anns = json.load(f)
    cat_list = []
    for category in imgs_anns["categories"]:
        cat_list.append(category["name"])
    return cat_list

def get_supermarket_dicts(annotation_file_path, image_list=False, psuedolabelfixurl = False):
    json_file = annotation_file_path
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    
    for image_record in imgs_anns["images"]:
        image_id = image_record['id']
        if not image_list or image_id in image_list:
            record = {}
            if not psuedolabelfixurl:
                file_name = os.path.join("../supermarket_dataset/images/", image_record["file_name"])
            else:
                file_name = image_record["file_name"]

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
    
def get_combined_dict(listofURLs, image_list = False, pseudoURL = False, pseudoimagelist = False):
    final_list = []
    for datasetURL in listofURLs:
        final_list = final_list + get_supermarket_dicts(datasetURL, image_list)
    
    if pseudoURL and pseudoimagelist:
        print("pseudoURL and imagelist exists, loading pseudolabels into memory")
        print(pseudoURL)
        final_list = final_list + get_supermarket_dicts(pseudoURL, pseudoimagelist, psuedolabelfixurl=True)

    return final_list

def entropycalc(alloutputs):
    priority_list = []
    for imageresults in alloutputs:
        if not imageresults["all_scores"]:
            entrop = 999
        else:
            entrop = np.max(entropy(imageresults["all_scores"], axis=1)) # Calculates the entropy for each detected instance in the image, then selects the max entropy as the image entropy
        priority_list.append([imageresults["id"], entrop])
    priority_list.sort(key= lambda x: x[1], reverse=True)
    return priority_list

def leastconfident(outputs):
    priority_list = []
    for imageresult in outputs:
        if not imageresult["scores"]:
            leastconfscore = 0
        else: 
            leastconfscore = min(imageresult["scores"])
        imageid = imageresult["id"]
        priority_list.append([imageid, leastconfscore])
    priority_list.sort(key=lambda x: x[1])
    return priority_list

def combined_calc(outputs):
    leastconfidentscores = leastconfident(outputs)
    entropyscores = entropycalc(outputs)
    return [leastconfidentscores, entropyscores]

#########################################################################

def create_model_and_train(iterator, outputdir, queue):
    # Calculating the number of images in the training set to determine epoch length
    image_count = len(iterator.labeled_image_list)
    print(f"Image count in training set: {image_count}")

    # CONSTANTS
    EPOCHS = 50
    LEARNING_RATE = 0.001
    ROI_HEADS_LOSS_CALCULATION = 512

    # Calculating the number of images in the training set to determine epoch length
    image_count = len(iterator.labeled_image_list)
    print(f"Expected count: {image_count}", flush=True)

    # ensuring not first iteration
    if iterator.timeperiod == 0:
        pseudolabelurl = False
        pseudoimagenumbers = False
    else:
        pseudolabelurl = f"{outputdir}/supermarket_unlabled_coco_format.json"
        pseudoimagenumbers = iterator.pseudlabelnumber_list
        print(f"pseudo image number list = {pseudoimagenumbers}", flush=True)

    # Registering the dataset
    DatasetCatalog.register("supermarket_train", lambda: get_combined_dict(["../supermarket_dataset/alteredannotations/D2S_training_newsplit.json"], iterator.labeled_image_list, pseudolabelurl, pseudoimagenumbers))
    actual_count = len(DatasetCatalog.get("supermarket_train"))
    print(f"actual count: {actual_count}", flush=True)
    #assert image_count == actual_count, f"Length of loaded dataset is not as expected, actual {actual_count}, expected {image_count}"
    # Registering the class list
    list_classes = class_list("../supermarket_dataset/alteredannotations/D2S_training_newsplit.json")
    MetadataCatalog.get("supermarket_train").set(thing_classes=list_classes)
    supermarket_metadata = MetadataCatalog.get("supermarket_train")

    step_1 = int((actual_count/2) * EPOCHS * 0.75)
    LEARNING_RATE_DECAY_STEP = (step_1,) #empty list if no decay, otherwise an iteratble with step numberw here to decay weight eg. (3000, ) will decay the weight at the 3000th iteration (with 2 images per iteration this is at image 6000)

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
    cfg.SOLVER.MAX_ITER = int((actual_count/2) * EPOCHS)    
    #cfg.SOLVER.MAX_ITER = 100   
    cfg.SOLVER.STEPS = LEARNING_RATE_DECAY_STEP        
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = ROI_HEADS_LOSS_CALCULATION   
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 60 
    cfg.INPUT.MASK_FORMAT = "bitmask" # note to self: bitmask here because the mvtec annotations uses runlength encoding instead of keypoints
    cfg.OUTPUT_DIR = outputdir

    trainer = DefaultTrainer(cfg) 
    trainer.resume_or_load(resume=False)
    trainer.train()

    #del trainer
    torch.cuda.empty_cache()

        # Load trained weights
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  

    # Register the test set
    DatasetCatalog.register("supermarket_test", lambda: get_supermarket_dicts("../supermarket_dataset/alteredannotations/D2S_validation_newsplit.json", False))
    list_classes = class_list("../supermarket_dataset/alteredannotations/D2S_validation_newsplit.json")
    MetadataCatalog.get("supermarket_test").set(thing_classes=list_classes)

    # COCO evaluation of results
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set a custom testing threshold
    predictor = DefaultPredictor(cfg)
    evaluator = COCOEvaluator("supermarket_test", cfg, False, output_dir=outputdir)
    val_loader = build_detection_test_loader(cfg, "supermarket_test")
    outcome = inference_on_dataset(predictor.model, val_loader, evaluator)
    with open(os.path.join(outputdir, f"{image_count}iteration_mAP_treshold0_5.json"), 'w') as f:
        json.dump(outcome, f)
    
    #del val_loader
    #del evaluator
    #del outcome

    #### Inferring unlabeled data for least-confident sorting ###
    # Loading unlabeled dataset 
    DatasetCatalog.register("supermarket_unlabled", lambda: get_combined_dict(["../supermarket_dataset/alteredannotations/D2S_training_newsplit.json"], iterator.unlabeled_image_list))
    list_classes = class_list("../supermarket_dataset/alteredannotations/D2S_training_newsplit.json")
    MetadataCatalog.get("supermarket_unlabled").set(thing_classes=list_classes)
    # Loading image-data into a list for reading
    imagelist = DatasetCatalog.get("supermarket_unlabled")

    # Inferencing on unlabled images
    alloutputs = []
    
    for number, imageinfo in enumerate(imagelist):
        image = im.imread(imageinfo["file_name"])
        with torch.no_grad():
            outputs = predictor(image)
        image_id = imageinfo["image_id"]
        scores = outputs["instances"].scores.tolist()
        all_scores = outputs["instances"].scores_all.tolist()
        imgdict = {"id":image_id, "scores": scores, "all_scores": all_scores}
        #imgdict = {"id":image_id, "scores": scores}
        alloutputs.append(imgdict)

    outputs = combined_calc(alloutputs)
    queue.put(outputs)

    evaluator = COCOEvaluator("supermarket_unlabled", cfg, False, output_dir=outputdir)

    torch.cuda.empty_cache()



# Main loop
if __name__ == '__main__':
    set_start_method('spawn')
    EXPERIMENTNAME = "cealLC10_0033"
    outputdir = "output" + EXPERIMENTNAME
    os.makedirs(outputdir, exist_ok=True)

    queue = multiprocessing.Queue()

    # Preparing the iteration manager with random choice
    iterator = IterationManager(1000, 50, 50) #Starts with 100 random images in Dl, each iteration transfers 100 from Du to Dl up to a maximum of 2800 images
    iterator.loadcocolike(["../supermarket_dataset/alteredannotations/D2S_training_newsplit.json"]) # load hte imagenumbers of the 2 datasets into memory
    iterator.initialsplit() #splits the imagelist from loadcocolike into a Du anda Dl list
    print(f"Length of Dl = {len(iterator.labeled_image_list)} ; Length of Du = {len(iterator.unlabeled_image_list)} ") #Length of labeled image list and unlabeled image list

    while iterator.keep_iterating:
        p = multiprocessing.Process(target=create_model_and_train, args=[iterator, outputdir, queue])
        p.start()
        combinedlist = queue.get()
        p.join()

        print(f"Entropy List sorted {combinedlist}")
        
        iterator.cealiteration(combinedlist[0], combinedlist[1]) # transfer XX images from Du to Dl through random choice (RANDOM)
        print(f"Length of Dl = {len(iterator.labeled_image_list)} ; Length of Du = {len(iterator.unlabeled_image_list)} ") # Print new lengths of labeled and unlabeled image list for confirmation



