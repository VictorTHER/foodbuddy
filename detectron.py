import os
import json
import random
from collections import defaultdict
import numpy as np
import torch
from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader
import cv2

# 1. Prétraitement des données
def filtered_dataset():
    with open("./raw_data/public_training_set_release_2.0/annotations.json", "r") as f:
        data = json.load(f)

    image_annotation_count = defaultdict(int)
    for annotation in data["annotations"]:
        image_annotation_count[annotation["image_id"]] += 1

    filtered_image_ids = {image_id for image_id, count in image_annotation_count.items() if count in [1, 2]}
    filtered_images = [img for img in data["images"] if img["id"] in filtered_image_ids]
    filtered_annotations = [anno for anno in data["annotations"] if anno["image_id"] in filtered_image_ids]

    filtered_data = {
        "images": filtered_images,
        "annotations": filtered_annotations,
        "categories": data["categories"]
    }

    with open("./raw_data/public_training_set_release_2.0/filtered_annotations.json", "w") as f:
        json.dump(filtered_data, f)

filtered_dataset()

with open("./raw_data/public_training_set_release_2.0/filtered_annotations.json", "r") as f:
    annotations = json.load(f)

# Compter les catégories
num_categories = len(annotations["categories"])
print(f"Nombre de catégories dans le dataset : {num_categories}")

# 2. Enregistrement des datasets
register_coco_instances("food/train", {}, "./raw_data/public_training_set_release_2.0/filtered_annotations.json", "./raw_data/public_training_set_release_2.0/images")
register_coco_instances("food/val", {}, "./raw_data/public_validation_set_2.0/annotations.json", "./raw_data/public_validation_set_2.0/images")

train_metadata = MetadataCatalog.get('food/train')

# 3. Configuration du modèle
def config():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("food/train",)
    cfg.DATASETS.TEST = ("food/val",)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.SOLVER.BASE_LR = 0.025
    cfg.SOLVER.MAX_ITER = 300
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_categories
    cfg.OUTPUT_DIR = "./output"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    return cfg

# 4. Entraînement
def train_model():
    cfg = config()
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
    return cfg

# 5. Évaluation
def evaluate_model(cfg):
    evaluator = COCOEvaluator("food/val", cfg, False, output_dir="./output")
    val_loader = build_detection_test_loader(cfg, "food/val")
    print(inference_on_dataset(DefaultPredictor(cfg).model, val_loader, evaluator))

# 6. Visualisation
def visualize(cfg):
    predictor = DefaultPredictor(cfg)
    img = cv2.imread("../raw_data/public_training_set_release_2.0/images/000001.jpg")
    outputs = predictor(img)
    visualizer = Visualizer(img, MetadataCatalog.get("food/train"), scale=1.2)
    v = visualizer.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imshow("Predictions", v.get_image())
    cv2.waitKey(0)

print()
cfg = train_model()
evaluate_model(cfg)
visualize(cfg)
