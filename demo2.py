import detectron2
from detectron2.utils.logger import setup_logger

import numpy as np
import os, json, cv2, random
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

logger = setup_logger()

INPUT = "IMG_8950.jpg"


def run():
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        )
    )
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    )
    cfg.MODEL.DEVICE = "cpu"
    predictor = DefaultPredictor(cfg)

    im = cv2.imread(INPUT)
    outputs = predictor(im)
    outputs_instances = outputs["instances"]
    logger.info(f"Found {len(outputs_instances)} instances")

    dataset = cfg.DATASETS.TRAIN[0]
    logger.info("Dataset %s,", dataset)

    catalog = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
    logger.info("Catalog %s", catalog)
    v = Visualizer(im[:, :, ::-1], catalog, scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"])
    OUTPUT = "out.jpg"

    cv2.imwrite(OUTPUT, out.get_image()[:, :, ::-1])


if __name__ == "__main__":
    run()
