from detectron2.utils.logger import setup_logger

import cv2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

logger = setup_logger()

INPUT = "input.jpg"
OUTPUT = "out.jpg"

DEVICE = "cpu"
MODEL_CONFIG = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"


def predict():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(MODEL_CONFIG))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(MODEL_CONFIG)
    cfg.MODEL.DEVICE = DEVICE

    predictor = DefaultPredictor(cfg)

    im = cv2.imread(INPUT)
    outputs = predictor(im)

    outputs_instances = outputs["instances"]
    logger.info(f"Found {len(outputs_instances)} instances")

    dataset = cfg.DATASETS.TRAIN[0]
    logger.info("Dataset %s,", dataset)

    # Find catalog for visualization purposes (what are the classes etc.)
    catalog = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
    logger.info("Catalog %s", catalog)

    v = Visualizer(im[:, :, ::-1], catalog, scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"])

    logger.info("Writing to file: %s", OUTPUT)
    cv2.imwrite(OUTPUT, out.get_image()[:, :, ::-1])


if __name__ == "__main__":
    predict()
