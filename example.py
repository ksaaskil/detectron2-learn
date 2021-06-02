import detectron2
from detectron2.engine.defaults import DefaultTrainer
from detectron2.utils.logger import setup_logger

import numpy as np
import os, json, cv2, random
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

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


from detectron2.structures import BoxMode


def get_balloon_dicts(img_dir):
    json_file = os.path.join(img_dir, "via_region_data.json")
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []

    for idx, v in enumerate(imgs_anns.values()):
        record = {}

        filename = os.path.join(img_dir, v["filename"])

        height, width = cv2.imread(filename).shape[:2]

        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width

        annos = v["regions"]

        objs = []

        for _, anno in annos.items():
            assert not anno["region_attributes"]
            anno = anno["shape_attributes"]

            px = anno["all_points_x"]
            py = anno["all_points_y"]

            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]

            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": 0,
            }
            objs.append(obj)

        record["annotations"] = objs
        dataset_dicts.append(record)

    return dataset_dicts


def prepare_balloon_metadata():
    logger.info("Registering dataset and catalog")
    for d in ["train", "val"]:
        DatasetCatalog.register(
            "balloon_" + d, lambda d=d: get_balloon_dicts("balloon/" + d)
        )
        MetadataCatalog.get("balloon_" + d).set(thing_classes=["balloon"])
    balloon_metadata = MetadataCatalog.get("balloon_train")
    logger.info("Balloon dataset metadata: %s", balloon_metadata)
    return balloon_metadata


def visualize_balloon_sample():
    balloon_metadata = MetadataCatalog.get("balloon_train")
    dataset_dicts = get_balloon_dicts("balloon/train")
    for d in random.sample(dataset_dicts, 1):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=balloon_metadata, scale=0.5)
        out = visualizer.draw_dataset_dict(d)
        cv2.imwrite("balloon.jpg", out.get_image()[:, :, ::-1])


def get_train_cfg():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(MODEL_CONFIG))
    cfg.DATASETS.TRAIN = ("balloon_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(MODEL_CONFIG)
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 1  # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.SOLVER.STEPS = []  # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (
        128  # faster, and good enough for this toy dataset (default: 512)
    )
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    cfg.MODEL.DEVICE = DEVICE
    # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
    return cfg


def train():
    balloon_metadata = prepare_balloon_metadata()
    visualize_balloon_sample()

    cfg = get_train_cfg()

    logger.info(f"Output dir: {cfg.OUTPUT_DIR}")

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()


if __name__ == "__main__":
    # predict()
    train()
