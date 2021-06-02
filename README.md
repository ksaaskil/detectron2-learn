## Instructions

Install PyTorch:

```bash
$ pip install torch torchvision
```

Install Detectron2:

```bash
$ pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

or alternatively from source:

```bash
$ git clone https://github.com/facebookresearch/detectron2.git
$ pip install -e detectron2
```

Verify Detectron2 works:

```bash
$ cd detectron2/demo
$ python demo.py --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml \
  --video-input LeftBag.mp4 \
  --opts MODEL.WEIGHTS detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl MODEL.DEVICE cpu
```
