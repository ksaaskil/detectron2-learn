## Instructions

Install dependencies including PyTorch:

```bash
$ pip install -r requirements.txt
```

Alternatively, [install PyTorch](https://pytorch.org/get-started/locally/) as required for your use case.

Install Detectron2:

```bash
$ pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

or alternatively from source:

```bash
$ git clone https://github.com/facebookresearch/detectron2.git
$ pip install -e detectron2
```

## Predict

Fetch example image:

```bash
$ wget http://images.cocodataset.org/val2017/000000439715.jpg -q -O input.jpg
```

Run the example:

```bash
$ python example.py
```

## Train

```bash
$ wget https://github.com/matterport/Mask_RCNN/releases/download/v2.1/balloon_dataset.zip
$ unzip balloon_dataset.zip
```

```bash
$ python train.py
```

View in Tensorboard:

```bash
$ tensorboard --logdir output/train
```
