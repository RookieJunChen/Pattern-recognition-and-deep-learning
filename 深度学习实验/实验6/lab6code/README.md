# Pytorch-YOLOv4

![](https://img.shields.io/static/v1?label=python&message=3.6|3.7&color=blue)
![](https://img.shields.io/static/v1?label=pytorch&message=1.4&color=<COLOR>)
[![](https://img.shields.io/static/v1?label=license&message=Apache2&color=green)](./License.txt)

```
│  .gitignore
│  demo.py	# 运行示例
│  environment.yml  # 环境配置
│  License.txt
│  logs  # 日志文件夹
│  models  # 模型存取文件夹
│  README.md
│  requirements.txt
│  tensor_logs  # tensorboard文件夹
│  tree.txt
│  __init__.py
│  
├─cfg  # 模型参数配置，*表示实验使用
│      cfg.py *
│      yolov3-tiny.cfg
│      yolov3.cfg
│      yolov4-custom.cfg
│      yolov4.cfg *
│      
├─data  # 实验结果
│  │  coco.names  # coco类别名
│  │  coco_val_outputs.json  # validation 结果，用于cocoapi
│  │  dog.jpg  # 结果示例
│  │  giraffe.jpg
│  │  prediction.jpg
│  │  voc.names
│  │  
│  └─outcome  # 实验结果示例
│          predictions_000000000139.jpg
 	. . . . . .
│          
├─Dataset # 模型数据集
│  │  backbone_dataset.py  # 用于训练resnet backbone
│  │  dataset.py  # 用于训练yolov4
│  │  transform.py  # 数据增强
│  │  __init__.py
│  │  
│  └─COCO  # COCO数据集存放位置
│      │  annotations  # 标注
│      │  images  # 图片
│      │  
│      └─bboxes # bboxes格式，与原COCO数据集格式不相同，应使用下面两个文件
│              train.txt
│              val.txt
│              
├─model  # 训练模型
│      Backbone.py  # Resnet模型
│      models.py  # yolov4模型
│      SENet.py  # SE 模块
│      
├─scripts  # 训练及测试脚本
│      eval_coco.sh  # 测试yolov4
│      train_backbone.sh  # 训练resnet backbone
│      train_coco.sh  # 训练yolov4
│      
├─tool  # 常用工具
│      box.py  # 转换bbox格式
│      camera.py  # 使用camera
│      coco_annotation.py
│      config.py
│      darknet2onnx.py
│      darknet2pytorch.py  # 模型格式转换
│      onnx2tensorflow.py
│      region_loss.py
│      torch_utils.py
│      utils.py
│      yolo_layer.py
│      __init__.py
│      
└─utils
        evaluate_on_coco.py  # 测试yolov4
        train.py  # 训练yolov4
        train_resnet.py  # 训练resnet backbone
```



# 0. 环境配置

1. anaconda 环境

```shell
conda env create -f environment.yml
```

2. 下载预训练模型及训练日志

预训练百度云链接为：链接：https://pan.baidu.com/s/1ay6wR3ej_4PW2x5SPvC40A 
提取码：o16u

训练日志百度云链接为：链接：https://pan.baidu.com/s/1JkntOOT3kyFDCUItJxGBkg 
提取码：lwpf

将这两个文件下载后解压缩到主目录下，如上图代码结构图所示

# 1.Backbone 训练

```shell
sh scripts/train_backbone.sh
文件路径与数据集路径可自主调整
```

- resnet系列backbone为自主训练

- darknet53直接使用预训练模型，文件位置为./models/yolov4.conv.137

# 2. Yolov4训练

```shell
sh scripts/train_coco.sh
数据集路径和文件路径可自主调整，但bbox文件应为Dataset/COCO/bboxes，代码中已给出对应文件
```



# 3. Yolov4测试

```shell
sh scripts/eval_coco.sh
数据集路径和文件路径可自主调整，但bbox文件应为Dataset/COCO/bboxes，代码中已给出对应文件
```



```
@article{yolov4,
  title={YOLOv4: YOLOv4: Optimal Speed and Accuracy of Object Detection},
  author={Alexey Bochkovskiy, Chien-Yao Wang, Hong-Yuan Mark Liao},
  journal = {arXiv},
  year={2020}
}
```