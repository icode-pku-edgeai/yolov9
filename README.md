# 训练相关基本操作
## 环境
+ python
+ torch
+ 其他
```
pip install -r requirements.txt -i  https://pypi.tuna.tsinghua.edu.cn/simple

```
## 数据集
+ 数据格式与yolov5一致
```
├── images
│   ├── train
│   └── val
└── labels
    ├── train
    └── val

```
## 命令行执行
### 训练必用,建议使用train_dual.py
+ --weights 初始化权重pt
+ --cfg 初始化模型配置yaml
+ --data 数据集加载，默认data文件夹下找yaml文件
+ --hyp 超参，默认data/hyps超参下找yaml
+ --epochs 训练代数
+ --batch-size 视显存大小而定
+ --img 训练尺度
+ --resume 断点重训
+ --device gpu数量
+ --workers 0肯定可以，其他数值请自行尝试
+ --patience epochs设置比较大的时候可以用到early-stop
+ --close-mosaic 马赛克数据增强提前关闭

### 验证必用，建议使用val_dual.py
+ --data 数据集加载，默认data文件夹下找yaml文件
+ --weights 想要验证的权重文件地址，支持pt、torchscript、onnx、openvino、trt、tf、paddle
+ --batch-size 视验证目标而定
+ --img 验证尺度
+ --conf-thres 置信度阈值，影响较大
+ --iou-thres iou阈值
+ --max-det 最大检测数量
+ --task 主要用val、test、speed
+ --device gpu数量
+ --workers 0肯定可以，其他数值请自行尝试
+ --half fp16推理

### 推理必用，建议使用detect_dual.py
+ --weights 想要推理的权重文件地址，支持pt、torchscript、onnx、openvino、trt、tf、paddle
+ --source 想要推理的目录，可以是图片、视频、文件夹、屏幕、摄像头
+ --img 推理尺度
+ --conf-thres 置信度阈值，影响较大
+ --iou-thres iou阈值
+ --max-det 最大检测数量
+ --device gpu数量
+ --nosave 不保存

### 导出必用export.py
+ --weights 想要导出的权重地址
+ --img 导出尺度
+ --batch-size 导出尺度
+ --device gpu数量
+ --half 半精度导出
+ --dynamic 动态导出
+ --simplify 调用onnx-simplify进行简化
+ --opset onnx版本
+ --include 导出格式，主要有onnx，torchscript，engine等等

### 全面验证benchmarks.py
+ torch、torchscript、onnx、openvino、trt、coreml、tf等等全部跑一遍
+ --weights 想要导出的权重地址
+ --img 导出尺度
+ --batch-size 导出尺度
+ --device gpu数量
+ --half 半精度导出
+ --pt-only 仅测试pt

# 代码基础介绍
## classify分类代码
## data 超参+数据集
## models 核心代码
### detect 主要模型yaml文件
### hub 其他模型yaml
### panoptic 全景分割模型
### segment 分割模型
### common.py 核心子模块
### experimental.py 实验模块
### tf.py tensorflow代码
### yolo.py 子模块调用
## panoptic 全景调用代码
## segment 分割调用代码
## utils 各类工具
+ activations.py 激活函数
+ augmentations.py 数据增强
+ autoanchor.py 自动计算anchor值
+ autobatch.py 自动生成batch
+ callbacks.py 处理回调函数
+ coco_utils.py coco数据集工具
+ dataloaders.py 数据加载工具
+ downloads.py 下载工具
+ general.py 大量常规工具
+ lion.py 狮群优化算法
+ loss.py损失函数及多级辅助监督的损失函数
+ metrics.py 评价尺度有关工具，例如ap计算
+ plots.py绘图工具
+ torch_utils torch工具
+ triton.py triton推理工具
## benchmarks.py 全平台验证
## detect.py detect_dual.py 单头和多头检测代码
## export.py 导出代码
## hubconf.py torch hub模型库
## train.py train_dual.py train_triple.py 单头和多头训练代码
## val.py val_dual.py val_triple.py单头和多头验证代码

# 核心优势
## PGI可编程梯度信息
+ 辅助监督框架，也就是train、val、detect对应的dual和triple等代码，对应也提出了多头的损失函数
+ yolov7的ELAN优化梯度路径，降参提点，变成GELAN，两个版本是同一个作者


