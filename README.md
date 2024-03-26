#云检测系统界面设计
## 1. 简介

该项目是一个用于展示 **遥感影像云检测（二分类图像分割任务）** 的系统界面，提供了用户友好的交互模式，使用户能上传图片、查看检测结果并对检测结果进行人工修正。

## 2. 安装与运行
> **特别提示：** 该项目仅包含推理过程，不包含训练过程（**建议采用mmsegmentation代码库进行训练**）。请将训练的配置文件放入 `./configs` 文件夹中，并将训练好的权重放入 `./checkpoints` 文件夹中。  

### 2.1. 安装  
[参考MMsegmentation代码库安装步骤进行安装](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/zh_cn/get_started.md)  
**步骤1** 创建一个 conda 环境，并激活
```bash
conda create --name cloudsystem python=3.8 -y
conda activate cloudsystem
```

**步骤2** 安装 PyTorch  
在GPU平台上：
```bash
conda install pytorch=1.11.0 torchvision=0.12.0 -c pytorch
```
在CPU平台上：
```bash
conda install pytorch=1.11.0 torchvision=0.12.0 cpuonly -c pytorch
```

**步骤3** 使用 MIM 安装 MMCV
```bash
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
```

**步骤4** 安装 MMSegmentation
```bash
pip install "mmsegmentation>=1.0.0"
```

**步骤5** 安装其他依赖包
```bash
cd CloudDetection_System
pip install -r requirement.txt
```

###2.2. 运行

```bash
python cloud_UI.py ./configs/segformer/segformer_mit-b2_8xb2-160k_ade20k-512x512_cloud.py ./checkpoints/segformer_b2_best_mIoU_iter_90000.pth
```
可以用自己的配置文件和权重文件进行替换。

## 3. 功能介绍
###3.1. 上传图像  
上传图像有两种方式，一种是通过点击**上传**按钮选择图像路径进行上传，另一种则是直接在上传按钮右侧的**输入框**中输入路径后按Enter键即可上传。  
上传之后会显示图像、上传路径和图像大小。
<div style="display: flex; flex-wrap: nowrap;">
    <img src="./picture/img_1.png" alt="Image" style="width: 600px; height: auto;">
</div>

###3.2. 查看检测结果 
点击**开始检测**按钮即可在右侧的检测结果位置得到推理出的分割结果，以及推理时间和含云量。
<div style="display: flex; flex-wrap: nowrap;">
    <img src="./picture/img_2.png" alt="Image" style="width: 600px; height: auto;">
</div>

###3.3. 对检测结果进行人工修正  
**双击**右侧的检测结果，即可跳转到人工修正的子窗口。**拉动下方的滑块**，可以调整推理结果在原图上的覆盖比例，方便更好地和原图进行对比来进行修正。  
<div style="display: flex; flex-wrap: nowrap;">
    <img src="./picture/img_3.png" alt="Image" style="width: 280px; height: auto;">
    <img src="./picture/img_4.png" alt="Image" style="width: 280px; height: auto; margin-left: 40px;">
</div>

点击上方选择画笔大小以及添加去除来调整画笔，用鼠标在图片上拖动即可进行修正。
<div style="display: flex; flex-wrap: nowrap;">
    <img src="./picture/img_5.png" alt="Image" style="width: 280px; height: auto;">
</div>

点击**保存画布**按钮，就可以将修正后的结果显示在主窗口界面上，含云量也会相应发生改变。
<div style="display: flex; flex-wrap: nowrap;">
    <img src="./picture/img_6.png" alt="Image" style="width: 600px; height: auto;">
</div>

###3.4. 保存结果  
拖动检测结果下方发滑块，可以选择保存的覆盖比例，点击**保存**按钮即可对结果进行保存，默认路径是图片的上传路径。
<div style="display: flex; flex-wrap: nowrap;">
    <img src="./picture/img_7.png" alt="Image" style="width: 600px; height: auto;">
</div>

## 4. 开源许可证

该项目采用 [Creative Commons 许可证 - CC BY-NC 4.0](LICENSE.md) 进行许可。详情请参阅许可证文件。

## 5. 联系我

如果您有任何疑问或建议，请通过[1192534463@qq.com](mailto:1192534463@qq.com)与我联系。

