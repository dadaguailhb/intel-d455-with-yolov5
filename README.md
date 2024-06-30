## 定位
demo已调通，为后续开发作准备
## 设备
Jetson agx orin 64g，intel-realsense d455
## 环境配置

首先为jetson设备安装torch，之后按照yolov5的环境要求配置环境
Clone repo and install requirements.txt in a Python>=3.8.0 environment, including PyTorch>=1.8.
```
cd yolov5
pip install -r requirements.txt  # install
```

之后安装reslsense包

## 运行demo
```
python main_pipeline.py
```

会把当前处理好的图片保存到当前目录，文件名为detection_output.png
## 参考
[Realsense D435i 通过YOLOv5、YOLOv8输出目标三维坐标](https://blog.csdn.net/Zeng999212/article/details/133926142?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_baidulandingword~default-1-133926142-blog-135587498.235^v43^pc_blog_bottom_relevance_base9&spm=1001.2101.3001.4242.2&utm_relevant_index=4)