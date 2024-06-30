import pyrealsense2 as rs
import W_detectAPI

if __name__ == '__main__':  # 入口
    pipeline = rs.pipeline()  # 定义流程pipeline，创建一个管道
    config = rs.config()  # 定义配置config

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 60)  # 配置depth流
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)  # 配置color流

    pipeline.start(config) 
    align = rs.align(rs.stream.color)

    a = W_detectAPI.DetectAPI(weights='yolov5s.pt')

    try:
        while True:
            # color_intrin, depth_intrin, img_color, img_depth, aligned_depth_frame = get_aligned_images()  # 获取对齐图像与相机参数

            frames = pipeline.wait_for_frames()  # 等待获取图像帧，获取颜色和深度的框架集
            aligned_frames = align.process(frames)  # 获取对齐帧，将深度框与颜色框对齐

            aligned_depth_frame = aligned_frames.get_depth_frame()  # 获取对齐帧中的的depth帧
            aligned_color_frame = aligned_frames.get_color_frame()  # 获取对齐帧中的的color帧

            print("successfull")

    finally:
        # Stop streaming
        pipeline.stop()