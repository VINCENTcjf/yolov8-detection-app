import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import os
from PIL import Image
import io

# 加载 YOLOv8 模型
model = YOLO("best.pt")

# 图片预测函数
def predict_image(image):
    # 将 Streamlit 上传的图片转换为 OpenCV 格式
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # 运行 YOLOv8 预测
    results = model.predict(source=image, conf=0.1)
    
    # 绘制检测结果
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # 转换张量为数值
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())
            label = f"{result.names[cls_id]} {conf:.2f}"
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # 转换回 RGB 格式以供 Streamlit 显示
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

# 视频预测函数
def predict_video(video_file):
    # 保存上传的视频文件
    video_path = "temp_video.mp4"
    with open(video_path, "wb") as f:
        f.write(video_file.read())
    
    # 读取视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Error: Could not open video file.")
        return None
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # 创建临时输出视频路径（使用 mp4v 编码）
    temp_output_path = "temp_output_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))
    if not out.isOpened():
        st.error("Error: Could not create temporary output video file.")
        cap.release()
        return None
    
    # 逐帧处理
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 运行 YOLOv8 预测
        results = model.predict(source=frame, conf=0.1)
        
        # 绘制检测结果
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls.item())
                conf = float(box.conf.item())
                label = f"{result.names[cls_id]} {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        out.write(frame)
        frame_count += 1
    
    cap.release()
    out.release()
    
    # 调试：检查临时文件是否存在
    if not os.path.exists(temp_output_path):
        st.error("Error: Temporary output video file was not created.")
        return None
    
    # 使用 FFmpeg 转换为 H.264 编码
    output_path = "output_video.mp4"
    ffmpeg_cmd = f"ffmpeg -i {temp_output_path} -vcodec libx264 -acodec aac {output_path} -y"
    try:
        result = os.system(ffmpeg_cmd)
        if result != 0:
            st.error("Error: FFmpeg conversion failed.")
            return None
    except Exception as e:
        st.error(f"Error converting video with FFmpeg: {e}")
        return None
    
    # 清理临时文件
    if os.path.exists(temp_output_path):
        os.remove(temp_output_path)
    
    # 调试：检查最终输出文件是否存在
    if not os.path.exists(output_path):
        st.error("Error: Output video file was not created after FFmpeg conversion.")
        return None
    
    # 检查文件大小
    file_size = os.path.getsize(output_path) / (1024 * 1024)  # 转换为 MB
    st.write(f"Processed {frame_count} frames. Output video size: {file_size:.2f} MB")
    
    # 清理输入临时文件
    os.remove(video_path)
    return output_path

# GIF 动图预测函数
def predict_gif(gif_file):
    # 读取 GIF 文件
    gif = Image.open(gif_file)
    frames = []
    try:
        while True:
            # 将当前帧转换为 OpenCV 格式
            frame = np.array(gif.convert("RGB"))
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # 运行 YOLOv8 预测
            results = model.predict(source=frame, conf=0.1)
            
            # 绘制检测结果
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls_id = int(box.cls.item())
                    conf = float(box.conf.item())
                    label = f"{result.names[cls_id]} {conf:.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # 转换回 RGB 格式
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))
            gif.seek(gif.tell() + 1)
    except EOFError:
        pass
    
    # 保存处理后的 GIF
    output_gif_path = "output_gif.gif"
    frames[0].save(
        output_gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=gif.info.get("duration", 100),
        loop=0
    )
    
    return output_gif_path

# Streamlit 界面
st.title("YOLOv8 Custom Detection App")
st.markdown("Upload an image, video, or GIF to detect objects: `paper-wood`, `glass`, `dead_fish`, `cloth`, `cigarette_butts`, `plastic`, `metal`.")

# 选项卡：图片检测、视频检测、GIF 检测
tab1, tab2, tab3 = st.tabs(["Image Detection", "Video Detection", "GIF Detection"])

# 图片检测
with tab1:
    st.header("Image Detection")
    image_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key="image")
    if image_file is not None:
        image = np.array(bytearray(image_file.read()), dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        if st.button("Detect Image"):
            with st.spinner("Detecting..."):
                result_image = predict_image(image)
                st.image(result_image, caption="Detected Image", use_column_width=True)

# 视频检测
with tab2:
    st.header("Video Detection")
    video_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"], key="video")
    if video_file is not None:
        st.video(video_file)
        
        if st.button("Detect Video"):
            with st.spinner("Processing video..."):
                result_video_path = predict_video(video_file)
                if result_video_path:
                    try:
                        # 读取视频文件为字节流
                        with open(result_video_path, "rb") as f:
                            video_bytes = f.read()
                        st.video(video_bytes, format="video/mp4")
                        
                        # 提供下载选项
                        st.download_button(
                            label="Download Processed Video",
                            data=video_bytes,
                            file_name="processed_video.mp4",
                            mime="video/mp4"
                        )
                    except Exception as e:
                        st.error(f"Error displaying video: {e}")
                    finally:
                        # 清理输出视频文件
                        if os.path.exists(result_video_path):
                            os.remove(result_video_path)
                else:
                    st.error("Video processing failed.")

# GIF 检测
with tab3:
    st.header("GIF Detection")
    gif_file = st.file_uploader("Upload a GIF", type=["gif"], key="gif")
    if gif_file is not None:
        st.image(gif_file, caption="Uploaded GIF", use_column_width=True)
        
        if st.button("Detect GIF"):
            with st.spinner("Processing GIF..."):
                result_gif_path = predict_gif(gif_file)
                if result_gif_path:
                    try:
                        st.image(result_gif_path, caption="Detected GIF", use_column_width=True)
                        # 提供下载选项
                        with open(result_gif_path, "rb") as f:
                            gif_bytes = f.read()
                        st.download_button(
                            label="Download Processed GIF",
                            data=gif_bytes,
                            file_name="processed_gif.gif",
                            mime="image/gif"
                        )
                    except Exception as e:
                        st.error(f"Error displaying GIF: {e}")
                    finally:
                        # 清理输出 GIF 文件
                        if os.path.exists(result_gif_path):
                            os.remove(result_gif_path)
                else:
                    st.error("GIF processing failed.")
