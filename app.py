import os
import time
import gradio as gr
import PIL.Image as Image
import numpy as np
import cv2
from ultralytics import YOLO

# Load the YOLO model
model = YOLO("./models/best.pt")

def predict_image(img, conf_threshold, iou_threshold, image_size):
    """Predicts objects in an image using a YOLOv8 model with adjustable thresholds."""
    results = model.predict(
        source=img,
        conf=conf_threshold,
        iou=iou_threshold,
        show_labels=True,
        show_conf=True,
        imgsz=image_size,
    )

    for r in results:
        im_array = r.plot()
        im = Image.fromarray(im_array[..., ::-1])

    return im

def predict_video(video_path, conf_threshold, iou_threshold, image_size):
    """Generator function that processes video frames and yields them in real-time."""
    try:
        # Status indicators
        frame_count = 0
        start_time = time.time()
        last_fps_update = start_time
        fps = 0
        
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            yield np.zeros((300, 400, 3), dtype=np.uint8)  # Return empty frame if video can't be opened
            return
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # Process frames
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            # Convert frame to RGB (YOLO expects RGB)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Run prediction on the frame
            results = model.predict(
                source=frame,
                conf=conf_threshold,
                iou=iou_threshold,
                show_labels=True,
                show_conf=True,
                imgsz=image_size,
            )
            
            # Get the annotated frame
            for r in results:
                annotated_frame = r.plot()
            
            # Calculate FPS every second
            current_time = time.time()
            if current_time - last_fps_update >= 1.0:  # Update FPS every second
                fps = frame_count / (current_time - start_time)
                last_fps_update = current_time
            
            # Add status info to the frame
            cv2.putText(
                annotated_frame,
                f"Frame: {frame_count}/{total_frames} | FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
            
            # Add progress indicator
            progress_percent = min(100, int((frame_count / total_frames) * 100)) if total_frames > 0 else 0
            cv2.rectangle(annotated_frame, (10, 50), (10 + int(3.8 * progress_percent), 70), (0, 255, 0), -1)
            cv2.rectangle(annotated_frame, (10, 50), (390, 70), (255, 255, 255), 2)
            
            # Yield the processed frame
            yield annotated_frame
            
            # Small delay to allow UI to update
            time.sleep(0.01)
            
    finally:
        # Ensure resources are released
        if 'cap' in locals():
            cap.release()

def process_webcam(video_input, conf_threshold, iou_threshold, image_size):
    """Processes frames from webcam input in real-time."""
    if video_input is None:
        return None, 0.0
        
    # For Gradio webcam input, we get a numpy array directly
    # No need to use OpenCV to read the frame
    
    start_time = time.time()
    
    # Run prediction on the frame
    frame_rgb = video_input  # Gradio gives us RGB format already
    
    results = model.predict(
        source=frame_rgb,
        conf=conf_threshold,
        iou=iou_threshold,
        show_labels=True,
        show_conf=True,
        imgsz=image_size,
    )
    
    # Get the annotated frame
    for r in results:
        annotated_frame = r.plot()
    
    # Calculate processing FPS
    elapsed_time = time.time() - start_time
    fps = 1 / elapsed_time if elapsed_time > 0 else 0
    
    # Add parameter info to the frame
    cv2.putText(
        annotated_frame, 
        f"FPS: {fps:.1f} | Conf: {conf_threshold:.2f} | IoU: {iou_threshold:.2f}", 
        (10, 30), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        0.7, 
        (0, 255, 0), 
        2
    )
    
    cv2.putText(
        annotated_frame, 
        f"Image Size: {image_size}px", 
        (10, 60), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        0.7, 
        (0, 255, 0), 
        2
    )
    
    # Return both the frame and the calculated FPS
    return annotated_frame, fps

# Example images for static image detection
example_list = [["input/" + example] for example in os.listdir("input")]

# Create tabs for different input types
def create_demo():
    with gr.Blocks(title="Real-Time AI Detect System") as demo:
        gr.Markdown("""# Real-Time AI Detect System""")
        
        with gr.Tabs():
            with gr.Tab("Static Image"):
                with gr.Row():
                    with gr.Column():
                        image_input = gr.Image(type="pil", label="Upload Image")
                        conf_slider = gr.Slider(minimum=0, maximum=1, value=0.02, label="Confidence threshold")
                        iou_slider = gr.Slider(minimum=0, maximum=1, value=0.3, label="IoU threshold")
                        img_size_slider = gr.Slider(
                            label="Image Size",
                            minimum=320,
                            maximum=1280,
                            step=32,
                            value=512,
                        )
                        image_button = gr.Button("Run Inference")
                    
                    with gr.Column():
                        image_output = gr.Image(type="pil", label="Result")
                
                gr.Examples(
                    examples=[
                        [example_list[0][0], 0.25, 0.45, 640],
                        [example_list[1][0], 0.25, 0.45, 960],
                        [example_list[2][0], 0.25, 0.45, 640],
                    ],
                    inputs=[image_input, conf_slider, iou_slider, img_size_slider],
                    outputs=image_output,
                )
                
                image_button.click(
                    fn=predict_image,
                    inputs=[image_input, conf_slider, iou_slider, img_size_slider],
                    outputs=image_output,
                )
                
            with gr.Tab("Video Upload"):
                with gr.Row():
                    with gr.Column():
                        video_input = gr.Video(label="Upload Video")
                        video_conf_slider = gr.Slider(minimum=0, maximum=1, value=0.25, label="Confidence threshold")
                        video_iou_slider = gr.Slider(minimum=0, maximum=1, value=0.45, label="IoU threshold")
                        video_img_size_slider = gr.Slider(
                            label="Image Size",
                            minimum=320,
                            maximum=1280,
                            step=32,
                            value=640,
                        )
                        video_button = gr.Button("Process Video")
                    
                    with gr.Column():
                        # Changed from Video to Image for real-time display of frames
                        video_output = gr.Image(label="Real-time Processing")
                        
                        # Add status indicator
                        with gr.Row():
                            status_indicator = gr.Textbox(label="Status", value="Ready", interactive=False)
                            fps_indicator = gr.Number(label="FPS", value=0, interactive=False)
                
                video_button.click(
                    fn=predict_video,
                    inputs=[video_input, video_conf_slider, video_iou_slider, video_img_size_slider],
                    outputs=video_output,
                    # For real-time outputs, we need to indicate this is a generator function
                    api_name="process_video_realtime",
                    scroll_to_output=True
                )
            
            with gr.Tab("RTSP (Real-time)"):
                with gr.Row():
                    with gr.Column(scale=1):
                        webcam_conf_slider = gr.Slider(minimum=0, maximum=1, value=0.25, label="Confidence threshold", interactive=True)
                        webcam_iou_slider = gr.Slider(minimum=0, maximum=1, value=0.45, label="IoU threshold", interactive=True)
                        webcam_img_size_slider = gr.Slider(
                            label="Image Size",
                            minimum=320,
                            maximum=1280,
                            step=32,
                            value=640,
                            interactive=True,
                        )
                        
                        # Add FPS indicator
                        fps_indicator = gr.Number(label="Processing FPS", value=0, interactive=False)
                    
                    with gr.Column(scale=2):
                        # Use gr.Image with webcam source instead of the deprecated Webcam component
                        webcam = gr.Image(sources=["webcam"], streaming=True, label="Live Feed", type="numpy")
                        webcam_output = gr.Image(label="Detection Results")
                
                # For real-time processing, we use streaming mode
                webcam.stream(
                    fn=process_webcam,
                    inputs=[webcam, webcam_conf_slider, webcam_iou_slider, webcam_img_size_slider],
                    outputs=[webcam_output, fps_indicator],
                    show_progress=False,
                    stream_every=0.1  # Capture a frame every 0.1 seconds
                )
                
                # Note: Real-time slider updates via JavaScript not supported in this Gradio version
                # For newer versions, you could use _js parameter for real-time updates
        
        return demo

# Create the demo interface
demo = create_demo()

if __name__ == "__main__":
    # For Hugging Face Spaces compatibility
    demo.launch()