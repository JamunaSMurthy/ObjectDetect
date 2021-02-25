"""
YOLO v1 Video Object Detection Inference Script

Performs real-time object detection on video files using pretrained YOLOv1 model.
Processes video frame-by-frame, detecting multiple object classes from BDD100K 
dataset, and outputs annotated video with bounding boxes and confidence scores.

Error Handling:
    - FileNotFoundError: Input video or model weights file missing
    - ValueError: Invalid argument parameters or corrupted model weights
    - RuntimeError: GPU initialization, model loading failure, or video codec issues
    - IOError: Failed to read/write video files corrupted frames

Features:
    - Frame-by-frame inference with progress tracking
    - Configurable output codec and FPS
    - Average FPS reporting across entire video
    - Robust error recovery for corrupted frames

Example:
    >>> python YOLO_to_video.py \\
    ...     --weights model_weights.pth \\
    ...     --input input.mp4 \\
    ...     --output output.mp4 \\
    ...     --threshold 0.5 \\
    ...     --fps 30 \\
    ...     --codec mp4v \\
    ...     --device 0
"""

from torchvision import transforms
from model import YOLOv1
from PIL import Image
import argparse
import time
import os
import cv2
import torch
import logging
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)

# BDD100K dataset classes
CATEGORY_LIST = ["other vehicle", "pedestrian", "traffic light", "traffic sign",
                 "truck", "train", "other person", "bus", "car", "rider",
                 "motorcycle", "bicycle", "trailer"]

# Class colors for bounding box visualization (BGR for OpenCV)
CATEGORY_COLORS = [(255, 255, 0), (255, 0, 0), (255, 128, 0), (0, 255, 255),
                   (255, 0, 255), (128, 255, 0), (0, 255, 128), (255, 0, 127),
                   (0, 255, 0), (0, 0, 255), (127, 0, 255), (0, 128, 255),
                   (128, 128, 128)]

# Model configuration
MODEL_INPUT_SIZE = 448
GRID_SIZE_MULTIPLIER = 32
DEFAULT_FPS = 30
DEFAULT_CODEC = "mp4v"


def parse_arguments():
    """
    Parse command-line arguments for YOLO inference on videos.
    
    Returns:
        argparse.Namespace: Parsed arguments with validated values
        
    Raises:
        ValueError: If required arguments are missing or invalid
    """
    parser = argparse.ArgumentParser(
        description="YOLO v1 object detection on videos with error handling"
    )
    parser.add_argument("-w", "--weights", required=True, type=str,
                        help="Path to model weights file (.pth)")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to input video file")
    parser.add_argument("-o", "--output", required=True, type=str,
                        help="Path to save output video with detections")
    parser.add_argument("-t", "--threshold", default=0.5, type=float,
                        help="Confidence threshold for bounding boxes (0.0-1.0)")
    parser.add_argument("-ss", "--split_size", default=14, type=int,
                        help="YOLO grid size (default: 14 for 14x14 grid)")
    parser.add_argument("-nb", "--num_boxes", default=2, type=int,
                        help="Number of bounding box predictions per grid cell")
    parser.add_argument("-nc", "--num_classes", default=13, type=int,
                        help="Number of object classes (default: 13 for BDD100K)")
    parser.add_argument("-d", "--device", default="0", type=str,
                        help="GPU device ID to use (default: 0, or 'cpu')")
    parser.add_argument("-f", "--fps", default=DEFAULT_FPS, type=int,
                        help=f"Output video FPS (default: {DEFAULT_FPS})")
    parser.add_argument("-c", "--codec", default=DEFAULT_CODEC, type=str,
                        help=f"Video codec for output (default: {DEFAULT_CODEC})")

    args = parser.parse_args()
    
    # Validate arguments
    if not (0.0 <= args.threshold <= 1.0):
        raise ValueError(f"Threshold must be between 0.0 and 1.0, got {args.threshold}")
    if args.split_size <= 0:
        raise ValueError(f"Split size must be positive, got {args.split_size}")
    if args.num_boxes <= 0:
        raise ValueError(f"Number of boxes must be positive, got {args.num_boxes}")
    if args.num_classes <= 0:
        raise ValueError(f"Number of classes must be positive, got {args.num_classes}")
    if args.fps <= 0:
        raise ValueError(f"FPS must be positive, got {args.fps}")
    if len(args.codec) != 4:
        raise ValueError(f"Codec must be 4 characters, got {args.codec}")
    
    return args


def setup_device(device_arg):
    """
    Setup PyTorch device with validation.
    
    Args:
        device_arg (str): Device ID ("0", "1", ...) or "cpu"
        
    Returns:
        torch.device: Configured device
        
    Raises:
        RuntimeError: If GPU requested but not available
    """
    try:
        if device_arg.lower() == "cpu":
            device = torch.device("cpu")
            logger.info("Using CPU for inference")
        else:
            try:
                device_id = int(device_arg)
                if device_id < 0:
                    raise ValueError("Device ID must be >= 0")
                
                if not torch.cuda.is_available():
                    logger.warning("CUDA not available, falling back to CPU")
                    return torch.device("cpu")
                
                if device_id >= torch.cuda.device_count():
                    raise RuntimeError(
                        f"GPU {device_id} not available. "
                        f"Available devices: {torch.cuda.device_count()}"
                    )
                
                os.environ["CUDA_VISIBLE_DEVICES"] = device_arg
                device = torch.device(f"cuda:{device_id}")
                logger.info(f"Using GPU {device_id} ({torch.cuda.get_device_name(device_id)})")
                
            except ValueError as e:
                raise ValueError(f"Invalid device specification: {device_arg}") from e
        
        return device
        
    except (RuntimeError, ValueError) as e:
        logger.error(f"Device setup failed: {e}")
        raise


def load_model_weights(weights_path, model, device):
    """
    Load and validate model weights.
    
    Args:
        weights_path (str): Path to weights file
        model (torch.nn.Module): Model to load weights into
        device (torch.device): Target device
        
    Raises:
        FileNotFoundError: If weights file doesn't exist
        ValueError: If weights file is corrupted or incompatible
    """
    try:
        weights_path = Path(weights_path)
        
        if not weights_path.exists():
            raise FileNotFoundError(f"Model weights not found: {weights_path}")
        
        if not weights_path.is_file():
            raise ValueError(f"Weights path is not a file: {weights_path}")
        
        try:
            logger.info(f"Loading weights from {weights_path}...")
            weights_dict = torch.load(weights_path, map_location=device)
            
            if "state_dict" not in weights_dict:
                raise ValueError("Weights file missing 'state_dict' key")
            
            model.load_state_dict(weights_dict["state_dict"])
            logger.info("Weights loaded successfully")
            
        except (EOFError, pickle.UnpicklingError) as e:
            raise ValueError(f"Corrupted weights file: {e}")
        except KeyError as e:
            raise ValueError(f"Incompatible weights format: missing {e}")
        except RuntimeError as e:
            raise ValueError(f"Weights shape mismatch: {e}")
            
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Weight loading failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading weights: {e}")
        raise RuntimeError(f"Failed to load weights: {e}")


def open_video_capture(video_path):
    """
    Open and validate video capture object.
    
    Args:
        video_path (str): Path to video file
        
    Returns:
        tuple: (cv2.VideoCapture, frame_width, frame_height, frame_count, fps)
        
    Raises:
        FileNotFoundError: If video file doesn't exist
        ValueError: If video cannot be opened or has invalid properties
    """
    try:
        video_path = Path(video_path)
        
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        if not video_path.is_file():
            raise ValueError(f"Video path is not a file: {video_path}")
        
        # Open video
        vs = cv2.VideoCapture(str(video_path))
        if not vs.isOpened():
            raise RuntimeError(f"Cannot open video file: {video_path}")
        
        # Get properties
        frame_width = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(vs.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = vs.get(cv2.CAP_PROP_FPS)
        
        # Validate properties
        if frame_width <= 0 or frame_height <= 0:
            raise ValueError(f"Invalid video dimensions: {frame_width}x{frame_height}")
        
        if frame_count <= 0:
            logger.warning(f"Cannot determine frame count, will process until EOF")
        
        if fps <= 0:
            logger.warning(f"Invalid FPS {fps}, will use default {DEFAULT_FPS}")
            fps = DEFAULT_FPS
        
        logger.info(f"Opened video: {video_path} ({frame_width}x{frame_height}, "
                   f"{frame_count} frames, {fps:.2f} FPS)")
        
        return vs, frame_width, frame_height, frame_count, fps
        
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        logger.error(f"Video loading failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error opening video: {e}")
        raise RuntimeError(f"Failed to open video: {e}")


def create_video_writer(output_path, frame_width, frame_height, fps, codec):
    """
    Create and validate video writer object.
    
    Args:
        output_path (str): Path for output video file
        frame_width (int): Frame width
        frame_height (int): Frame height
        fps (int): Frames per second
        codec (str): FourCC codec code (4 characters)
        
    Returns:
        cv2.VideoWriter: Configured video writer
        
    Raises:
        ValueError: If writer cannot be initialized
        IOError: If output path is not writable
    """
    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create codec
        fourcc = cv2.VideoWriter_fourcc(*codec)
        
        # Create writer
        out = cv2.VideoWriter(str(output_path), fourcc, fps, 
                             (frame_width, frame_height))
        
        if not out.isOpened():
            raise RuntimeError(f"Cannot create video writer with codec {codec}")
        
        logger.info(f"Video writer initialized: {output_path} "
                   f"({frame_width}x{frame_height}, {fps} FPS, {codec})")
        
        return out
        
    except (ValueError, RuntimeError, IOError) as e:
        logger.error(f"Video writer initialization failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error creating video writer: {e}")
        raise RuntimeError(f"Failed to create video writer: {e}")


def process_frame(frame, model, device, threshold, num_boxes, split_size, transform):
    """
    Run YOLO inference on a frame.
    
    Args:
        frame (np.ndarray): Input frame in BGR format
        model: YOLO model
        device: PyTorch device
        threshold (float): Confidence threshold
        num_boxes (int): Boxes per grid cell
        split_size (int): Grid size
        transform: Image transform pipeline
        
    Returns:
        tuple: (output_tensor, fps)
        
    Raises:
        RuntimeError: If inference fails
    """
    try:
        if frame is None or frame.size == 0:
            raise ValueError("Invalid frame")
        
        # Convert and transform
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img_tensor = transform(pil_img).unsqueeze(0).to(device)
        
        # Inference
        model.eval()
        with torch.no_grad():
            start_time = time.time()
            output = model(img_tensor)
            inference_time = time.time() - start_time
        
        if inference_time <= 0:
            raise RuntimeError("Invalid inference time")
        
        fps = int(1.0 / inference_time)
        return output, fps
        
    except (ValueError, RuntimeError) as e:
        logger.warning(f"Frame processing failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error processing frame: {e}")
        raise RuntimeError(f"Frame inference failed: {e}")


def draw_frame_detections(frame, output, frame_width, frame_height, threshold, 
                          num_boxes, split_size, fps):
    """
    Draw bounding boxes and labels on a frame.
    
    Args:
        frame (np.ndarray): Input frame
        output (torch.Tensor): Model output
        frame_width (int): Frame width
        frame_height (int): Frame height
        threshold (float): Confidence threshold
        num_boxes (int): Boxes per grid cell
        split_size (int): Grid size
        fps (int): Current FPS
        
    Returns:
        tuple: (annotated_frame, detection_count)
    """
    try:
        output_frame = frame.copy()
        
        # Scale factors
        ratio_x = frame_width / MODEL_INPUT_SIZE
        ratio_y = frame_height / MODEL_INPUT_SIZE
        
        # Get class predictions
        class_indices = torch.argmax(output[0, :, :, 5+num_boxes*5:], dim=2)
        
        detection_count = 0
        cell_dim = MODEL_INPUT_SIZE / split_size
        
        for cell_y in range(output.shape[1]):
            for cell_x in range(output.shape[2]):
                try:
                    # Find best box for this cell
                    best_box_idx = 0
                    max_confidence = 0
                    
                    for box_idx in range(num_boxes):
                        conf = output[0, cell_y, cell_x, box_idx * 5]
                        if conf > max_confidence:
                            max_confidence = conf
                            best_box_idx = box_idx
                    
                    # Check threshold
                    if output[0, cell_y, cell_x, best_box_idx * 5] < threshold:
                        continue
                    
                    detection_count += 1
                    
                    # Extract components
                    confidence = output[0, cell_y, cell_x, best_box_idx * 5].item()
                    bbox_start = best_box_idx * 5 + 1
                    bbox_coords = output[0, cell_y, cell_x, bbox_start:bbox_start + 4]
                    class_idx = class_indices[cell_y, cell_x].item()
                    
                    # Validate class
                    if class_idx >= len(CATEGORY_LIST):
                        continue
                    
                    # Convert to pixels
                    center_x = bbox_coords[0].item() * cell_dim + cell_dim * cell_x
                    center_y = bbox_coords[1].item() * cell_dim + cell_dim * cell_y
                    width = bbox_coords[2].item() * MODEL_INPUT_SIZE
                    height = bbox_coords[3].item() * MODEL_INPUT_SIZE
                    
                    # Scale to original frame
                    x1 = max(0, int((center_x - width / 2) * ratio_x))
                    y1 = max(0, int((center_y - height / 2) * ratio_y))
                    x2 = min(frame_width, int((center_x + width / 2) * ratio_x))
                    y2 = min(frame_height, int((center_y + height / 2) * ratio_y))
                    
                    if x1 >= x2 or y1 >= y2:
                        continue
                    
                    color = CATEGORY_COLORS[class_idx]
                    
                    # Draw box
                    cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw label
                    label_text = f"{CATEGORY_LIST[class_idx]} {confidence:.2f}"
                    label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_DUPLEX, 0.6, 1)
                    label_width = label_size[0][0]
                    label_height = label_size[0][1]
                    
                    cv2.rectangle(output_frame, (x1, max(0, y1 - label_height - 6)),
                                 (x1 + label_width + 6, y1), color, -1)
                    cv2.putText(output_frame, label_text, (x1 + 3, y1 - 3),
                               cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
                    
                except (ValueError, IndexError):
                    continue
        
        # Add FPS display
        fps_text = f"{fps} FPS"
        cv2.putText(output_frame, fps_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
                   0.8, (0, 255, 0), 2)
        
        return output_frame, detection_count
        
    except Exception as e:
        logger.warning(f"Error drawing detections: {e}")
        return frame, 0


def main():
    """Main video processing pipeline."""
    try:
        logger.info("=" * 70)
        logger.info("YOLO v1 Object Detection - Video Inference")
        logger.info("=" * 70)
        
        # Parse arguments
        args = parse_arguments()
        logger.info(f"Configuration: threshold={args.threshold}, "
                   f"split_size={args.split_size}, fps={args.fps}, codec={args.codec}")
        
        # Setup device
        device = setup_device(args.device)
        
        # Initialize model
        logger.info("Initializing YOLO model...")
        model = YOLOv1(args.split_size, args.num_boxes, args.num_classes).to(device)
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model parameters: {num_params:,}")
        
        # Load weights
        load_model_weights(args.weights, model, device)
        
        # Open video
        vs, frame_width, frame_height, frame_count, _ = open_video_capture(args.input)
        
        # Create output video
        out = create_video_writer(args.output, frame_width, frame_height, 
                                 args.fps, args.codec)
        
        # Setup transform
        transform = transforms.Compose([
            transforms.Resize((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), Image.NEAREST),
            transforms.ToTensor(),
        ])
        
        # Process video
        logger.info("Starting frame processing...")
        frame_idx = 0
        total_fps = 0
        total_detections = 0
        skipped_frames = 0
        
        while True:
            grabbed, frame = vs.read()
            if not grabbed:
                break
            
            frame_idx += 1
            
            # Log progress
            if frame_count > 0:
                progress = frame_idx / frame_count * 100
                logger.info(f"Processing frame {frame_idx}/{frame_count} ({progress:.1f}%)")
            else:
                logger.info(f"Processing frame {frame_idx}")
            
            try:
                # Run inference
                output, fps = process_frame(frame, model, device, args.threshold,
                                          args.num_boxes, args.split_size, transform)
                
                # Draw detections
                annotated_frame, num_detections = draw_frame_detections(
                    frame, output, frame_width, frame_height, args.threshold,
                    args.num_boxes, args.split_size, fps
                )
                
                total_fps += fps
                total_detections += num_detections
                
                logger.debug(f"Frame {frame_idx}: {fps} FPS, {num_detections} detections")
                
                # Write frame
                out.write(annotated_frame)
                
            except Exception as e:
                logger.warning(f"Error processing frame {frame_idx}: {e}, skipping")
                skipped_frames += 1
                out.write(frame)  # Write original frame
                continue
        
        # Cleanup
        vs.release()
        out.release()
        
        # Summary
        if frame_idx == 0:
            raise RuntimeError("No frames were processed")
        
        avg_fps = int(total_fps / frame_idx) if frame_idx > 0 else 0
        logger.info("=" * 70)
        logger.info("Video processing completed successfully")
        logger.info(f"Total frames: {frame_idx}")
        logger.info(f"Skipped frames: {skipped_frames}")
        logger.info(f"Total detections: {total_detections}")
        logger.info(f"Average FPS: {avg_fps}")
        logger.info(f"Output: {args.output}")
        logger.info("=" * 70)
        
    except Exception as e:
        logger.error(f"Video processing failed: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())