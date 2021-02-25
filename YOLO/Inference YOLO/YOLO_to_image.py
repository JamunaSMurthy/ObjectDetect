"""
YOLO v1 Image Object Detection Inference Script

Performs real-time object detection on single images using pretrained YOLOv1 model.
Detects multiple object classes from BDD100K dataset and draws bounding boxes with
confidence scores on output image.

Error Handling:
    - FileNotFoundError: Input image or model weights file missing
    - ValueError: Invalid argument parameters or corrupted model weights
    - RuntimeError: GPU initialization or model loading failure
    - IOError: Failed to read/write image files

Example:
    >>> python YOLO_to_image.py \\
    ...     --weights model_weights.pth \\
    ...     --input test.jpg \\
    ...     --output output.jpg \\
    ...     --threshold 0.5 \\
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

# Class colors for bounding box visualization (RGB)
CATEGORY_COLORS = [(255, 255, 0), (255, 0, 0), (255, 128, 0), (0, 255, 255),
                   (255, 0, 255), (128, 255, 0), (0, 255, 128), (255, 0, 127),
                   (0, 255, 0), (0, 0, 255), (127, 0, 255), (0, 128, 255),
                   (128, 128, 128)]

# Model configuration
MODEL_INPUT_SIZE = 448
GRID_SIZE_MULTIPLIER = 32


def parse_arguments():
    """
    Parse command-line arguments for YOLO inference on images.
    
    Returns:
        argparse.Namespace: Parsed arguments with validated values
        
    Raises:
        ValueError: If required arguments are missing or invalid
    """
    parser = argparse.ArgumentParser(
        description="YOLO v1 object detection on images with error handling"
    )
    parser.add_argument("-w", "--weights", required=True, type=str,
                        help="Path to model weights file (.pth)")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to input image file")
    parser.add_argument("-o", "--output", required=True, type=str,
                        help="Path to save output image with detections")
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


def load_and_validate_image(image_path):
    """
    Load and validate input image.
    
    Args:
        image_path (str): Path to image file
        
    Returns:
        tuple: (cv2_image, pil_image, width, height)
        
    Raises:
        FileNotFoundError: If image file doesn't exist
        ValueError: If image is corrupted or invalid format
    """
    try:
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        if not image_path.is_file():
            raise ValueError(f"Image path is not a file: {image_path}")
        
        # Read with OpenCV
        cv_img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if cv_img is None:
            raise ValueError(f"Failed to load image (corrupted or unsupported format): {image_path}")
        
        height, width = cv_img.shape[:2]
        if width <= 0 or height <= 0:
            raise ValueError(f"Image has invalid dimensions: {width}x{height}")
        
        # Convert to PIL for transforms
        pil_img = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
        
        logger.info(f"Loaded image: {image_path} ({width}x{height})")
        return cv_img, pil_img, width, height
        
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Image loading failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading image: {e}")
        raise RuntimeError(f"Failed to load image: {e}")


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


def run_inference(model, img_tensor, device):
    """
    Run YOLO inference on image tensor.
    
    Args:
        model (torch.nn.Module): Loaded YOLO model
        img_tensor (torch.Tensor): Preprocessed image tensor
        device (torch.device): Target device
        
    Returns:
        tuple: (output_tensor, inference_fps)
        
    Raises:
        RuntimeError: If inference fails
    """
    try:
        model.eval()
        
        with torch.no_grad():
            start_time = time.time()
            output = model(img_tensor)
            inference_time = time.time() - start_time
            
        if inference_time <= 0:
            raise RuntimeError("Invalid inference time")
        
        fps = int(1.0 / inference_time)
        logger.info(f"Inference completed: {fps} FPS ({inference_time*1000:.1f}ms)")
        
        return output, fps
        
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise RuntimeError(f"Model inference failed: {e}")


def draw_detections(cv_img, output, orig_width, orig_height, threshold, num_boxes, split_size):
    """
    Draw bounding boxes and labels on output image.
    
    Args:
        cv_img (np.ndarray): Input image in OpenCV format
        output (torch.Tensor): Model output tensor
        orig_width (int): Original image width
        orig_height (int): Original image height
        threshold (float): Confidence threshold
        num_boxes (int): Boxes per grid cell
        split_size (int): Grid size
        
    Returns:
        np.ndarray: Image with drawn detections
        int: Number of detections
    """
    try:
        # Scale factors for original image
        ratio_x = orig_width / MODEL_INPUT_SIZE
        ratio_y = orig_height / MODEL_INPUT_SIZE
        
        # Get class predictions
        class_indices = torch.argmax(output[0, :, :, 5+num_boxes*5:], dim=2)
        
        detection_count = 0
        cell_dim = MODEL_INPUT_SIZE / split_size
        
        for cell_y in range(output.shape[1]):
            for cell_x in range(output.shape[2]):
                # Find best box for this cell
                best_box_idx = 0
                max_confidence = 0
                
                for box_idx in range(num_boxes):
                    conf = output[0, cell_y, cell_x, box_idx * 5]
                    if conf > max_confidence:
                        max_confidence = conf
                        best_box_idx = box_idx
                
                # Check confidence threshold
                if output[0, cell_y, cell_x, best_box_idx * 5] < threshold:
                    continue
                
                detection_count += 1
                
                try:
                    # Extract prediction components
                    confidence = output[0, cell_y, cell_x, best_box_idx * 5].item()
                    bbox_start = best_box_idx * 5 + 1
                    bbox_coords = output[0, cell_y, cell_x, bbox_start:bbox_start + 4]
                    class_idx = class_indices[cell_y, cell_x].item()
                    
                    # Validate class index
                    if class_idx >= len(CATEGORY_LIST):
                        logger.warning(f"Invalid class index {class_idx}, skipping")
                        continue
                    
                    # Convert to pixel coordinates
                    center_x = bbox_coords[0].item() * cell_dim + cell_dim * cell_x
                    center_y = bbox_coords[1].item() * cell_dim + cell_dim * cell_y
                    width = bbox_coords[2].item() * MODEL_INPUT_SIZE
                    height = bbox_coords[3].item() * MODEL_INPUT_SIZE
                    
                    # Scale to original image
                    x1 = max(0, int((center_x - width / 2) * ratio_x))
                    y1 = max(0, int((center_y - height / 2) * ratio_y))
                    x2 = min(orig_width, int((center_x + width / 2) * ratio_x))
                    y2 = min(orig_height, int((center_y + height / 2) * ratio_y))
                    
                    if x1 >= x2 or y1 >= y2:
                        continue
                    
                    # Get class color
                    color = CATEGORY_COLORS[class_idx]
                    
                    # Draw bounding box
                    cv2.rectangle(cv_img, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw label with background
                    label_text = f"{CATEGORY_LIST[class_idx]} {confidence:.2f}"
                    label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_DUPLEX, 0.6, 1)
                    label_width = label_size[0][0]
                    label_height = label_size[0][1]
                    
                    cv2.rectangle(cv_img, (x1, max(0, y1 - label_height - 6)),
                                 (x1 + label_width + 6, y1), color, -1)
                    cv2.putText(cv_img, label_text, (x1 + 3, y1 - 3),
                               cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
                    
                except (ValueError, IndexError) as e:
                    logger.debug(f"Error drawing box at ({cell_x}, {cell_y}): {e}")
                    continue
        
        return cv_img, detection_count
        
    except Exception as e:
        logger.error(f"Error drawing detections: {e}")
        raise RuntimeError(f"Failed to draw detections: {e}")


def save_output_image(cv_img, output_path, fps):
    """
    Save output image with FPS counter.
    
    Args:
        cv_img (np.ndarray): Image to save
        output_path (str): Output file path
        fps (int): FPS to display
        
    Raises:
        IOError: If image cannot be written
    """
    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Add FPS counter
        fps_text = f"{fps} FPS"
        cv2.putText(cv_img, fps_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
                   0.8, (0, 255, 0), 2)
        
        success = cv2.imwrite(str(output_path), cv_img)
        if not success:
            raise IOError(f"Failed to write image to {output_path}")
        
        logger.info(f"Output image saved: {output_path}")
        
    except IOError as e:
        logger.error(f"Failed to save output image: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error saving image: {e}")
        raise RuntimeError(f"Failed to save image: {e}")


def main():
    """Main inference pipeline for YOLO on single images."""
    try:
        logger.info("=" * 60)
        logger.info("YOLO v1 Object Detection - Image Inference")
        logger.info("=" * 60)
        
        # Parse and validate arguments
        args = parse_arguments()
        logger.info(f"Configuration: threshold={args.threshold}, "
                   f"split_size={args.split_size}, "
                   f"num_classes={args.num_classes}")
        
        # Setup device
        device = setup_device(args.device)
        
        # Initialize model
        logger.info("Initializing YOLO model...")
        model = YOLOv1(args.split_size, args.num_boxes, args.num_classes).to(device)
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model parameters: {num_params:,}")
        
        # Load weights
        load_model_weights(args.weights, model, device)
        
        # Load input image
        cv_img, pil_img, img_width, img_height = load_and_validate_image(args.input)
        
        # Preprocess image
        logger.info("Preprocessing image...")
        transform = transforms.Compose([
            transforms.Resize((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), Image.NEAREST),
            transforms.ToTensor(),
        ])
        img_tensor = transform(pil_img).unsqueeze(0).to(device)
        logger.info(f"Tensor shape: {img_tensor.shape}")
        
        # Run inference
        logger.info("Running inference...")
        output, fps = run_inference(model, img_tensor, device)
        
        # Draw detections
        logger.info("Drawing detections...")
        output_img, num_detections = draw_detections(
            cv_img, output, img_width, img_height, args.threshold,
            args.num_boxes, args.split_size
        )
        logger.info(f"Detections: {num_detections} objects found")
        
        # Save output
        save_output_image(output_img, args.output, fps)
        
        logger.info("=" * 60)
        logger.info("Inference completed successfully")
        logger.info("=" * 60)
        
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        logger.error(f"Inference pipeline failed: {e}")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error in main: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())