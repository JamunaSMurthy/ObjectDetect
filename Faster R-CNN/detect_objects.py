"""
Faster R-CNN object detection script for images and videos.

Supports batch image processing and real-time video stream detection
with visualization of bounding boxes and confidence scores.

Example:
	python detect_objects.py --images_dir ./data/images --save_output
	python detect_objects.py --video_path ./data/video.mp4 --video_input --save_output
"""
import os
import cv2
import time
import argparse
import logging
from pathlib import Path
from detector import DetectorTF2

logger = logging.getLogger(__name__)


def setup_logging():
	"""Configure logging for this module."""
	logging.basicConfig(
		level=logging.INFO,
		format='%(asctime)s - %(levelname)s - %(message)s'
	)


def DetectFromVideo(detector, video_path, save_output=False, output_dir='output/'):
	"""
	Detect objects in video stream.
	
	Args:
		detector (DetectorTF2): Initialized detector instance
		video_path (str): Path to input video file
		save_output (bool): Whether to save annotated video
		output_dir (str): Directory for output video
	
	Raises:
		FileNotFoundError: If video file not found
		ValueError: If video cannot be opened
	"""
	if not Path(video_path).exists():
		raise FileNotFoundError(f"Video file not found: {video_path}")
	
	try:
		cap = cv2.VideoCapture(video_path)
		if not cap.isOpened():
			raise ValueError(f"Cannot open video: {video_path}")
		
		out = None
		if save_output:
			Path(output_dir).mkdir(parents=True, exist_ok=True)
			output_path = os.path.join(output_dir, 'detection_' + Path(video_path).name)
			frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
			frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
			out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (frame_width, frame_height))
			logger.info(f"Writing output to: {output_path}")

		frame_count = 0
		while cap.isOpened():
			ret, img = cap.read()
			if not ret:
				break

			frame_count += 1
			timestamp1 = time.time()
			det_boxes = detector.DetectFromImage(img)
			elapsed_time = round((time.time() - timestamp1) * 1000)  # ms
			img = detector.DisplayDetections(img, det_boxes, det_time=elapsed_time)

			if save_output and out is not None:
				out.write(img)
			
			if frame_count % 30 == 0:
				logger.info(f"Processed {frame_count} frames")

		cap.release()
		if out is not None:
			out.release()
		logger.info(f"Video processing complete. Total frames: {frame_count}")
		
	except Exception as e:
		logger.error(f"Error processing video: {e}")
		raise
	finally:
		if cap is not None:
			cap.release()


def DetectImagesFromFolder(detector, images_dir, save_output=False, output_dir='output/'):
	"""
	Detect objects in all images in a directory.
	
	Args:
		detector (DetectorTF2): Initialized detector instance
		images_dir (str): Path to directory with images
		save_output (bool): Whether to save annotated images
		output_dir (str): Directory for output images
	
	Raises:
		FileNotFoundError: If directory not found
	"""
	if not Path(images_dir).exists():
		raise FileNotFoundError(f"Images directory not found: {images_dir}")
	
	if save_output:
		Path(output_dir).mkdir(parents=True, exist_ok=True)
	
	try:
		image_files = list(Path(images_dir).glob('*'))
		valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
		image_files = [f for f in image_files if f.suffix.lower() in valid_extensions]
		
		if not image_files:
			logger.warning(f"No image files found in {images_dir}")
			return
		
		logger.info(f"Found {len(image_files)} images to process")
		
		for image_path in image_files:
			try:
				logger.info(f"Processing: {image_path.name}")
				img = cv2.imread(str(image_path))
				
				if img is None:
					logger.warning(f"Could not read image: {image_path}")
					continue

				img_help = cv2.resize(img, (640, 640), interpolation=cv2.INTER_AREA)
				det_boxes = detector.DetectFromImage(img_help)
				
				x_factor = img.shape[0] / 640
				y_factor = img.shape[1] / 640

				for i in range(len(det_boxes)):
					det_boxes[i][0] = int(det_boxes[i][0] * y_factor)
					det_boxes[i][1] = int(det_boxes[i][1] * x_factor)
					det_boxes[i][2] = int(det_boxes[i][2] * y_factor)
					det_boxes[i][3] = int(det_boxes[i][3] * x_factor)

				img = detector.DisplayDetections(img, det_boxes)
				logger.info(f"Detected {len(det_boxes)} objects")

				if save_output:
					img_out = os.path.join(output_dir, image_path.name)
					cv2.imwrite(img_out, img)
					logger.info(f"Saved to: {img_out}")
					
			except Exception as e:
				logger.error(f"Error processing {image_path}: {e}")
				continue
		
		logger.info(f"Batch processing complete")
		
	except Exception as e:
		logger.error(f"Error in batch processing: {e}")
		raise


if __name__ == "__main__":
	setup_logging()
	
	try:
		parser = argparse.ArgumentParser(description='Object Detection from Images or Video')
		parser.add_argument('--model_path', help='Path to frozen detection model',
							default='models/efficientdet_d0_coco17_tpu-32/saved_model')
		parser.add_argument('--path_to_labelmap', help='Path to labelmap (.pbtxt) file',
							default='models/mscoco_label_map.pbtxt')
		parser.add_argument('--class_ids', help='id of classes to detect, expects string with ids delimited by ","',
							type=str, default=None)
		parser.add_argument('--threshold', help='Detection Threshold', type=float, default=0.4)
		parser.add_argument('--images_dir', help='Directory to input images', default='data/samples/images/')
		parser.add_argument('--video_path', help='Path to input video', default='data/samples/pedestrian_test.mp4')
		parser.add_argument('--output_directory', help='Path to output images and video', default='data/samples/output')
		parser.add_argument('--video_input', help='Flag for video input', action='store_true')
		parser.add_argument('--save_output', help='Flag to save results', action='store_true')
		parser.add_argument('--device', help='GPU device ID (default: 0)', default='0')
		args = parser.parse_args()

		id_list = None
		if args.class_ids is not None:
			try:
				id_list = [int(item) for item in args.class_ids.split(',')]
			except ValueError as e:
				logger.error(f"Invalid class_ids format: {e}")
				raise

		logger.info(f"Initializing detector with threshold={args.threshold}")
		detector = DetectorTF2(args.model_path, args.path_to_labelmap, 
							   class_id=id_list, threshold=args.threshold, 
							   device_id=args.device)

		if args.video_input:
			DetectFromVideo(detector, args.video_path, args.save_output, args.output_directory)
		else:
			DetectImagesFromFolder(detector, args.images_dir, args.save_output, args.output_directory)
		
		logger.info("Detection completed successfully")
		
	except Exception as e:
		logger.error(f"Fatal error: {e}")
		raise

	if args.video_input:
		DetectFromVideo(detector, args.video_path, save_output=args.save_output, output_dir=args.output_directory)
	else:
		DetectImagesFromFolder(detector, args.images_dir, save_output=args.save_output, output_dir=args.output_directory)

	print("Done ...")
	#cv2.destroyAllWindows()
