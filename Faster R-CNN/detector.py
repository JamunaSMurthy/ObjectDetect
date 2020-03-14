"""
Faster R-CNN detector using TensorFlow 2.x

Provides object detection on images and videos using pre-trained Faster R-CNN models.
Supports custom class filtering and confidence thresholding.
"""
import cv2
import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util
import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class DetectorTF2:
	"""Faster R-CNN detector for real-time object detection using TensorFlow 2.x"""

	def __init__(self, path_to_checkpoint, path_to_labelmap, class_id=None, threshold=0.5, device_id=None):
		"""
		Initialize the Faster R-CNN detector with model and label configuration.
		
		"""
		Detect objects in a single image.
		
		Args:
			img (np.ndarray): Input image as numpy array (BGR format from OpenCV)
		
		Returns:
			list: Detections as [[x_min, y_min, x_max, y_max, class_label, confidence], ...]
		
		Raises:
			ValueError: If image is invalid
		"""
		try:
			if img is None:
				raise ValueError("Image is None or invalid")
			if not isinstance(img, np.ndarray):
				raise TypeError(f"Expected numpy array, got {type(img)}")
			
			im_height, im_width = img.shape[:2]
			
			# Expand dimensions since the model expects images to have shape: [1, None, None, 3]
			input_tensor = np.expand_dims(img, 0)
			detections = self.detect_fn(input_tensor)

			bboxes = detections['detection_boxes'][0].numpy()
			bclasses = detections['detection_classes'][0].numpy().astype(np.int32)
			bscores = detections['detection_scores'][0].numpy()
			det_boxes = self.ExtractBBoxes(bboxes, bclasses, bscores, im_width, im_height)
			
			logger.debug(f"Detected {len(det_boxes)} objects in image")
			return det_boxes
			
		except Exception as e:
			logger.error(f"Error during detection: {e}")
			raise


	def ExtractBBoxes(self, bboxes, bclasses, bscores, im_width, im_height):
		"""
		Extract valid bounding boxes from raw detections.
		
		Args:
			bboxes (np.ndarray): Detection boxes
			bclasses (np.ndarray): Class IDs
			bscores (np.ndarray): Confidence scores
			im_width (int): Image width
			im_height (int): Image height
		
		Returns:
			list: Filtered detections meeting threshold and class criteria
		"""
		bbox = []
		try:
			for idx in range(len(bboxes)):
				if self.class_id is None or bclasses[idx] in self.class_id:
					if bscores[idx] >= self.Threshold:
						y_min = int(bboxes[idx][0] * im_height)
						x_min = int(bboxes[idx][1] * im_width)
						y_max = int(bboxes[idx][2] * im_height)
						x_max = int(bboxes[idx][3] * im_width)
						class_label = self.category_index[int(bclasses[idx])]['name']
						bbox.append([x_min, y_min, x_max, y_max, class_label, float(bscores[idx])])
		except Exception as e:
			logger.error(f"Error extracting bounding boxes: {e}")
			raise
		
			if not isinstance(class_id, list):
				raise TypeError(f"class_id must be list or None, got {type(class_id)}")
			if not all(isinstance(id, int) for id in class_id):
				raise ValueError("class_id must contain only integers")
		
		# Configure GPU device
		if device_id is None:
			device_id = os.getenv("CUDA_VISIBLE_DEVICES", "0")
		os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
		logger.info(f"GPU device set to: {device_id}")
		
		self.class_id = class_id
		self.Threshold = threshold
		
		try:
			# Loading label map
			logger.info(f"Loading label map from {path_to_labelmap}")
			label_map = label_map_util.load_labelmap(path_to_labelmap)
			categories = label_map_util.convert_label_map_to_categories(
				label_map, max_num_classes=90, use_display_name=True
			)
			self.category_index = label_map_util.create_category_index(categories)
		"""
		Draw bounding boxes and labels on image.
		
		Args:
			image (np.ndarray): Input image
			boxes_list (list): Detection boxes from DetectFromImage
			det_time (float, optional): Detection time in milliseconds for FPS display
		
		Returns:
			np.ndarray: Image with drawn detections
		"""
		try:
			if image is None or not isinstance(image, np.ndarray):
				raise ValueError("Invalid image input")
			
			if not boxes_list:
				return image  # input list is empty
			
			img = image.copy()
			for idx in range(len(boxes_list)):
				x_min = int(boxes_list[idx][0])
				y_min = int(boxes_list[idx][1])
				x_max = int(boxes_list[idx][2])
				y_max = int(boxes_list[idx][3])
				cls = str(boxes_list[idx][4])
				score = str(np.round(boxes_list[idx][-1], 2))

				text = cls + ": " + score
				cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
				cv2.rectangle(img, (x_min, y_min - 20), (x_min, y_min), (255, 255, 255), -1)
				cv2.putText(img, text, (x_min + 5, y_min - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

			if det_time is not None:
				fps = round(1000. / det_time, 1)
				fps_txt = str(fps) + " FPS"
				cv2.putText(img, fps_txt, (25, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

			return img
			
		except Exception as e:
			logger.error(f"Error displaying detections: {e}")
			raise detections['detection_classes'][0].numpy().astype(np.int32)
		bscores = detections['detection_scores'][0].numpy()
		det_boxes = self.ExtractBBoxes(bboxes, bclasses, bscores, im_width, im_height)

		return det_boxes


	def ExtractBBoxes(self, bboxes, bclasses, bscores, im_width, im_height):
		bbox = []
		for idx in range(len(bboxes)):
			if self.class_id is None or bclasses[idx] in self.class_id:
				if bscores[idx] >= self.Threshold:
					y_min = int(bboxes[idx][0] * im_height)
					x_min = int(bboxes[idx][1] * im_width)
					y_max = int(bboxes[idx][2] * im_height)
					x_max = int(bboxes[idx][3] * im_width)
					class_label = self.category_index[int(bclasses[idx])]['name']
					bbox.append([x_min, y_min, x_max, y_max, class_label, float(bscores[idx])])
		return bbox


	def DisplayDetections(self, image, boxes_list, det_time=None):
		if not boxes_list: return image  # input list is empty
		img = image.copy()
		for idx in range(len(boxes_list)):
			x_min = boxes_list[idx][0]
			y_min = boxes_list[idx][1]
			x_max = boxes_list[idx][2]
			y_max = boxes_list[idx][3]
			cls =  str(boxes_list[idx][4])
			score = str(np.round(boxes_list[idx][-1], 2))

			text = cls + ": " + score
			cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
			cv2.rectangle(img, (x_min, y_min - 20), (x_min, y_min), (255, 255, 255), -1)
			cv2.putText(img, text, (x_min + 5, y_min - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

		if det_time != None:
			fps = round(1000. / det_time, 1)
			fps_txt = str(fps) + " FPS"
			cv2.putText(img, fps_txt, (25, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

		return img

