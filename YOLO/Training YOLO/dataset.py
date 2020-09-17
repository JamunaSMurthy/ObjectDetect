"""
YOLO v1 Dataset Loader
Handles loading and preprocessing BDD100K or custom JSON-annotated datasets.
"""
import torch
from torchvision import transforms
from os import listdir
from PIL import Image
import json
import random
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class DataLoader:
	"""Loads and preprocesses image and label data for YOLO v1 training."""

	def __init__(self, img_files_path, target_files_path, category_list, split_size, 
				 batch_size, load_size):
		"""
		Initialize DataLoader with paths and configuration.
		
		Args:
			img_files_path (str): Path to training images directory
			target_files_path (str): Path to JSON labels file
			category_list (list): List of class names
			split_size (int): Grid size (default 14)
			batch_size (int): Batch size
			load_size (int): Number of batches to load per call
		
		Raises:
			FileNotFoundError: If paths don't exist
			ValueError: If parameters are invalid
		"""
		
		# Validate paths
		if not Path(img_files_path).exists():
			raise FileNotFoundError(f"Images path not found: {img_files_path}")
		if not Path(target_files_path).exists():
			raise FileNotFoundError(f"Labels path not found: {target_files_path}")
		
		if batch_size <= 0 or split_size <= 0:
			raise ValueError("batch_size and split_size must be positive")
		
		self.img_files_path = img_files_path
		self.target_files_path = target_files_path
		self.category_list = category_list
		self.num_classes = len(category_list)
		self.split_size = split_size
		self.batch_size = batch_size
		self.load_size = load_size
		
		self.img_files = []
		self.target_files = []
		self.data = []
		self.img_tensors = []
		self.target_tensors = []
		self._current_img = None  # Cache for transform_label_to_tensor
		
		# Image preprocessing
		self.transform = transforms.Compose([
			transforms.Resize((448, 448), Image.NEAREST),
			transforms.ToTensor(),
		])
		
		logger.info(f"DataLoader initialized: {len(category_list)} classes, split_size={split_size}")
	

	def LoadFiles(self):
		"""
		Load image filenames and labels from disk.
		
		Raises:
			FileNotFoundError: If labels file is corrupt
			json.JSONDecodeError: If JSON is invalid
		"""
		try:
			# Load image filenames
			self.img_files = listdir(self.img_files_path)
			if not self.img_files:
				logger.warning(f"No images found in {self.img_files_path}")
				return
			
			logger.info(f"Loaded {len(self.img_files)} images")
			
			# Load labels database
			try:
				with open(self.target_files_path, 'r') as f:
					self.target_files = json.load(f)
				logger.info(f"Loaded {len(self.target_files)} label entries")
			except json.JSONDecodeError as e:
				raise ValueError(f"Invalid JSON in label file: {e}")
			except Exception as e:
				raise FileNotFoundError(f"Cannot read label file: {e}")
				
		except Exception as e:
			logger.error(f"Error loading files: {e}")
			raise
	

	def LoadData(self):
		"""
		Load and preprocess image-label pairs into batches.
		
		Raises:
			RuntimeError: If data loading fails
		"""
		try:
			# Reset cache
			self.data = []
			self.img_tensors = []
			self.target_tensors = []

			for i in range(len(self.img_files)):
				# Check if batch is full
				if len(self.img_tensors) == self.batch_size:
					self.data.append((torch.stack(self.img_tensors),
									  torch.stack(self.target_tensors)))
					self.img_tensors = []
					self.target_tensors = []
					logger.info(f'Batch {len(self.data)} / {self.load_size} loaded ({len(self.data)/self.load_size*100:.1f}%)')
				
				if len(self.data) == self.load_size:
					break  # Desired load size reached
				
				# Extract and add image-label pair
				self.extract_image_and_label()
			
			if not self.data and self.img_tensors:
				# Add remaining partial batch
				self.data.append((torch.stack(self.img_tensors),
								  torch.stack(self.target_tensors)))
			
			logger.info(f"Loaded {len(self.data)} batches")
			
		except Exception as e:
			logger.error(f"Error loading data: {e}")
			raise
	

	def extract_image_and_label(self):
		"""Extract random image-label pair and append to batch."""
		try:
			img_tensor, chosen_image = self.extract_image()
			target_tensor = self.extract_json_label(chosen_image)
			
			if target_tensor is not None:
				self.img_tensors.append(img_tensor)
				self.target_tensors.append(target_tensor)
			else:
				logger.warning(f"No valid label found for {chosen_image}")
				
		except Exception as e:
			logger.error(f"Error extracting image-label pair: {e}")
	

	def extract_image(self):
		"""
		Load and preprocess a random image.
		
		Returns:
			tuple: (image_tensor, image_filename)
		"""
		if not self.img_files:
			raise RuntimeError("No more images to load")
		
		try:
			f = random.choice(self.img_files)
			self.img_files.remove(f)
			
			img_path = Path(self.img_files_path) / f
			if not img_path.exists():
				raise FileNotFoundError(f"Image not found: {img_path}")
			
			img = Image.open(img_path)
			if img.mode != 'RGB':
				img = img.convert('RGB')  # Ensure RGB format
			
			self._current_img = img  # Cache for label transformation
			img_tensor = self.transform(img)
			
			return img_tensor, f
			
		except Exception as e:
			logger.error(f"Error loading image: {e}")
			raise
	

	def extract_json_label(self, chosen_image):
		"""
		Find and process JSON label for given image.
		
		Args:
			chosen_image (str): Image filename
		
		Returns:
			tensor or None: Label tensor if found, None otherwise
		"""
		try:
			for json_el in self.target_files:
				if json_el.get('name') == chosen_image:
					img_label = json_el
					
					if not img_label.get('labels'):
						logger.debug(f"No labels for {chosen_image}")
						return None
					
					target_tensor = self.transform_label_to_tensor(img_label)
					return target_tensor
			
			logger.debug(f"Image not found in label database: {chosen_image}")
			return None
			
		except Exception as e:
			logger.error(f"Error extracting label for {chosen_image}: {e}")
			return None
	

	def transform_label_to_tensor(self, img_label):
		"""
		Transform JSON label to tensor format.
		
		Args:
			img_label (dict): JSON label dictionary
		
		Returns:
			tensor: Target tensor of shape (split_size, split_size, 5+num_classes)
		"""
		try:
			if self._current_img is None:
				raise RuntimeError("No current image cached")
			
			target_tensor = torch.zeros(self.split_size, self.split_size, 5 + self.num_classes)

			for label_dict in img_label.get("labels", []):
				try:
					# Get category
					category = label_dict.get("category")
					if category not in self.category_list:
						continue
					
					ctg_idx = self.category_list.index(category)

					# Get bounding box (with scale to 448x448)
					box2d = label_dict.get("box2d", {})
					x1 = box2d.get("x1", 0) * (448 / self._current_img.size[0])
					y1 = box2d.get("y1", 0) * (448 / self._current_img.size[1])
					x2 = box2d.get("x2", 0) * (448 / self._current_img.size[0])
					y2 = box2d.get("y2", 0) * (448 / self._current_img.size[1])
					
					if x1 == x2 or y1 == y2:
						continue  # Invalid box
					
					# Convert to center coordinates
					x_mid = abs(x2 - x1) / 2 + x1
					y_mid = abs(y2 - y1) / 2 + y1
					width = abs(x2 - x1)
					height = abs(y2 - y1)

					# Get cell position
					cell_dim = 448 / self.split_size
					cell_pos_x = min(int(x_mid // cell_dim), self.split_size - 1)
					cell_pos_y = min(int(y_mid // cell_dim), self.split_size - 1)

					# Skip if cell already has object
					if target_tensor[cell_pos_y][cell_pos_x][0] == 1:
						continue
					
					# Store in tensor
					target_tensor[cell_pos_y][cell_pos_x][0] = 1  # Confidence
					target_tensor[cell_pos_y][cell_pos_x][1] = (x_mid % cell_dim) / cell_dim  # x
					target_tensor[cell_pos_y][cell_pos_x][2] = (y_mid % cell_dim) / cell_dim  # y
					target_tensor[cell_pos_y][cell_pos_x][3] = width / 448  # width
					target_tensor[cell_pos_y][cell_pos_x][4] = height / 448  # height
					target_tensor[cell_pos_y][cell_pos_x][ctg_idx + 5] = 1  # class probability
					
				except (KeyError, TypeError, IndexError) as e:
					logger.warning(f"Invalid label entry: {e}")
					continue

			return target_tensor
			
		except Exception as e:
			logger.error(f"Error transforming label: {e}")
			raise
