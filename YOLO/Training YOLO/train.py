"""
YOLO v1 training script for object detection.

Trains the YOLOv1 model on custom datasets with progress tracking,
checkpoint saving, and loss monitoring.

Example:
	python train.py -tip ./images/train -ttp ./labels/train.json -ne 100 -bs 10
"""
import logging
from pathlib import Path
from model import YOLOv1
from loss import YOLO_Loss
from dataset import DataLoader
from utils import load_checkpoint, save_checkpoint
import torch.optim as optim
import torch
import time
import os
import argparse

logger = logging.getLogger(__name__)


def setup_logging():
	"""Configure logging for training."""
	logging.basicConfig(
		level=logging.INFO,
		format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
	)


def TrainNetwork(num_epochs, split_size, batch_size, load_size, num_boxes, num_classes, 
                 train_img_files_path, train_target_files_path, category_list, model, 
                 device, optimizer, load_model_file, lambda_coord, lambda_noobj):
	"""
	Train the YOLOv1 model.
	
	Args:
		num_epochs (int): Number of training epochs
		split_size (int): Grid size (default 14)
		batch_size (int): Batch size for training
		load_size (int): Number of batches to load at once
		num_boxes (int): Number of boxes per grid cell
		num_classes (int): Number of object classes
		train_img_files_path (str): Path to training images
		train_target_files_path (str): Path to training labels (JSON)
		category_list (list): List of class names
		model: YOLOv1 model instance
		device: GPU/CPU device
		optimizer: PyTorch optimizer
		load_model_file (str): Checkpoint file path
		lambda_coord (float): Coordinate loss weight
		lambda_noobj (float): No-object loss weight
	"""
	model.train()
	
	# Validate paths
	if not Path(train_img_files_path).exists():
		raise FileNotFoundError(f"Training images path not found: {train_img_files_path}")
	if not Path(train_target_files_path).exists():
		raise FileNotFoundError(f"Training labels path not found: {train_target_files_path}")
	
	try:
		# Initialize data loader
		logger.info(f"Initializing DataLoader for {len(category_list)} classes")
		data = DataLoader(train_img_files_path, train_target_files_path, category_list, 
						  split_size, batch_size, load_size)
		
		# Initialize loss log
		loss_log = {}
		torch.save(loss_log, "loss_log.pt")
		logger.info("Starting training...")
		
		for epoch in range(num_epochs):
			epoch_losses = []
			
			logger.info(f"\n{'='*60}")
			logger.info(f"Epoch {epoch+1}/{num_epochs}")
			logger.info(f"{'='*60}")
			
			try:
				data.LoadFiles()  # Reset for new epoch
			except Exception as e:
				logger.error(f"Error loading files for epoch {epoch+1}: {e}")
				raise

			while len(data.img_files) > 0:
				all_batch_losses = 0.0
				
				try:
					logger.info(f"Loading batches... ({len(data.img_files)} files remaining)")
					data.LoadData()
				except Exception as e:
					logger.error(f"Error loading data: {e}")
					raise
				
				if not data.data:
					logger.warning("No data loaded")
					break
				
				for batch_idx, (img_data, target_data) in enumerate(data.data):
					try:
						img_data = img_data.to(device)
						target_data = target_data.to(device)
						
						optimizer.zero_grad()
						predictions = model(img_data)
						
						yolo_loss = YOLO_Loss(predictions, target_data, split_size, num_boxes, 
											  num_classes, lambda_coord, lambda_noobj)
						yolo_loss.loss()
						loss = yolo_loss.final_loss
						
						if loss is None or torch.isnan(loss):
							logger.error(f"Invalid loss value at batch {batch_idx}")
							continue
						
						all_batch_losses += loss.item()
						
						loss.backward()
						optimizer.step()
						
						if (batch_idx + 1) % 10 == 0:
							logger.info(f'Epoch {epoch+1}/{num_epochs} | Batch {batch_idx+1}/{len(data.data)} | Loss: {loss.item():.6f}')
					
					except Exception as e:
						logger.error(f"Error in batch {batch_idx}: {e}")
						continue
				
				if len(data.data) > 0:
					avg_batch_loss = all_batch_losses / len(data.data)
					epoch_losses.append(avg_batch_loss)
					logger.info(f"Average batch loss: {avg_batch_loss:.6f}")
			
			# Save epoch results
			if epoch_losses:
				loss_log = torch.load('loss_log.pt')
				mean_loss = sum(epoch_losses) / len(epoch_losses)
				loss_log[f'Epoch {epoch+1}'] = mean_loss
				torch.save(loss_log, 'loss_log.pt')
				logger.info(f"Epoch {epoch+1} complete | Mean loss: {mean_loss:.6f}")
			
			# Save checkpoint
			try:
				checkpoint = {
					"state_dict": model.state_dict(),
					"optimizer": optimizer.state_dict(),
				}
				save_checkpoint(checkpoint, filename=load_model_file)
				logger.info(f"Checkpoint saved: {load_model_file}")
			except Exception as e:
				logger.error(f"Error saving checkpoint: {e}")
			
			time.sleep(2)
		
		logger.info("Training completed successfully")
	
	except Exception as e:
		logger.error(f"Fatal training error: {e}")
		raise


def main():
	"""Main training entry point."""
	setup_logging()
	
	try:
		# Parse arguments
		ap = argparse.ArgumentParser(description='Train YOLOv1 model')
		ap.add_argument("-tip", "--train_img_files_path", default="bdd100k/images/100k/train/",
						help="Path to training images")
		ap.add_argument("-ttp", "--train_target_files_path",
						default="bdd100k_labels_release/bdd100k/labels/det_v2_train_release.json",
						help="Path to training labels (JSON)")
		ap.add_argument("-lr", "--learning_rate", type=float, default=1e-5, help="Learning rate")
		ap.add_argument("-bs", "--batch_size", type=int, default=10, help="Batch size")
		ap.add_argument("-ne", "--number_epochs", type=int, default=100, help="Number of epochs")
		ap.add_argument("-ls", "--load_size", type=int, default=1000, help="Load size")
		ap.add_argument("-nb", "--number_boxes", type=int, default=2, help="Number of boxes")
		ap.add_argument("-lc", "--lambda_coord", type=float, default=5, help="Coord loss weight")
		ap.add_argument("-ln", "--lambda_noobj", type=float, default=0.5, help="No-obj loss weight")
		ap.add_argument("-lm", "--load_model", type=int, default=1, help="Load model (0/1)")
		ap.add_argument("-lmf", "--load_model_file", default="YOLO_bdd100k.pt", help="Model file")
		ap.add_argument("-device", type=str, default="0", help="GPU device ID")
		
		args = ap.parse_args()
		
		# Validate arguments
		if args.learning_rate <= 0:
			raise ValueError("Learning rate must be positive")
		if args.batch_size <= 0:
			raise ValueError("Batch size must be positive")
		if args.number_epochs <= 0:
			raise ValueError("Number of epochs must be positive")
		
		# Dataset parameters
		category_list = ["other vehicle", "pedestrian", "traffic light", "traffic sign",
						"truck", "train", "other person", "bus", "car", "rider",
						"motorcycle", "bicycle", "trailer"]
		
		# Configuration
		learning_rate = args.learning_rate
		batch_size = args.batch_size
		num_epochs = args.number_epochs
		load_size = args.load_size
		split_size = 14
		num_boxes = args.number_boxes
		lambda_coord = args.lambda_coord
		lambda_noobj = args.lambda_noobj
		num_classes = len(category_list)
		load_model = args.load_model
		
		# Set device
		os.environ["CUDA_VISIBLE_DEVICES"] = args.device
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		logger.info(f"Device: {device}")
		
		# Initialize model
		logger.info(f"Initializing YOLOv1 model (grid={split_size}, boxes={num_boxes}, classes={num_classes})")
		model = YOLOv1(split_size, num_boxes, num_classes).to(device)
		
		# Count parameters
		num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
		logger.info(f"Model parameters: {num_params:,}")
		
		# Initialize optimizer
		optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
		logger.info(f"Optimizer: SGD (lr={learning_rate}, momentum=0.9)")
		
		# Load checkpoint if requested
		if load_model:
			try:
				logger.info(f"Loading checkpoint: {args.load_model_file}")
				load_checkpoint(torch.load(args.load_model_file), model, optimizer)
			except FileNotFoundError:
				logger.warning(f"Checkpoint not found: {args.load_model_file}. Starting from scratch.")
			except Exception as e:
				logger.error(f"Error loading checkpoint: {e}")
				raise
		
		# Start training
		logger.info("Starting training...")
		TrainNetwork(num_epochs, split_size, batch_size, load_size, num_boxes,
					 num_classes, args.train_img_files_path, args.train_target_files_path,
					 category_list, model, device, optimizer, args.load_model_file,
					 lambda_coord, lambda_noobj)
		
		logger.info("Training finished successfully")
	
	except Exception as e:
		logger.error(f"Fatal error: {e}")
		raise


if __name__ == "__main__":
	main()
