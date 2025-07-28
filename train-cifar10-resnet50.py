from pynvml import nvmlInit, nvmlShutdown, nvmlDeviceGetHandleByIndex, nvmlDeviceGetUtilizationRates
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from datetime import datetime
import torch.optim as optim
import torch.nn as nn
import pandas as pd
import subprocess
import time, os
import warnings
import torch
import math

warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.models._utils")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.utils.data.dataloader")

def get_gpu_utilizations():
	try:
		output = subprocess.check_output(
			["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
			encoding='utf-8'
		)
		return [float(x.strip()) for x in output.strip().split('\n')]
	except Exception as e:
		print(f"[WARN] Failed to get GPU utilization: {e}")
		return [0.0]

def get_data_size(dataset):
	sample, _ = dataset[0]
	single_sample_bytes = sample.element_size() * sample.nelement()
	total_bytes = single_sample_bytes * len(dataset)
	return total_bytes / (1024 * 1024)

def count_parameters(model):
	return sum(p.numel() for p in model.parameters())

def count_layers(model):
	total = 0
	for _ in model.modules():
		total += 1
	return total - 1

def get_resnet50(num_classes=10):
	model = models.resnet50(weights=None, num_classes=num_classes)
	model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
	model.maxpool = nn.Identity()
	return model

def train_resnet50():
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
	gpu_count = len(visible_devices.split(",")) if visible_devices else torch.cuda.device_count()

	batch_size = int(os.environ.get('OPTIMIZER_BATCH_SIZE', 128 * max(1, gpu_count)))
	num_workers = int(os.environ.get('OPTIMIZER_NUM_WORKERS', min(4, os.cpu_count() or 1)))
	pin_memory = os.environ.get('OPTIMIZER_PIN_MEMORY', 'true').lower() == 'true'
	epochs = int(os.environ.get('OPTIMIZER_EPOCHS', 20))

	try:
		gpu_hourly_cost = float(os.environ['COST_PER_GPU_HOUR'])
		cpu_hourly_cost = float(os.environ['COST_PER_CPU_HOUR'])
		memory_hourly_cost = float(os.environ['COST_PER_GB_HOUR'])
		estimated_memory_gb = float(os.environ.get('OPTIMIZER_MEMORY_USED_GB', 8))
	except KeyError as e:
		raise RuntimeError(f"Missing required cost environment variable: {e}")

	print(f"\nTraining Configuration:")
	print(f"Using device: {device}")
	print(f"Number of GPUs: {gpu_count}")
	print(f"Batch size: {batch_size}")
	print(f"Number of workers: {num_workers}")
	print(f"Pin memory: {pin_memory}")
	print(f"Total epochs: {epochs}\n")

	metrics_df = pd.DataFrame(columns=[
		'timestamp', 'epoch', 'model_name', 'model_params', 'model_layers',
		'dataset', 'data_count', 'data_size_mb', 'data_dim', 'batch_size',
		'device_type', 'cpu_count', 'gpu_count', 'train_loss', 'train_accuracy',
		'test_loss', 'test_accuracy', 'epoch_time', 'time_so_far', 'throughput',
		'gpu_memory_used_gb', 'cost_so_far'
	])

	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
	])

	train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
	test_dataset = datasets.CIFAR10('./data', train=False, download=True, transform=transform)
	data_count = len(train_dataset)
	data_size_mb = get_data_size(train_dataset)

	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
	test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

	model = get_resnet50()
	if gpu_count > 1:
		model = nn.DataParallel(model)
		print(f"DataParallel enabled across {gpu_count} GPUs")
	model = model.to(device)

	model_name = "resnet50"
	model_params = count_parameters(model)
	model_layers = count_layers(model)
	device_type = "GPU" if torch.cuda.is_available() else "CPU"

	optimizer = optim.Adam(model.parameters(), lr=0.001)
	criterion = nn.CrossEntropyLoss()

	training_start_time = time.time()

	nvmlInit()
	gpu_handles = [nvmlDeviceGetHandleByIndex(i) for i in range(torch.cuda.device_count())]

	for epoch in range(epochs):
		epoch_start_time = time.time()
		model.train()
		train_loss = 0
		correct = 0
		total = 0

		for batch_idx, (data, target) in enumerate(train_loader):
			data, target = data.to(device), target.to(device)
			optimizer.zero_grad()
			output = model(data)
			loss = criterion(output, target)
			loss.backward()
			optimizer.step()
			train_loss += loss.item()
			pred = output.argmax(dim=1, keepdim=True)
			correct += pred.eq(target.view_as(pred)).sum().item()
			total += target.size(0)

		model.eval()
		test_loss = 0
		test_correct = 0
		with torch.no_grad():
			for data, target in test_loader:
				data, target = data.to(device), target.to(device)
				output = model(data)
				test_loss += criterion(output, target).item()
				pred = output.argmax(dim=1, keepdim=True)
				test_correct += pred.eq(target.view_as(pred)).sum().item()

		epoch_end_time = time.time()
		epoch_time = epoch_end_time - epoch_start_time
		time_so_far = epoch_end_time - training_start_time
		throughput = len(train_dataset) / epoch_time

		util_this_epoch = [nvmlDeviceGetUtilizationRates(h).gpu for h in gpu_handles]

		metrics = {
			'timestamp': datetime.now(),
			'epoch': epoch,
			'model_name': model_name,
			'model_params': model_params,
			'model_layers': model_layers,
			'dataset': 'cifar10',
			'data_count': data_count,
			'data_size_mb': data_size_mb,
			'data_dim': '32x32',
			'batch_size': batch_size,
			'device_type': device_type,
			'cpu_count': os.cpu_count(),
			'gpu_count': gpu_count,
			'train_loss': train_loss / len(train_loader),
			'train_accuracy': 100. * correct / total,
			'test_loss': test_loss / len(test_loader),
			'test_accuracy': 100. * test_correct / len(test_dataset),
			'epoch_time': epoch_time,
			'time_so_far': time_so_far,
			'throughput': throughput,
			'gpu_memory_used_gb': torch.cuda.memory_allocated() / 1024**3 if gpu_count > 0 else 0.0
		}

		for i, usage in enumerate(util_this_epoch[:gpu_count]):
			metrics[f'gpu_utilization_{i}'] = usage

		epoch_cost = (epoch_time / 3600) * (
			gpu_count * gpu_hourly_cost +
			os.cpu_count() * cpu_hourly_cost +
			estimated_memory_gb * memory_hourly_cost
		)
		metrics['cost_so_far'] = epoch_cost if epoch == 0 else metrics_df['cost_so_far'].iloc[-1] + epoch_cost

		for col in metrics:
			if col not in metrics_df.columns:
				metrics_df[col] = pd.NA
				
		# Then directly add the new row using loc
		metrics_df.loc[len(metrics_df)] = metrics

		print(f"Epoch {epoch}: Loss: {metrics['train_loss']:.4f}, Acc: {metrics['test_accuracy']:.2f}%, Time: {epoch_time:.2f}s, Total Time: {time_so_far:.2f}s")
		metrics_df.to_csv('log-cifar10-resnet50.csv', index=False)

	total_time = time.time() - training_start_time
	metrics_df.to_csv('log-cifar10-resnet50.csv', index=False)
	nvmlShutdown()

	print("\nTraining Summary:")
	print(f"Total training time: {total_time/60:.2f} minutes")
	print(f"Final test accuracy: {metrics_df['test_accuracy'].iloc[-1]:.2f}%")
	print(f"Model parameters: {model_params:,}")
	print(f"Number of layers: {model_layers}")
	print(f"Dataset size: {data_size_mb:.2f} MB")
	print(f"Average throughput: {metrics_df['throughput'].mean():.2f} samples/sec")

	return metrics_df

if __name__ == "__main__":
	train_resnet50()
