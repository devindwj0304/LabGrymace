'''
Copyright (C)
This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this program. If not, see https://tldrlegal.com/license/gnu-general-public-license-v3-(gpl-3)#fulltext.

For license issues, please contact:

Dr. Bing Ye
Life Sciences Institute
University of Michigan
210 Washtenaw Avenue, Room 5403
Ann Arbor, MI 48109-2216
USA

Email: bingye@umich.edu
'''

# !New Update from Wenjin -- this file differs from upstream LabGym 2.9.1:
# Configures TensorFlow and PyTorch GPU memory before any submodule is imported,
# so the two frameworks can share one GPU instead of the first one claiming all of
# it (TF memory growth + PYTORCH_CUDA_ALLOC_CONF). Without this the analysis runs
# out of GPU memory. Version is pinned to 2.9.0.
# Every change is marked with the same tag below.

# !New Update from Wenjin
# These GPU settings are applied at package import, before any submodule imports
# TensorFlow or PyTorch, so the two frameworks share the GPU and avoid out-of-memory
# errors.
import os

# Configure TensorFlow environment variables
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'

# Configure PyTorch to use less GPU memory (allow sharing with TensorFlow)
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

# Import and configure TensorFlow first
import tensorflow as tf
try:
	# Set memory growth for all GPUs
	gpus = tf.config.list_physical_devices('GPU')
	if gpus:
		for gpu in gpus:
			tf.config.experimental.set_memory_growth(gpu, True)
		print(f"[OK] LabGym: TensorFlow GPU memory growth enabled for {len(gpus)} GPU(s)")
	else:
		print("[WARNING] LabGym: No GPU detected, using CPU")
except Exception as e:
	print(f"[WARNING] LabGym: Could not configure TensorFlow GPU: {e}")

# Now import PyTorch (it will share GPU with TensorFlow)
import torch
try:
	if torch.cuda.is_available():
		# PyTorch will automatically share GPU memory with TensorFlow
		# thanks to TF_FORCE_GPU_ALLOW_GROWTH and PYTORCH_CUDA_ALLOC_CONF
		print(f"[OK] LabGym: PyTorch GPU enabled (shared memory mode)")
	else:
		print("[WARNING] LabGym: PyTorch using CPU")
except Exception as e:
	print(f"[WARNING] LabGym: Could not configure PyTorch GPU: {e}")

__version__='2.9.0'


# !New Update from Wenjin

