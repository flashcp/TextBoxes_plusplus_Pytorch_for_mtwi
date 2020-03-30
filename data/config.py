# config.py
import os.path

# gets home dir cross platform
HOME = os.path.expanduser("~")

# for making bounding boxes pretty
COLORS = ((255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
          (0, 255, 255, 128), (255, 0, 255, 128), (255, 255, 0, 128))

MEANS = (104, 117, 123)

# SSD300 CONFIGS
mtwi384 = {
    'num_classes': 2,
    'lr_steps': (80000, 100000, 120000),
    'max_iter': 90100,
    'feature_maps': [48, 24, 12, 6, 4, 2],
    'min_dim': 384,
    'steps': [8, 16, 32, 64, 100, 200],
    'min_sizes': [38, 76, 142, 207, 272, 337],
    'max_sizes': [76, 142, 207, 272, 337, 403],
    'aspect_ratios': [[2, 3], [2, 3, 5], [2, 3, 5], [2, 3, 5], [2, 3], [2, 3]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'mtwi',
}

# SSD768 CONFIGS
mtwi768 = {
    'num_classes': 2,
    'lr_steps': (80000, 100000, 120000),
    'max_iter': 90100,
    'feature_maps': [96, 48, 24, 12, 10, 8],
    'min_dim': 768,
    'steps': [8, 16, 32, 64, 80, 100],
    'min_sizes': [76, 153, 284, 414, 545, 675],
    'max_sizes': [153, 284, 414, 545, 675, 806],
    'aspect_ratios': [[2, 3], [2, 3, 5], [2, 3, 5], [2, 3, 5], [2, 3], [2, 3]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'mtwi',
}
