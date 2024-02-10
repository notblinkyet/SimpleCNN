import torch
import pickle
import numpy as np
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from matplotlib import colors, pyplot as plt
import pickle
from sklearn.preprocessing import LabelEncoder
from PIL import Image
from pathlib import Path
%matplotlib inline

TRAIN_DIR = Path('/content/train')
TEST_DIR = Path("/content/testset")

train_val_files = sorted(list(TRAIN_DIR.rglob('*.jpg')))
test_files = sorted(list(TEST_DIR.rglob('*.jpg')))
DATA_MODES = ["test", "val", "train"]
RESCALE_SIZE = 224

class SpesialDataset(Dataset):
  def __init__(self, files, mode):
    super().__init__()
    self.files = files
    self.mode = mode

    if self.mode not in DATA_MODES:
      print(f"{self.mode} is not correct; correct modes: {DATA_MODES}")
      raise NameError

    self.len_ = len(self.files)

    self.label_encoder = LabelEncoder()

    if self.mode != "test":

      self.labels = [path.parent.name for path in self.files]
      self.label_encoder.fit(self.labels)

      with open("label_encode.pkl", "wb") as le_dump_file:
        pickle.dump(self.label_encoder, le_dump_file)

  def __len__(self):
    return self.len_

  def load_sample(self, file):
    image = Image.open(file)
    image.load()
    return image

  def _prepare_sample(self, image):
      image = image.resize((RESCALE_SIZE, RESCALE_SIZE))
      return np.array(image)

  def __getitem__(self, index):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    x = self.load_sample(self.files[index])
    x = self._prepare_sample(x)
    x = np.array(x / 255, dtype='float32')
    x = transform(x)
    if self.mode == 'test':
      return x
    else:
      label = self.labels[index]
      label_id = self.label_encoder.transform([label])
      y = label_id.item()
      return x, y
