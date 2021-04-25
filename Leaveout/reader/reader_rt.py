import numpy as np
import cv2 
import os
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms

def gazeto2d(gaze):
  yaw = np.arctan2(-gaze[0], -gaze[2])
  pitch = np.arcsin(-gaze[1])
  return np.array([yaw, pitch])

class loader(Dataset): 
  def __init__(self, path, root, header=True):
    self.lines = []
    if isinstance(path, list):
      for i in path:
        with open(i) as f:
          line = f.readlines()
          if header: line.pop(0)
          self.lines.extend(line)
    else:
      with open(path) as f:
        self.lines = f.readlines()
        if header: self.lines.pop(0)

    self.root = root
    self.transform = transforms.ToTensor()

  def __len__(self):
    return len(self.lines)

  def __getitem__(self, idx):
    line = self.lines[idx]
    line = line.strip().split(" ")

    name = line[0]
    gaze2d = line[6]
    head2d = line[7]
    face = line[0]
    lefteye = line[1]
    righteye = line[2]

    label = np.array(gaze2d.split(",")).astype("float")
    label = torch.from_numpy(label).type(torch.FloatTensor)

    headpose = np.array(head2d.split(",")).astype("float")
    headpose = torch.from_numpy(headpose).type(torch.FloatTensor)
    
    fimg = cv2.imread(os.path.join(self.root, face))
    fimg = cv2.resize(fimg, (96, 96))
    fimg = self.transform(fimg)


    rimg = cv2.imread(os.path.join(self.root, righteye))
    rimg = cv2.resize(rimg, (96, 64))
    rimg = self.transform(rimg)

    limg = cv2.imread(os.path.join(self.root, lefteye))
    limg = cv2.resize(limg, (96, 64))
    limg = self.transform(limg)


    img = {"left": limg,
           "right": rimg,
           "face": fimg,
           "head_pose":headpose,
           "name":name}

    return img, label

def txtload(labelpath, imagepath, batch_size, shuffle=True, num_workers=0, header=True):
  dataset = loader(labelpath, imagepath, header)
  print(f"[Read Data]: Total num: {len(dataset)}")
  print(f"[Read Data]: Label path: {labelpath}")
  load = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
  return load


if __name__ == "__main__":
  path = './p00.label'
  d = loader(path)
  print(len(d))
  (data, label) = d.__getitem__(0)

