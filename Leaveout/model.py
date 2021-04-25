import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import math

class DilatedNet(nn.Module):
    def __init__(self):
        super(DilatedNet, self).__init__()
        # 36 * 60
        # 18 * 30
        # 9  * 15
        eyeconv = models.vgg16(pretrained=True).features
        # self.eyeStreamConv.load_state_dict(vgg16.features[0:9].state_dict())
        # self.faceStreamPretrainedConv.load_state_dict(vgg16.features[0:15].state_dict())
    
        # Eye Stream, is composed of Conv DConv and FC layers. 
        self.eyeStreamConv = nn.Sequential(
            eyeconv[0], #nn.Conv2d(3,   64,  3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            eyeconv[2], #nn.Conv2d(64,  64,  3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            eyeconv[5], #nn.Conv2d(64,  128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            eyeconv[7], #nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.eyeStreamDConv = nn.Sequential(
            nn.Conv2d(128, 64, 1, 1), 
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, 3, 1,  dilation=(2, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, 3, 1,  dilation=(3, 3)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, 3, 1, dilation=(4, 5)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, 3, 1, dilation=(5, 11)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.eyeStreamFC = nn.Sequential(
            nn.Linear(128*4*6, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
       
        # Face Stream, is composed of Conv and FC layers.
        faceconv = models.vgg16(pretrained=True).features
        self.faceStreamPretrainedConv = nn.Sequential(
            # 224*224
            faceconv[0], # nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            faceconv[2], # nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(2, 2),

            # 112*112
            faceconv[5], # nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            faceconv[7], # nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),  
            nn.MaxPool2d(2, 2),
            
            # 56*56
            faceconv[10], # nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            faceconv[12], # nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            faceconv[14], # nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

  
        self.faceStreamConv = nn.Sequential(
            nn.Conv2d(256, 64, 1),
            nn.MaxPool2d(2,2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # 28*28
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2,2),
            nn.ReLU(inplace=True),
        )

        self.faceStreamFC = nn.Sequential(
            nn.Linear(64 * 6 * 6, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        
        self.totalFC = nn.Sequential(
            nn.Linear(256+256+32, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 2)
        )

    def forward(self, x_in):
        # Get face feature
        faceFeatureMap = self.faceStreamPretrainedConv(x_in['face'])
        faceFeatureMap = self.faceStreamConv(faceFeatureMap)

        faceFeature = torch.flatten(faceFeatureMap, start_dim=1)
        faceFeature = self.faceStreamFC(faceFeature)

        # Get left feature
        leftEyeFeature = self.eyeStreamConv(x_in['left'])
        leftEyeFeature = self.eyeStreamDConv(leftEyeFeature)
        leftEyeFeature = torch.flatten(leftEyeFeature, start_dim=1)
        leftEyeFeature = self.eyeStreamFC(leftEyeFeature)
 
        # Get Right feature
        rightEyeFeature = self.eyeStreamConv(x_in['right'])
        rightEyeFeature = self.eyeStreamDConv(rightEyeFeature)
        rightEyeFeature = torch.flatten(rightEyeFeature, start_dim=1)
        rightEyeFeature = self.eyeStreamFC(rightEyeFeature)
 
        features = torch.cat((faceFeature, leftEyeFeature, rightEyeFeature), 1)

        gaze = self.totalFC(features)

        return gaze

if __name__ == '__main__':
    m = DilatedNet()
    m.to("cuda")
    feature = {"face":torch.zeros(64, 3, 96, 96).to("cuda"),
                "left":torch.zeros(64, 3, 64,96).to("cuda"),
                "right":torch.zeros(64, 3, 64,96).to("cuda")
              }
    a = m(feature)
    print(m)

