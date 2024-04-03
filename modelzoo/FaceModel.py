import torch
from torch import nn
import torch.nn.functional as F

from facenet_pytorch.models.inception_resnet_v1 import InceptionResnetV1
from modelzoo.Blocks import RDB, ConvBlock
from modelzoo.FC import FC

class InceptionV1(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Create an inception resnet (in eval mode):
        resnet = InceptionResnetV1(pretrained='vggface2', classify=True).eval()
        for param in resnet.parameters():
            param.requires_grad = False
        resnet.logits = nn.Linear(512, 100)
        self.model = resnet

    def forward(self, x):
        return x, self.model(x)

class InceptionV2(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Create an inception resnet (in eval mode):
        resnet = InceptionResnetV1(pretrained='vggface2', classify=True).eval()
        for param in resnet.parameters():
            param.requires_grad = False
        resnet.logits = nn.Sequential(nn.Linear(512, 200),
                                      nn.ReLU(),
                                      nn.Dropout(),
                                      nn.Linear(200, 100),
                                      )
        self.model = resnet

    def forward(self, x):
        return x, self.model(x)

class FaceModel(nn.Module):
    def __init__(self, image_shape, n_classes, ae, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ae = ae
        self.flat = nn.Flatten()
        x_hat, lat = self.ae(torch.zeros((1,3,*image_shape)))
        c1 = lat.shape[1]
        c2 = c1 // 2
        c3 = c2 // 2
        c4 = c3 // 2
        self.rdb1 = RDB(c1, 6, 16)
        self.conv1 = ConvBlock(c1, c2)
        self.rdb2 = RDB(c2, 6, 16)
        self.conv2 = ConvBlock(c2, c3)
        self.rdb3 = RDB(c3, 6, 16)
        self.conv3 = ConvBlock(c3, c4)

        lat = self.rdb1(lat)
        lat = self.conv1(lat)
        lat = self.rdb2(lat)
        lat = self.conv2(lat)
        lat = self.rdb3(lat)
        lat = self.conv3(lat)

        n = self.flat(lat).shape[-1]
        print(f"Latent dimension {n}")
        self.fc = FC(dims=[n, n_classes*10, n_classes])

    def forward(self, x):
        x_hat, lat = self.ae(x)
        lat = self.rdb1(lat)
        lat = self.conv1(lat)
        lat = self.rdb2(lat)
        lat = self.conv2(lat)
        lat = self.rdb3(lat)
        lat = self.conv3(lat)
        lat = self.flat(lat)
        label = self.fc(lat)
        return x_hat, label


class FaceModelSmall(nn.Module):
    def __init__(self, image_shape, n_classes, ae, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ae = ae
        self.flat = nn.Flatten()
        x_hat, lat = self.ae(torch.zeros((1,3,*image_shape)))
        c1 = lat.shape[1]
        c2 = c1 // 2
        c3 = c2 // 2
        c4 = c3 // 2
        self.conv = ConvBlock(c1, c4)
        self.drop = nn.Dropout()
        lat = self.conv(lat)

        n = self.flat(lat).shape[-1]
        print(f"Latent dimension {n}")
        self.fc = FC(dims=[n, n_classes*2, n_classes])

    def forward(self, x):
        x_hat, lat = self.ae(x)
        lat = self.conv(lat)
        lat = self.flat(lat)
        label = self.fc(lat)
        return x_hat, label


class Simple(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(16, 32, 5)
        self.fc = FC(dims=[32 * 28 * 28, 6000, 3000, 100])

    def forward(self, x_in):
        x = self.pool(F.relu(self.conv1(x_in)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        label = self.fc(x)
        return x_in.clone(), label


class SimpleMedium(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(16, 32, 5)
        self.conv4 = nn.Conv2d(32, 64, 5)
        self.fc = FC(dims=[64 * 12 * 12, 1000, 200, 100])

    def forward(self, x_in):
        x = self.pool(F.relu(self.conv1(x_in)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        label = self.fc(x)
        return x_in.clone(), label

class SimpleMedium2(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(16, 32, 5)
        self.fc = FC(dims=[32 * 28 * 28, 500, 200, 100])

    def forward(self, x_in):
        x = self.pool(F.relu(self.conv1(x_in)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        label = self.fc(x)
        return x_in.clone(), label


class SimpleSmall(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(16, 32, 5)
        self.conv4 = nn.Conv2d(32, 64, 5)
        self.fc = FC(dims=[64 * 12 * 12, 200, 100])

    def forward(self, x_in):
        x = self.pool(F.relu(self.conv1(x_in)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        label = self.fc(x)
        return x_in.clone(), label

