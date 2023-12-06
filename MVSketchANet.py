import torch
import torch.nn as nn

# SubNet of Triple
class SketchSubNet(nn.Module):
    def __init__(self, num_feat=100, num_views=5):
        super(SketchSubNet, self).__init__()
        self.num_feat = num_feat
        self.num_views = num_views
        # Sketch-a-net
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=64,
                kernel_size=15,
                stride=3,
                padding=0
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=3,
                stride=2
            )
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=5,
                stride=1,
                padding=0
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=3,
                stride=2
            )
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=3,
                stride=2
            )
        )
        self.fc6 = nn.Sequential(
            nn.Linear(256 * 7 * 7, 2048),
            nn.ReLU(),
            nn.Dropout(p=0.55)
        )
        self.fc7 = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(p=0.55)
        )
        self.feat = nn.Linear(512, num_feat)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        y = x
        # MVCNN, View Pooling
        y = y.view((int(x.shape[0] / self.num_views), self.num_views, y.shape[-3], y.shape[-2], y.shape[-1]))
        x = torch.max(y, 1)[0].view(y.shape[0], -1)
        x = self.fc6(x)
        x = self.fc7(x)
        x = self.feat(x)
        return x

    # return sub net
    def get_subNet(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.size(0), -1)

        x = self.fc6(x)
        x = self.fc7(x)
        x = self.feat(x)
        return x

    def get_net(self, x):
        return self.forward(x)


# Triple Net
class SketchMViewTripletNet(nn.Module):
    def __init__(self, SketchSubNet):
        super(SketchMViewTripletNet, self).__init__()
        self.branch_net = SketchSubNet

        #
        self.conv1_a = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=64,
                kernel_size=15,
                stride=3,
                padding=0
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=3,
                stride=2
            )
        )
        self.conv2_a = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=5,
                stride=1,
                padding=0
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=3,
                stride=2
            )
        )
        self.conv3_a = nn.Sequential(
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU()
        )

    def anchorResult(self, x):
        x = self.branch_net.get_subNet(x)
        return x

    def forward(self, anchor_input, positive_input, negative_input):
        feat_a = self.anchorResult(anchor_input)
        feat_p = self.branch_net(positive_input)
        feat_n = self.branch_net(negative_input)
        return feat_a, feat_p, feat_n

    def get_sketch_feat(self, x):
        return self.anchorResult(x)

    def get_Image_feat(self, x):
        return self.branch_net(x)
