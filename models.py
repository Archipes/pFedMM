import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision.models as models


class LinearLayer(nn.Module):
    def __init__(self, input_dimension, num_classes, bias=True):
        super(LinearLayer, self).__init__()
        self.input_dimension = input_dimension
        self.num_classes = num_classes
        self.fc = nn.Linear(input_dimension, num_classes, bias=bias)

    def forward(self, x):
        return self.fc(x)


class FemnistCNN(nn.Module):
    """
    Implements a model with two convolutional layers followed by pooling, and a final dense layer with 2048 units.
    Same architecture used for FEMNIST in "LEAF: A Benchmark for Federated Settings"__
    We use `zero`-padding instead of  `same`-padding used in
     https://github.com/TalwalkarLab/leaf/blob/master/models/femnist/cnn.py.
    """
    def __init__(self, num_classes):
        super(FemnistCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5)

        self.fc1 = nn.Linear(64 * 4 * 4, 2048)
        self.output = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.output(x)
        return x

class EmnistFeature(nn.Module):
    def __init__(self):
        super(EmnistFeature, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.fc1 = nn.Linear(64 * 4 * 4, 2048)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        return x
    
class EmnistClassifier(nn.Module):
    def __init__(self, num_classes):
        super(EmnistClassifier, self).__init__()
        self.classifier = nn.Linear(2048, num_classes)
    
    def forward(self, x):
        x = F.relu(self.classifier(x))
        return x

class MobileNetFeature(nn.Module):
    def __init__(self):
        super(MobileNetFeature, self).__init__()
        self.feature = models.mobilenet_v2(pretrained=True).features
    def forward(self, x):
        x = self.feature(x)
        # Cannot use "squeeze" as batch-size can be 1
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        return x # 1280
    
class MobileNetClassifier(nn.Module):
    def __init__(self, num_classes):
        super(MobileNetClassifier, self).__init__()
        self.classifier = models.mobilenet_v2(pretrained=True).classifier
        self.classifier[1] = nn.Linear(self.classifier[1].in_features, num_classes)
    def forward(self, x):
        x = self.classifier(x)
        return x

class SwitchNorm1d(nn.Module):
    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self._inited = False
        self.weight_bn_ln = nn.Parameter(torch.tensor([0.5, 0.5], dtype=torch.float))
        self.register_parameter("gamma", None)
        self.register_parameter("beta",  None)

    def _maybe_init(self, Fdim: int):
        if not self._inited:
            self.gamma = nn.Parameter(torch.ones(1, Fdim))
            self.beta  = nn.Parameter(torch.zeros(1, Fdim))
            self._inited = True

    def forward(self, x: torch.Tensor):
        assert x.dim() == 2, "SwitchNorm1dLazy expects [N, F]"
        self._maybe_init(x.size(1))
        # BN 统计（跨 batch，逐特征）
        mu_bn, var_bn = x.mean(0, keepdim=True), x.var(0, keepdim=True, unbiased=False)
        x_bn = (x - mu_bn) / torch.sqrt(var_bn + self.eps)
        # LN 统计（样本内，跨特征）
        mu_ln, var_ln = x.mean(1, keepdim=True), x.var(1, keepdim=True, unbiased=False)
        x_ln = (x - mu_ln) / torch.sqrt(var_ln + self.eps)
        w = F.softmax(self.weight_bn_ln, dim=0)
        y = w[0] * x_bn + w[1] * x_ln
        return self.gamma * y + self.beta

class GateNetwork(nn.Module):
    def __init__(self, m: int, input= 28*28, temperature: float = 1.0, use_switchnorm: bool = False):
        super().__init__()
        self.flatten = nn.Flatten()
        self.sn = SwitchNorm1d() if use_switchnorm else nn.Identity()

        self.fc1 = nn.Linear(input, 128)      # in_features 懒推断
        self.bn1 = nn.BatchNorm1d(128)
        self.act1 = nn.Sigmoid()

        self.fc2 = nn.Linear(128, m)
        self.bn2 = nn.BatchNorm1d(m)

        self.temperature = temperature

    def forward(self, x):
        if x.dim() > 2:
            x = self.flatten(x)                 # [N, F]

        h1 = self.sn(x)                         # first: [N, F]
        if x.size(0) > 1:
            h2_pre = self.bn1(self.fc1(h1))         # second: [N, 128] (before Sigmoid)
        else:
            h2_pre = self.fc1(h1)
        h2 = self.act1(h2_pre)                  # second_act: [N, 128]
        if x.size(0) > 1:
            logits = self.bn2(self.fc2(h2))         # [N, m]
        else: 
            logits = self.fc2(h2)
        out = F.softmax(logits / self.temperature, dim=1)  # [N, m]
        return out

class CIFAR10CNN(nn.Module):
    def __init__(self, num_classes):
        super(CIFAR10CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5)
        
        self.fc1 = nn.Linear(64 * 5 * 5, 2048)
        self.output = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.output(x)
        return x


class NextCharacterLSTM(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, output_size, n_layers):
        super(NextCharacterLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.encoder = nn.Embedding(input_size, embed_size)

        self.rnn =\
            nn.LSTM(
                input_size=embed_size,
                hidden_size=hidden_size,
                num_layers=n_layers,
                batch_first=True
            )

        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, input_):
        encoded = self.encoder(input_)
        output, _ = self.rnn(encoded)
        output = self.decoder(output)
        output = output.permute(0, 2, 1)  # change dimension to (B, C, T)
        return output



def get_vgg11(n_classes):
    """
    creates VGG11 model with `n_classes` outputs
    :param n_classes:
    :return: nn.Module
    """
    model = models.vgg11(pretrained=True)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, n_classes)

    return model


def get_squeezenet(n_classes):
    """
    creates SqueezeNet model with `n_classes` outputs
    :param n_classes:
    :return: nn.Module
    """
    model = models.squeezenet1_0(pretrained=True)
    model.classifier[1] = nn.Conv2d(512, n_classes, kernel_size=(1, 1), stride=(1, 1))
    model.num_classes = n_classes

    return model


def get_mobilenet(n_classes):
    """
    creates MobileNet model with `n_classes` outputs
    :param n_classes:
    :return: nn.Module
    """
    model = models.mobilenet_v2(pretrained=True)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, n_classes)

    return model


def get_resnet18(n_classes):
    """
    creates Resnet model with `n_classes` outputs
    :param n_classes:
    :return: nn.Module
    """
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, n_classes)

    return model


def get_resnet34(n_classes):
    """
    creates Resnet34 model with `n_classes` outputs
    :param n_classes:
    :return: nn.Module
    """
    model = models.resnet34(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, n_classes)

    return model
