import torch
from torchvision.models.resnet import Bottleneck , ResNet
from transformers import BertTokenizer, BertModel
import torch.utils.model_zoo as model_zoo
from scipy import spatial
import numpy as np
import cv2

use_gpu = torch.cuda.is_available()
means = np.array([103.939, 116.779, 123.68]) / 255
model_urls = {
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def get_cosine_similarity(a, b):
    return 1 - spatial.distance.cosine(a, b)


def get_image_space_vector(image_path, model, pick_layer):
    img = cv2.imread(image_path, cv2.COLOR_RGB2BGR)
    img = img[:, :, ::-1]

    img = np.transpose(img, (2, 0, 1)) / 255.
    img[0] -= means[0]  # reduce B's mean
    img[1] -= means[1]  # reduce G's mean
    img[2] -= means[2]  # reduce R's mean
    img = np.expand_dims(img, axis=0)

    if use_gpu:
        inputs = torch.autograd.Variable(torch.from_numpy(img).cuda().float())
    else:
        inputs = torch.autograd.Variable(torch.from_numpy(img).float())
    d_hist = model(inputs)[pick_layer]
    d_hist = d_hist.data.cpu().numpy().flatten()
#     d_hist /= np.sum(d_hist)  # normalize
    return d_hist


def get_text_space_vector(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    return outputs[1][0]


class ResidualNet(ResNet):
    def __init__(self, model='resnet152', pretrained=True):
        super().__init__(Bottleneck, [3, 8, 36, 3], 1000)
        if pretrained:
            self.load_state_dict(model_zoo.load_url(model_urls[model]))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)  # x after layer4, shape = N * 512 * H/32 * W/32
        max_pool = torch.nn.MaxPool2d((x.size(-2),x.size(-1)), stride=(x.size(-2),x.size(-1)), padding=0, ceil_mode=False)
        Max = max_pool(x)  # avg.size = N * 512 * 1 * 1
        Max = Max.view(Max.size(0), -1)  # avg.size = N * 512
        avg_pool = torch.nn.AvgPool2d((x.size(-2),x.size(-1)), stride=(x.size(-2),x.size(-1)), padding=0, ceil_mode=False, count_include_pad=True)
        avg = avg_pool(x)  # avg.size = N * 512 * 1 * 1
        avg = avg.view(avg.size(0), -1)  # avg.size = N * 512
        fc = self.fc(avg)  # fc.size = N * 1000
        output = {
            'max': Max,
            'avg': avg,
            'fc' : fc
        }
        return output


class RoBertaChinese():
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')
        self.model = BertModel.from_pretrained('hfl/chinese-roberta-wwm-ext')