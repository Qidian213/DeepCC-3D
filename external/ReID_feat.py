import torch
from torch import nn
from PIL import Image
import numpy as np
import torchvision.transforms as T
from external.resnet import ResNet

class Baseline(nn.Module):
    in_planes = 2048

    def __init__(self, num_classes, last_stride):
        super(Baseline, self).__init__()
        self.base = ResNet(last_stride)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.num_classes = num_classes

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)  # no shift
        
        self.rank_bn = nn.BatchNorm1d(self.in_planes)
        self.rank_bn.bias.requires_grad_(False)  # no shift
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)

    def forward(self, x):

        global_feat = self.gap(self.base(x))  # (b, 2048, 1, 1)
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)
        feat = self.bottleneck(global_feat)  # normalize for angular softmax
        return feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            if 'classifier' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])

def euclidean_dist_rank(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist

def process_img(img_path):
    img = Image.open(img_path).convert('RGB')
    img = transform(img).unsqueeze(0)
    img = img.to(device)
    return img

class ImageEncoder(object):
    def __init__(self, checkpoint_filename):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = Baseline(num_classes = 702,last_stride =1)
        self.model.load_param(checkpoint_filename)
        self.model.to(self.device)
        self.model.eval()
        
        self.normalize_transform = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transform = T.Compose([T.Resize([256, 128]),T.ToTensor(),self.normalize_transform])
   #     self.num = 0
        
    def get_bb_image(self, img, bb,camera_size):
        bb = np.round(bb)
        if bb[2] < 20 or bb[3] < 20:
            image = Image.new('RGB', (128, 256), (255, 255, 255))
            return image

        left = np.maximum(0,bb[0]).astype('int')
        right = np.minimum(camera_size[0]-1,bb[0]+bb[2]).astype('int')
        top = np.maximum(0,bb[1]).astype('int')
        bottom = np.minimum(camera_size[1]-1,bb[1]+bb[3]).astype('int')
        if left == right or top == bottom:
            image = Image.new('RGB', (128, 256), (255, 255, 255))
            return image
        snapshot = img.crop((left,top,right,bottom))#img[top:bottom,left:right,:]
        return snapshot

    def encoder(self,image, boxes,camera_size):
        image = Image.fromarray(image[...,::-1])
        image_patches = []
        for box in boxes:
            patch = self.get_bb_image(image,box,camera_size)
          #  patch.save('img/patch'+str(self.num)+'.jpg',quality=95)
            patch = self.transform(patch).unsqueeze(0)
            image_patches.append(patch)
         #   self.num = self.num +1
        image_patches = torch.cat(image_patches, dim=0)
        image_patches = image_patches.to(self.device)

        with torch.no_grad():
            features = self.model(image_patches)
        return features

def main():
    encoder = ImageEncoder('resnet50_model_80.pth')

#def main():
#    model = Baseline(num_classes = 702,last_stride =1)
#    model.load_param('xent+0.2rank_fi/resnet50_model_80.pth')
#    model.to(device)
#    model.eval()

#    feats = []
#    with torch.no_grad():
#        img1 = process_img('/home/zzg/Datasets/DukeReiD/DukeMTMC-reID/query/0033_c1_f0057706.jpg')
#        feat1 = model(img1)
#        feats.append(feat1)

#        img2 = process_img('/home/zzg/Datasets/DukeReiD/DukeMTMC-reID/query/0033_c6_f0045755.jpg')
#        feat2 = model(img2)
#        feats.append(feat2)

#        img3 = process_img('/home/zzg/Datasets/DukeReiD/DukeMTMC-reID/query/0034_c2_f0057453.jpg')
#        feat3 = model(img3)
#        feats.append(feat3)
#        
#    feats = torch.cat(feats, dim=0)
#    dist = euclidean_dist_rank(feats,feats)
#    print(dist)

if __name__ == '__main__':
    main()
    
