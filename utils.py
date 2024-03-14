import torch
import numpy as np
import math
from numbers import Number
import torchvision.transforms.functional as TF
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import random
from PIL import ImageFilter

def get_statistics(dataset: str):
    """
    Returns statistics of the dataset given a string of dataset name. To add new dataset, please add required statistics here
    """
    assert dataset in [
        "cifar10",
        "cifar100",
        "tinyimagenet",
        "imagenet200",
        "imagenet",
        "clear10",
        "clear100"
    ]
    mean = {
        "cifar10": (0.4914, 0.4822, 0.4465),
        "cifar100": (0.5071, 0.4867, 0.4408),
        "tinyimagenet": (0.4802, 0.4481, 0.3975),
        "imagenet200": (0.485, 0.456, 0.406),
        "imagenet": (0.485, 0.456, 0.406),
        "clear10": (0.485, 0.456, 0.406),
        "clear100": (0.485, 0.456, 0.406)
    }

    std = {
        "cifar10": (0.2023, 0.1994, 0.2010),
        "cifar100": (0.2675, 0.2565, 0.2761),
        "tinyimagenet": (0.2302, 0.2265, 0.2262),
        "imagenet200": (0.229, 0.224, 0.225),
        "imagenet": (0.229, 0.224, 0.225),
        "clear10": (0.229, 0.224, 0.225),
        "clear100": (0.229, 0.224, 0.225)
    }
    

    classes = {
        "cifar10": 10,
        "cifar100": 100,
        "tinyimagenet": 200,
        "imagenet200": 200,
        "imagenet": 1000,
        "clear10": 10,
        "clear100": 100
    }

    in_channels = {
        "cifar10": 3,
        "cifar100": 3,
        "tinyimagenet": 3,
        "imagenet200": 3,
        "imagenet": 3,
        "clear10": 3,
        "clear100": 3,
    }

    inp_size = {
        "cifar10": 32,
        "cifar100": 32,
        "tinyimagenet": 64,
        "imagenet200": 224,
        "imagenet": 224,
        "clear10": 224,
        "clear100": 224,
    }
    return (
        mean[dataset],
        std[dataset],
        classes[dataset],
        inp_size[dataset],
        in_channels[dataset],
    )

def centering(K, device):
    n = K.shape[0]
    unit = torch.ones([n, n]).to(device)
    I = torch.eye(n).to(device)
    H = I - unit / n
    return torch.mm(torch.mm(H.float(), K.float()), H.float())  # HKH are the same with KH, KH is the first centering, H(KH) do the second time, results are the sme with one time centering
    # return torch.mm(H, K)  # KH

def rbf(X, sigma=None):
    GX = torch.mm(X, X.T)
    KX = torch.diag(GX) - GX + (torch.diag(GX) - GX).T
    if sigma is None:
        mdist = torch.median(KX[KX != 0])
        sigma = math.sqrt(mdist)
    KX *= - 0.5 / (sigma * sigma)
    KX = torch.exp(KX)
    return KX

def kernel_HSIC(X, Y, sigma):
    return torch.sum(centering(rbf(X, sigma)) * centering(rbf(Y, sigma)))

def linear_HSIC(X, Y, device):
    L_X = torch.mm(X, X.T)
    L_Y = torch.mm(Y, Y.T)
    return torch.sum(centering(L_X, device) * centering(L_Y, device))

# 10,512  / 10, 512 => 값
# 160*512 / 20 , 512 => 160 * 20 matrix
# 160 * 512 * 1 / 20 * 512 * 1 => 160 * 20

def linear_CKA(X, Y, device):
    hsic = linear_HSIC(X, Y, device)
    var1 = torch.sqrt(linear_HSIC(X, X, device))
    var2 = torch.sqrt(linear_HSIC(Y, Y, device))
    return hsic / (var1 * var2)


def kernel_CKA(X, Y, sigma=None):
    hsic = kernel_HSIC(X, Y, sigma)
    var1 = torch.sqrt(kernel_HSIC(X, X, sigma))
    var2 = torch.sqrt(kernel_HSIC(Y, Y, sigma))
    return hsic / (var1 * var2)

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

class GaussianTransform:
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        tmp = self.base_transform(x)
        x = torch.randn(tmp.shape)
        return x

class TwoOriginalTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        x_1 = self.base_transform(x)
        x_2 = self.base_transform(x)
        return [x_1, x_2]

class HardTransform:
    def __init__(self, base_transform, hard_type="rotate"):
        self.base_transform = base_transform
        self.hard_type = hard_type
        print("hard_type", self.hard_type)
        
    def shuffle(self, x, order=[0,1,2,3]):
        _, height, width = x.shape
        # For easy implementation, assume image width and height are even numbers.
        center_x = width // 2
        center_y = height // 2
        x1 = torch.zeros_like(x)
        
        x_split = [x[:, :center_y, :center_x], x[:, :center_y, center_x:], x[:, center_y:, :center_x], x[:, center_y:, center_x:]]
        
        x1[:, :center_y, :center_x] = x_split[order[0]]
        x1[:, :center_y, center_x:] = x_split[order[1]]
        x1[:, center_y:, :center_x] = x_split[order[2]]
        x1[:, center_y:, center_x:] = x_split[order[3]]
        
        return x1
    
    def __call__(self, x):
        if self.hard_type == "rotate":
            '''
            x1 = TF.rotate(self.base_transform(x), 90)
            x2 = TF.rotate(self.base_transform(x), 180)
            x3 = TF.rotate(self.base_transform(x), 270)
            '''
            x_base = self.base_transform(x)
            x1 = TF.rotate(x_base, 90)
            x2 = TF.rotate(x_base, 180)
            x3 = TF.rotate(x_base, 270)
            
        elif self.hard_type == "permute":
            '''
            rotate reverse clock wise
            1 3
            0 2
            '''
            x1 = self.shuffle(self.base_transform(x), order=[1, 3, 0, 2])
            
            '''
            rotate clock wise
            2 0
            3 1
            '''
            x2 = self.shuffle(self.base_transform(x), order=[2, 0, 3, 1])
            
            '''
            cross flip
            3 2
            1 0
            '''
            x3 = self.shuffle(self.base_transform(x), order=[3, 2, 1, 0])
        
        elif self.hard_type == "noise":
            gaussian_transform = GaussianTransform(self.base_transform)
            x1 = gaussian_transform(x)
            x2 = gaussian_transform(x)
            x3 = gaussian_transform(x)
            
            # x1 = self.add_noise(self.base_transform(x), 0.4)
            # x2 = self.add_noise(self.base_transform(x), 0.6)
            # x3 = self.add_noise(self.base_transform(x), 0.8)
            
        elif self.hard_type == "mix":
            gaussian_transform = GaussianTransform(self.base_transform)
            x1 = TF.affine(TF.rotate(self.base_transform(x), 90),angle=0, translate = [56, 0], scale=1.0, shear=0) 
            x2 = self.shuffle(self.base_transform(x), order=[1, 3, 0, 2])
            x3 = gaussian_transform(x)
                    
        else:
            raise NotImplementedError("You must choose one of [rotate, permute, noise, mix]")
        
        return [x1, x2, x3]
        
class TrainTransform:
    def __init__(self, base_transform, test_transform, method):
        self.base_transform = base_transform
        self.test_transform = test_transform
        self.method = method

    def __call__(self, x):
        x_0 = self.base_transform(x)
        x_1 = self.test_transform(x)
        if self.method == "xder":
            return x_0, x_1
        return x_0

class TestTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        x_0 = self.base_transform(x)
        x_90 = TF.rotate(self.base_transform(x), 90)
        return [x_0, x_90]

class MocoAugmentation:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        x_q = self.base_transform(x)
        x_k = TF.rotate(self.base_transform(x), 90)
        return [x_q, x_k]

def get_distance_matrix(feature_list, label_list):
    cls_feature_mean = []
    cls_list = []
    for label in torch.unique(label_list):
        idx = label_list == label
        label_name = "class" + str(label.item())    
        cls_feature_mean.append(torch.mean(feature_list[idx], dim=0))
        cls_list.append(label_name)
    
    dist_matrix = torch.cdist(torch.stack(cls_feature_mean), torch.stack(cls_feature_mean), p=2)
    print("cls_list")
    print(cls_list)
    print("dist_matrix")
    print(dist_matrix)
    

def plot_tsne(iteration, features, features_labels):
    feature_list = torch.cat(features).cpu()
    label_list = torch.cat(features_labels)
    print("feature_list", feature_list.shape)
    print("label_list", label_list.shape)
    color_list = ["violet", "limegreen", "orange","pink","blue","brown","red","grey","yellow","green"]
    tsne_model = TSNE(n_components=2)
    cluster = np.array(tsne_model.fit_transform(feature_list))
    plt.figure()
    for i in torch.unique(label_list):
        idx = label_list == i
        label = "class" + str(i)
        plt.scatter(cluster[idx.cpu().numpy(), 0], cluster[idx.cpu().numpy(), 1], marker='.', c=color_list[i], label=label)
        plt.legend()

    tsne_fig_name = str(iteration) + ".png"
    plt.savefig(tsne_fig_name)


@torch.no_grad()
def momentum_update_model(ema_model, model, ema_coeff):
    """
    Momentum update of the key encoder
    """ 
    for param_q, param_k in zip(model.parameters(), ema_model.parameters()):
        param_k.data = param_k.data * ema_coeff + param_q.data * (1.0 - ema_coeff)

def dot_regression_accuracy(preds, targets, topk=1, thr=0.):
    preds = preds.float()
    pred_scores, pred_labels = preds.topk(topk, dim=1)
    pred_labels = pred_labels.t()
    
    corrects = pred_labels.eq(targets.view(1, -1).expand_as(pred_labels))
    corrects = corrects & (pred_scores.t() > thr)
    return corrects.squeeze()

def generate_random_orthogonal_matrix(in_channel, num_classes):
    rand_mat = np.random.random(size=(in_channel, num_classes))
    orth_vec, _ = np.linalg.qr(rand_mat)
    orth_vec = torch.tensor(orth_vec).float()
    '''
    assert torch.allclose(torch.matmul(orth_vec.T, orth_vec), torch.eye(num_classes), atol=1.e-7), \
        "The max irregular value is : {}".format(
            torch.max(torch.abs(torch.matmul(orth_vec.T, orth_vec) - torch.eye(num_classes))))
    '''
    return orth_vec

def etf_initialize(in_channel, num_classes):
    orth_vec = generate_random_orthogonal_matrix(in_channel, num_classes)
    i_nc_nc = torch.eye(num_classes)
    one_nc_nc: torch.Tensor = torch.mul(torch.ones(num_classes, num_classes), (1 / num_classes))
    etf_vec = torch.mul(torch.matmul(orth_vec, i_nc_nc - one_nc_nc),
                        math.sqrt(num_classes / (num_classes - 1)))
    print("ETF Classifier Shape", etf_vec.shape)
    return etf_vec
'''

def etf_initialize(in_channel, num_classes = 1):
    orth_vec = generate_random_orthogonal_matrix(in_channel, num_classes)
    i_nc_nc = torch.eye(num_classes)
    one_nc_nc: torch.Tensor = torch.mul(torch.ones(num_classes, num_classes), (1 / num_classes))
    etf_vec = torch.mul(torch.matmul(orth_vec[:, :num_classes], i_nc_nc - one_nc_nc),
                        math.sqrt(num_classes / (num_classes - 1)))
    print("ETF Classifier Shape", etf_vec.shape)
    return orth_vec, etf_vec

def generate_random_orthogonal_matrix(in_channel, num_classes):
    rand_mat = np.random.random(size=(in_channel, in_channel))
    orth_vec, _ = np.linalg.qr(rand_mat)
    orth_vec = torch.tensor(orth_vec).float()
    assert torch.allclose(torch.matmul(orth_vec[:,:num_classes].T, orth_vec[:,:num_classes]), torch.eye(num_classes), atol=1.e-7), \
        "The max irregular value is : {}".format(
            torch.max(torch.abs(torch.matmul(orth_vec[:,:num_classes].T, orth_vec[:,:num_classes]) - torch.eye(num_classes))))
    return orth_vec
'''
def dynamic_etf_initialize(num_classes, orth_vec):
    i_nc_nc = torch.eye(num_classes)
    one_nc_nc: torch.Tensor = torch.mul(torch.ones(num_classes, num_classes), (1 / num_classes))
    etf_vec = torch.mul(torch.matmul(orth_vec[:, :num_classes], i_nc_nc - one_nc_nc),
                        math.sqrt(num_classes / (num_classes - 1)))
    return etf_vec

def normalize(x, mean, std):
    assert len(x.shape) == 4
    return (x - torch.tensor(mean).unsqueeze(0).unsqueeze(2).unsqueeze(3).to(x.device)) \
        / torch.tensor(std).unsqueeze(0).unsqueeze(2).unsqueeze(3).to(x.device)

def random_flip(x):
    assert len(x.shape) == 4
    mask = torch.rand(x.shape[0]) < 0.5
    x[mask] = x[mask].flip(3)
    return x

def random_grayscale(x, prob=0.2):
    assert len(x.shape) == 4
    mask = torch.rand(x.shape[0]) < prob
    x[mask] = (x[mask] * torch.tensor([[0.299, 0.587, 0.114]]).unsqueeze(
        2).unsqueeze(2).to(x.device)).sum(1, keepdim=True).repeat_interleave(3, 1)
    return x


class strong_aug():
    def __init__(self, size, mean, std):
        from torchvision import transforms
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(size=size, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8)
        ])
        self.mean = mean
        self.std = std

    def __call__(self, x):
        x.mul_(torch.tensor(self.std).view(3, 1, 1).cuda()).add_(torch.tensor(self.mean).view(3, 1, 1).cuda())
        flip = random_flip(x)
        return normalize(random_grayscale(
            torch.stack(
                [self.transform(a) for a in flip]
            )), self.mean, self.std)

