# if working on mac, uncomment this
#import os
#os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch # root package
import torch.nn as nn # neural network module
import torch.nn.functional as F # layers, activations
import torch.optim as optim # optimizer

from PIL import Image # Pillow Iamge Library

# torchvision package contains popular datasets,
# model architectures and common image transformations
import torch # root package
import torch.nn as nn # neural network module
import torch.nn.functional as F # layers, activations
import torch.optim as optim # optimizer

from PIL import Image # Pillow Iamge Library

# torchvision package contains popular datasets,
# model architectures and common image transformations
import torchvision.transforms as transforms
import torchvision.models as models

import copy
import numpy as np
import matplotlib.pyplot as plt

import sys
import argparse

################################## Step 1: Read User Input ##################################

parser = argparse.ArgumentParser(description='Neural Style Transfer')
parser.add_argument('--content', help='enter path for content image', dest='content', type=str)
parser.add_argument('--style', help='enter path for style image', dest='style', type=str)
parser.add_argument('--alpha', help='weight associated with content', dest='alpha', type=int)
parser.add_argument('--beta', help='weight associated with style', dest='beta', type=int)
parser.add_argument('--size', help='image size of squared output', dest='size', type=int)
parser.add_argument('--iter', help='number of iterations for training', dest='iter', type=int)
args = parser.parse_args()
content = args.content
style = args.style
alpha = args.alpha
beta = args.beta
img_size = args.size
# default setting
if alpha is None:
    alpha = 1
if beta is None:
    beta = 80000
if img_size is None:
    img_size = 512
iteration = args.iter
if iteration is None:
    iteration = 2000

################################## Step 2: Preprocess Images ##################################

# chain together several image transformations
# this resize will normalize imgae from 0-255 to 0-1
resize = transforms.Compose([
        # if height > width, then image will be rescaled to (size * height / width, size)
        transforms.Resize(img_size), # rescaling
        transforms.CenterCrop(img_size), 
        transforms.ToTensor(),
    ])

def img_transform(img_path):
    img = Image.open(img_path).convert('RGB') # return a <class 'PIL.Image.Image'>
    img = resize(img).unsqueeze(0)
    return img # within 0 and 1

def img_detransform(img_path):
    img = img_path.cpu().clone()
    img = img.squeeze(0)
    img = transforms.ToPILImage()(img)
    plt.imshow(img)
    return img

################################## Step 3: Model Architecture ##################################

#device = 'cuda' if torch.cuda.is_available() else 'cpu'

# define a layer that does the normalization so that we can 
# add it to our sequential to streamline the training process
class NormLayer(nn.Module):
    def __init__(self):
        super(NormLayer, self).__init__()
        self.mu = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)

    def forward(self, img):
        return (img - self.mu) / self.std

# define the content layer to get the target content and calculate content loss
class ContentLayer(nn.Module):
    
    def __init__(self, p):
        super(ContentLayer, self).__init__()
        self.p = p.detach()
    
    def forward(self, x):
        self.loss = nn.MSELoss()(x, self.p)
        return x

# define the style layer to get the target style and calculate style loss
class StyleLayer(nn.Module):
    
    def gram_matrix(self, activation):
        # batch, channel, height, width
        B, Nl, H, W = activation.shape
        Ml = H * W
        # number of filter Nl times length of activation
        activation = activation.view(Nl, Ml)
        # multiply the transpose to get the gram matrix
        G = torch.mm(activation, activation.t())
        # the coeff part in the formula in paper
        G_norm = G.div(Nl*Ml)
        return G_norm
    
    def __init__(self, A_activation):
        super(StyleLayer, self).__init__()
        self.A = self.gram_matrix(A_activation)
        self.A = self.A.detach()
    
    def forward(self, G_activation):
        G = self.gram_matrix(G_activation)
        self.E = nn.MSELoss()(G, self.A)
        return G_activation

# define NST model architecture
def NST(vgg19, content_img, style_img, input_img, device):
    
    # layers chosen from paper, can experiment with other options
    
    # content: conv4_2
    content_layer_list = ['conv_22']
    # style: conv1_1, conv2_1, conv3_1, conv4_1, conv5_1
    style_layer_list = ['conv_1', 'conv_6', 'conv_11', 'conv_20', 'conv_29']

    # normalize the layer everytime we start a new training iteration
    net = nn.Sequential(NormLayer())
    
    # get the original layers in pretrained model
    # and add the content, style layers
    count = 1
    for layer in vgg19.features:
        if isinstance(layer, nn.Conv2d):
            rename = 'conv_' + str(count)
        elif isinstance(layer, nn.ReLU):
            rename = 'relu_' + str(count)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            rename = 'pool_' + str(count)
        elif isinstance(layer, nn.BatchNorm2d):
            rename = 'bn_' + str(count)
        count += 1
        
        net.add_module(rename,layer)

        # get the target for both content and style
        if rename in content_layer_list:
            net.to(device)
            p = net(content_img).detach()
            content_layer = ContentLayer(p)
            net.add_module('content_' + str(count),content_layer)
        if rename in style_layer_list:
            net.to(device)
            A_activation = net(style_img).detach()
            style_layer = StyleLayer(A_activation)
            net.add_module('style_' + str(count), style_layer)
            if rename == 'conv_29':
                break
 
    return net

if __name__ == '__main__':
    
    ################################## Step 4: Training ##################################
    print ('Start Training...\n')
    print ('Settings:')
    print ('* alpha =', alpha)
    print ('* beta =', beta)
    print ('* image size =', img_size)
    print ('* iteration =', iteration)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print ('* device =', device)
    print ('\n')
    
    # image transformation
    content_img = img_transform(content).to(device)
    style_img = img_transform(style).to(device)
    input_img = torch.randn(content_img.data.size(), device=device, requires_grad=True)
    
    # call pretrained model
    vgg19 = models.vgg19(pretrained=True).to(device)

    net = NST(vgg19, content_img, style_img, input_img, device)
    for p in net.parameters():
        p.requires_grad = False

    optimizer = optim.LBFGS([input_img.requires_grad_()])

    i = [0]
    while i[0] <= iteration:
        def closure():
            # values outside of 0 and 1 will not be valid pixel in image setting
            input_img.data.clamp_(0, 1)
            
            content_loss = 0
            style_loss = []
            
            optimizer.zero_grad()
            net(input_img)
            
            for layer in net:
                if isinstance(layer, ContentLayer):
                    content_loss = layer.loss
                elif isinstance(layer, StyleLayer):
                    style_loss.append(layer.E)
            style_loss = sum(style_loss)
            
            loss = content_loss * alpha + style_loss * beta
            loss.backward()
            
            if i[0] % 100 == 0:
                print ('Iteration: {}'.format(i[0]))
                print ('Loss: {:4f}'.format(loss))
                print('[Content Loss] {:4f} [Style Loss] {:4f}'.format(
                    content_loss * alpha, style_loss * beta))
                print ('\n')

            i[0] += 1
            
            return loss
        optimizer.step(closure)
    print ('Finished Training!')

    ################################## Step 5: Visualization ##################################
    input_img.data.clamp_(0, 1) 
    outcome = img_detransform(input_img)
    outcome.save('outcome.jpg')