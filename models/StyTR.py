import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)
from function import normal,normal_style
from function import calc_mean_std
import scipy.stats as stats
from models.ViT_helper import DropPath, to_2tuple, trunc_normal_


# img is divided into patch_size x patch_size
# patch is projected into a embed_dim using self.proj _ a convolutional layer
# output i a tensor containing all the patch embeddings
class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=256, patch_size=8, in_chans=3, embed_dim=512):
        super().__init__()
        # turn it into 2 tuple, H and W will be treated as separate values.
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        # W / H
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        # Conv2d(3, 512, 8, 8)
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        B, C, H, W = x.shape
        """
        B: batch size, C: number of input channels
        Standard tensor shape
        """
        # use a proj (conv) layer to reduce the spatial dimensions
        x = self.proj(x)

        return x


decoder = nn.Sequential(
  # add 1 pixel to each side using reflection
    nn.ReflectionPad2d((1, 1, 1, 1)),
    # 2dconvolution with 256 filters 
    # reduce feature depth
    nn.Conv2d(512, 256, (3, 3)),
    # replace negative with 0
    nn.ReLU(),
    # upsmaple_ double the size
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 128, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 64, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 3, (3, 3)),

    # summary: 512 features channels into 3 only
    
)

vgg = nn.Sequential(
    # standarize the input
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    # conv 3 to 64 channels
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1, this is the last layer used
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU()  # relu5-4
)

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        # hidden dim: number of neurons in the hidden layers.
        # total how many nerons in total of layers
        h = [hidden_dim] * (num_layers - 1)
        # create a list of Linear layers is a fully connected layer with n input neurons and k output neurons
        # ModuleList to track the layers
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        # uses linear layer first then Relu
        # every hidden layers have Relu
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
    #
    """
    Relu will create a non-linear boundaries and prevent vanishing GD because postive input = 1
    Relu also help the sparsity cause negative = 0 => generalization
    """


class StyTrans(nn.Module):
    """ This is the style transform transformer module """
    
    def __init__(self,encoder,decoder,PatchEmbed, transformer,args):

        super().__init__()
        enc_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1
        self.enc_5 = nn.Sequential(*enc_layers[31:44])  # relu4_1 -> relu5_1
        
        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4', 'enc_5']:
            for param in getattr(self, name).parameters():
              # requrires gradient to compute or not for tracking the G for backpropagation
                param.requires_grad = False

        # still using MSE
        self.mse_loss = nn.MSELoss()
        self.transformer = transformer
        hidden_dim = transformer.d_model       
        self.decode = decoder
        self.embedding = PatchEmbed

    def encode_with_intermediate(self, input):
      # results[0] is simply the input imgs
        results = [input]
        for i in range(5):
            # get the layer names
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            # process the output of the previous layer 
            results.append(func(results[-1]))
        return results[1:]

    def calc_content_loss(self, input, target):
      # assert is just a combine of if and except
      assert (input.size() == target.size())
      assert (target.requires_grad is False)
      return self.mse_loss(input, target)

    def calc_style_loss(self, input, target):
        assert (input.size() == target.size())
        # target is fixed and not trainable
        assert (target.requires_grad is False)
        # calculate mean and standard deviation (do lech chuan) of a tensor 
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + \
               self.mse_loss(input_std, target_std)

    def forward(self, samples_c: NestedTensor,samples_s: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

        """
        # nestedtensors stores a batch of tensors with potentially different shapes while maintaing individual dims.
        content_input = samples_c
        style_input = samples_s

        """
        tensor: is a mathematical object, are a multi-dimensional arrays of numbers that represent complex data
        """
        if isinstance(samples_c, (list, torch.Tensor)):
            samples_c = nested_tensor_from_tensor_list(samples_c)  # support different-sized images padding is used for mask [tensor, mask] 
        if isinstance(samples_s, (list, torch.Tensor)):
            samples_s = nested_tensor_from_tensor_list(samples_s)
        
        """
        so it will encode using vgg19
        then it will process in each layer untill relu_4 then return results of every layers
        """
        content_feats = self.encode_with_intermediate(samples_c.tensors)
        style_feats = self.encode_with_intermediate(samples_s.tensors)

        ### Linear projection to make it a tensor, make it smaller and increase the channels
        # repair to become dataset to train
        style = self.embedding(samples_s.tensors)
        content = self.embedding(samples_c.tensors)
        
        # postional embedding is calculated in transformer.py
        pos_s = None
        pos_c = None

        mask = None
        # call transformer.py, create a entity
        hs = self.transformer(style, mask , content, pos_c, pos_s)
           
        Ics = self.decode(hs)

        Ics_feats = self.encode_with_intermediate(Ics)
        # compute 2 last layer features of the content img and trasformed img, normalize everything
        loss_c = self.calc_content_loss(normal(Ics_feats[-1]), normal(content_feats[-1]))+self.calc_content_loss(normal(Ics_feats[-2]), normal(content_feats[-2]))
        # Style loss
        loss_s = self.calc_style_loss(Ics_feats[0], style_feats[0])
        for i in range(1, 5):
            loss_s += self.calc_style_loss(Ics_feats[i], style_feats[i])
            
        
        Icc = self.decode(self.transformer(content, mask , content, pos_c, pos_c))
        Iss = self.decode(self.transformer(style, mask , style, pos_s, pos_s))    

        #Identity losses lambda 1    
        loss_lambda1 = self.calc_content_loss(Icc,content_input)+self.calc_content_loss(Iss,style_input)
        
        #Identity losses lambda 2
        Icc_feats=self.encode_with_intermediate(Icc)
        Iss_feats=self.encode_with_intermediate(Iss)
        loss_lambda2 = self.calc_content_loss(Icc_feats[0], content_feats[0])+self.calc_content_loss(Iss_feats[0], style_feats[0])
        for i in range(1, 5):
            loss_lambda2 += self.calc_content_loss(Icc_feats[i], content_feats[i]) + self.calc_content_loss(Iss_feats[i], style_feats[i])
        # Please select and comment out one of the following two sentences
        return Ics,  loss_c, loss_s, loss_lambda1, loss_lambda2   #train
        # return Ics    #test 