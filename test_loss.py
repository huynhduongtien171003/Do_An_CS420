import argparse
from pathlib import Path
import os
import torch
import torch.nn as nn
from PIL import Image
from os.path import basename
from os.path import splitext
from torchvision import transforms
from torchvision.utils import save_image
from function import calc_mean_std, normal, coral
import models.transformer as transformer
import models.StyTR as StyTR
import matplotlib.pyplot as plt
from matplotlib import cm
from function import normal
import numpy as np
import time
def test_transform(size, crop):
    transform_list = []
   
    if size != 0: 
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform
def style_transform(h,w):
    k = (h,w)
    size = int(np.max(k))
    print(type(size))
    transform_list = []    
    transform_list.append(transforms.CenterCrop((h,w)))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform

def content_transform():
    
    transform_list = []   
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform

  

parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--content', type=str,
                    help='File path to the content image')
parser.add_argument('--content_dir', type=str,
                    help='Directory path to a batch of content images')
parser.add_argument('--style', type=str,
                    help='File path to the style image, or multiple style \
                    images separated by commas if you want to do style \
                    interpolation or spatial control')
parser.add_argument('--style_dir', type=str,
                    help='Directory path to a batch of style images')
parser.add_argument('--output', type=str, default='output',
                    help='Directory to save the output image(s)')
parser.add_argument('--vgg', type=str, default='./experiments/vgg_normalised.pth')
parser.add_argument('--decoder_path', type=str, default='./experiments/decoder_iter_160000.pth')
parser.add_argument('--Trans_path', type=str, default='./experiments/transformer_iter_160000.pth')
parser.add_argument('--embedding_path', type=str, default='./experiments/embedding_iter_160000.pth')


parser.add_argument('--style_interpolation_weights', type=str, default="")
parser.add_argument('--a', type=float, default=1.0)
parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
parser.add_argument('--hidden_dim', default=512, type=int,
                        help="Size of the embeddings (dimension of the transformer)")


args = parser.parse_args()




# Advanced options
content_size=512
style_size=512
crop='store_true'
save_ext='.jpg'
output_path=args.output
preserve_color='store_true'
alpha=args.a


device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

# Either --content or --content_dir should be given.
if args.content:
    content_paths = [Path(args.content)]
else:
    content_dir = Path(args.content_dir)
    content_paths = [f for f in content_dir.glob('*')]

# Either --style or --style_dir should be given.
if args.style:
    style_paths = [Path(args.style)]    
else:
    style_dir = Path(args.style_dir)
    style_paths = [f for f in style_dir.glob('*')]

if not os.path.exists(output_path):
  os.mkdir(output_path)


vgg = StyTR.vgg
vgg.load_state_dict(torch.load(args.vgg, weights_only=False))
vgg = nn.Sequential(*list(vgg.children())[:44])

decoder = StyTR.decoder
Trans = transformer.Transformer()
embedding = StyTR.PatchEmbed()

decoder.eval()
Trans.eval()
vgg.eval()
from collections import OrderedDict
new_state_dict = OrderedDict()
state_dict = torch.load(args.decoder_path, weights_only=False)
for k, v in state_dict.items():
    #namekey = k[7:] # remove `module.`
    namekey = k
    new_state_dict[namekey] = v
decoder.load_state_dict(new_state_dict)

new_state_dict = OrderedDict()
state_dict = torch.load(args.Trans_path, weights_only=False)
for k, v in state_dict.items():
    #namekey = k[7:] # remove `module.`
    namekey = k
    new_state_dict[namekey] = v
Trans.load_state_dict(new_state_dict)

new_state_dict = OrderedDict()
state_dict = torch.load(args.embedding_path, weights_only=False)
for k, v in state_dict.items():
    #namekey = k[7:] # remove `module.`
    namekey = k
    new_state_dict[namekey] = v
embedding.load_state_dict(new_state_dict)

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda:0" if USE_CUDA else "cpu")

network = StyTR.StyTrans(vgg,decoder,embedding,Trans,args)
network.eval()
network.to(device)

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

content_tf = test_transform(content_size, crop)
style_tf = test_transform(style_size, crop)


total_content_loss = 0.0  # Initialize total content loss accumulator
total_style_loss = 0.0

for content_path in content_paths:
    for style_path in style_paths:
        print(content_path)
        print("Loading content image:", content_path)
        print("Loading style image:", style_path)

        content_tf1 = content_transform()       
        content = content_tf(Image.open(content_path).convert("RGB"))

        h, w, c = np.shape(content)
        style_tf1 = style_transform(h, w)
        style = style_tf(Image.open(style_path).convert("RGB"))

        style = style.to(device).unsqueeze(0)
        content = content.to(device).unsqueeze(0)

        with torch.no_grad():
            # Forward pass through the network
            output, loss_c, loss_s, *_ = network(content, style)

            # Accumulate the content loss for all processed images
            total_content_loss += loss_c.item()
            total_style_loss += loss_s.item()

            # Print output shape
            print("Output tensor shape:", output.shape)

        output = output.cpu()

        # Save the stylized output image
        output_name = '{:s}/{:s}_stylized_{:s}{:s}'.format(
            output_path, splitext(basename(content_path))[0],
            splitext(basename(style_path))[0], save_ext
        )

        print("Saving output image to:", output_name)
        if output is not None:
            save_image(output, output_name)
            print("Image saved successfully!")
        else:
            print("Output image tensor is None.")

# After processing all images, print the total content loss
print("Total content loss:", total_content_loss)
print("Total style loss:", total_style_loss)
print("Content images found:", [str(path) for path in content_paths])
print("Style images found:", [str(path) for path in style_paths])

