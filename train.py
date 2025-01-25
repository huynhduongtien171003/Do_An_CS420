import argparse
import os
import torch
import torch.nn as nn
import torch.utils.data as data
from PIL import Image
from PIL import ImageFile
from tensorboardX import SummaryWriter
from torchvision import transforms
from tqdm import tqdm
from pathlib import Path
import models.transformer as transformer
import models.StyTR  as StyTR 
from sampler import InfiniteSamplerWrapper
from torchvision.utils import save_image

# resize img to 512x512


def train_transform():
    transform_list = [
        transforms.Resize(size=(512, 512)),
        transforms.RandomCrop(256),
        # convert into PyTorch tensor shape ( C, H, W) and normalized pixel values
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)


class FlatFolderDataset(data.Dataset):
    def __init__(self, root, transform):
        super(FlatFolderDataset, self).__init__()
        self.root = root
        print(self.root)
        self.path = os.listdir(self.root)
    #   root/
        # folder 1/
        # ├── folder1/
        # │   ├── file1.jpg
        # │   ├── file2.jpg
    #     folder2/
        # ├── folder2/
        # │   ├── file3.jpg
        # │   ├── file4.jpg
        
        if os.path.isdir(os.path.join(self.root, self.path[0])):
            self.paths = []
            for folder_name in os.listdir(self.root):
                subfolder_path = os.path.join(self.root, folder_name)  # First-level folder
                if os.path.isdir(subfolder_path):  # Ensure it's a directory
                    for subfolder_name in os.listdir(subfolder_path):  # Second-level folder
                        subsubfolder_path = os.path.join(subfolder_path, subfolder_name)
                        if os.path.isdir(subsubfolder_path):  # Ensure it's a directory
                            for file_name in os.listdir(subsubfolder_path):  # Iterate files
                                self.paths.append(os.path.join(subsubfolder_path, file_name))
        else:
            self.paths = list(Path(self.root).glob('*'))


        self.transform = transform

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(str(path)).convert('RGB')
        img = self.transform(img)
        return img
    def __len__(self):
        return len(self.paths)
    def name(self):
        return 'FlatFolderDataset'

def adjust_learning_rate(optimizer, iteration_count):
    """Imitating the original implementation"""
    # args.lr_decay: control how quickly the learning rate decrease (defy elsewhere in the code)
    lr = 2e-4 / (1.0 + args.lr_decay * (iteration_count - 1e4))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def warmup_learning_rate(optimizer, iteration_count):
    """Imitating the original implementation"""
    lr = args.lr * 0.1 * (1.0 + 3e-4 * iteration_count)
    # print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--content_dir', default='/kaggle/input/coco-image-caption/train2014/train2014', type=str,   
                    help='Directory path to a batch of content images')

parser.add_argument('--style_dir', default='/kaggle/input/wikiart/Abstract_Expressionism', type=str,  #wikiart dataset crawled from https://www.wikiart.org/
                    help='Directory path to a batch of style images')
parser.add_argument('--vgg', type=str, default='/kaggle/input/styr_2/pytorch/default/2/StyTR-2/StyTR-2/experiments/vgg_normalised.pth')  #run the train.py, please download the pretrained vgg checkpoint

# training options
parser.add_argument('--save_dir', default='./experiments',
                    help='Directory to save the model')
parser.add_argument('--log_dir', default='./logs',
                    help='Directory to save the log')
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--lr_decay', type=float, default=1e-5)
parser.add_argument('--max_iter', type=int, default=160000)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--style_weight', type=float, default=10.0)
parser.add_argument('--content_weight', type=float, default=7.0)
parser.add_argument('--n_threads', type=int, default=16)
parser.add_argument('--save_model_interval', type=int, default=10000)

# use sine and cosing to encode the position
# also there are 2 ways to choose the embedding technique
parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
parser.add_argument('--hidden_dim', default=512, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
args = parser.parse_args()

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda:0" if USE_CUDA else "cpu")

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

if not os.path.exists(args.log_dir):
    os.mkdir(args.log_dir)
# logs data for TensorBoard to visualize
writer = SummaryWriter(log_dir=args.log_dir)

vgg = StyTR.vgg
# load weight from a file - i think it is vgg_normalised.pth
vgg.load_state_dict(torch.load(args.vgg))
# create a list of layer form vgg and take only first 44 layers
# and then create a new nn.Sequential model containing 44 layers
vgg = nn.Sequential(*list(vgg.children())[:44])

# split img into patch and project it into patch embedding tensor 
decoder = StyTR.decoder
embedding = StyTR.PatchEmbed()

Trans = transformer.Transformer()

# disable gradient tracking, create network model not for backpropagation
with torch.no_grad():
  # vgg_feature extraction, decoder_to take split, embedding_img into patches, trans_transformer, args_additional configuration
    network = StyTR.StyTrans(vgg,decoder,embedding, Trans,args)
network.train()

# this to move model to device
network.to(device)

# to show the GPU for the device
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

# to run parallel cpu if device has
network = nn.DataParallel(network, device_ids=[0,1])

# transform the content and style 
# it just resized the images in content and style
content_tf = train_transform()
style_tf = train_transform()

content_dataset = FlatFolderDataset(args.content_dir, content_tf)
style_dataset = FlatFolderDataset(args.style_dir, style_tf)

# DataLoader is a utility class for efficient loading
# help to get more control over the iteration process
content_iter = iter(data.DataLoader(
    content_dataset, batch_size=args.batch_size,
    sampler=InfiniteSamplerWrapper(content_dataset),
    num_workers=args.n_threads))
style_iter = iter(data.DataLoader(
    style_dataset, batch_size=args.batch_size,
    sampler=InfiniteSamplerWrapper(style_dataset),
    num_workers=args.n_threads))
"""
InfiniteSamplerWrapper: 
    If we use RandomSampler or SequentialSampler might run out of data because it can stop after full pass over dataset
    But Infinite help to loop repeatedly
"""

# make sure they will be updated during training
# using optimizer Adam 
# suite for large dataset and large parameters, work well with CNN
optimizer = torch.optim.Adam([ 
                              {'params': network.module.transformer.parameters()},
                              {'params': network.module.decode.parameters()},
                              {'params': network.module.embedding.parameters()},        
                              ], lr=args.lr)



if not os.path.exists(args.save_dir+"/test"):
    os.makedirs(args.save_dir+"/test")

# tqdm provide a progress bar for the loop. 
for i in tqdm(range(args.max_iter)):
    # first 10k, warm_up
    if i < 1e4:
        warmup_learning_rate(optimizer, iteration_count=i)
    # after 10k adjust
    else:
        adjust_learning_rate(optimizer, iteration_count=i)

    # print('learning_rate: %s' % str(optimizer.param_groups[0]['lr']))
    # fetch the next batch from their iterators
    content_images = next(content_iter).to(device)
    style_images = next(style_iter).to(device)  

    # out will store the output of network
    # l_identity lossses to preserve properties during transfer
    # StyTR.StyTrans = network 
    out, loss_c, loss_s,l_identity1, l_identity2 = network(content_images, style_images)

    # save intermediate output
    # concatinate content and style vertically
    # combine multiple tensor 
    if i % 100 == 0:
        output_name = '{:s}/test/{:s}{:s}'.format(
                        args.save_dir, str(i),".jpg"
                    )
        out = torch.cat((content_images,out),0)
        out = torch.cat((style_images,out),0)
        save_image(out, output_name)

    # calculate loss 
    loss_c = args.content_weight * loss_c
    loss_s = args.style_weight * loss_s
    loss = loss_c + loss_s + (l_identity1 * 70) + (l_identity2 * 1) 
  
    print(loss.sum().cpu().detach().numpy(),"-content:",loss_c.sum().cpu().detach().numpy(),"-style:",loss_s.sum().cpu().detach().numpy()
              ,"-l1:",l_identity1.sum().cpu().detach().numpy(),"-l2:",l_identity2.sum().cpu().detach().numpy()
              )
       
    optimizer.zero_grad()
    loss.sum().backward()
    optimizer.step()

    writer.add_scalar('loss_content', loss_c.sum().item(), i + 1)
    writer.add_scalar('loss_style', loss_s.sum().item(), i + 1)
    writer.add_scalar('loss_identity1', l_identity1.sum().item(), i + 1)
    writer.add_scalar('loss_identity2', l_identity2.sum().item(), i + 1)
    writer.add_scalar('total_loss', loss.sum().item(), i + 1)

    if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.max_iter:
        state_dict = network.module.transformer.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict,
                   '{:s}/transformer_iter_{:d}.pth'.format(args.save_dir,
                                                           i + 1))

        state_dict = network.module.decode.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict,
                   '{:s}/decoder_iter_{:d}.pth'.format(args.save_dir,
                                                           i + 1))
        state_dict = network.module.embedding.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict,
                   '{:s}/embedding_iter_{:d}.pth'.format(args.save_dir,
                                                           i + 1))

                                                    
writer.close()


