"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import os
from pathlib import Path
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES']='0' # 0 is the 2080ti
#os.environ['CUDA_VISIBLE_DEVICES']=''
import torch

from utils import *
import argparse
from torch.autograd import Variable
from trainer import MUNIT_Trainer, UNIT_Trainer
import torch.backends.cudnn as cudnn
import torch

import socket
computer_name = socket.gethostname()

try:
    from itertools import izip as zip
except ImportError: # will be 3.x series
    pass
import os
import sys
import tensorboardX
import shutil
import utils

torch.cuda.empty_cache()

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/handwriting.yaml', help='Path to the config file.')
parser.add_argument('--output_path', type=str, default='.', help="outputs path")
parser.add_argument("--resume", action="store_true")
parser.add_argument('--trainer', type=str, default='MUNIT', help="MUNIT|UNIT")
parser.add_argument('--check_files', action="store_true")

opts = parser.parse_args()

if computer_name=="Galois":
    opts.config = 'configs/handwriting_online.yaml'
    opts.check_files = True
    print("Running on Galois, config {}, check_files {}".format(opts.config, opts.check_files))

cudnn.benchmark = True

# Load experiment setting
config = get_config(opts.config)
print(config)
max_iter = config['max_iter']
display_size = config['display_size']
config['vgg_model_path'] = opts.output_path

# Setup model and data loader
if opts.trainer == 'MUNIT':
    trainer = MUNIT_Trainer(config)
elif opts.trainer == 'UNIT':
    trainer = UNIT_Trainer(config)
else:
    sys.exit("Only support MUNIT|UNIT")

#trainer.cuda()

#train_loader_a, train_loader_b, test_loader_a, test_loader_b, folders = get_all_data_loaders(config)
(train_loader_a, tr_a), (train_loader_b, tr_b), (test_loader_a, test_a), (test_loader_b, test_b), folders = get_all_data_loaders_better(config)

if opts.check_files and False:
    print("Checking files...")
    for folder in folders:
        print(folder)
        utils.check_files(folder)
    print("Done checking files.")

# Setup logger and output folders
model_name = os.path.splitext(os.path.basename(opts.config))[0]

log_dir = utils.increment_path(name="Run", base_path=os.path.join(opts.output_path + "/logs", model_name), make_directory=True)
train_writer = tensorboardX.SummaryWriter(log_dir)
output_directory = os.path.join(opts.output_path + "/outputs", model_name)
output_directory = utils.increment_path(name="Run", base_path=output_directory, make_directory=True)
checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml')) # copy config file to output folder

# Start training
iterations = trainer.resume(checkpoint_directory, hyperparameters=config) if opts.resume else 0

output_directory = Path("./output")
output_directory.mkdir(parents=True, exist_ok=True)

for it, (images_a, images_b) in enumerate(zip(train_loader_a, train_loader_b)):
    _it = 2*it
    trainer.update_learning_rate()
    update_images_a, update_images_b = images_a.cuda().detach(), images_b.cuda().detach()

    with torch.no_grad():
        #print(images_a.shape, images_b.shape)
        #image_outputs = trainer.sample(images_a, images_b)

    for image in images_a:
        save_path_a = output_directory / Path(tr_a[_it][1]).name
        utils.save_image(save_path_a, output_directory)
        _it += 1
    #save_path_b = output_directory / Path(tr_b[it][1]).name
    #utils.save_image(save_path_b, output_directory)

