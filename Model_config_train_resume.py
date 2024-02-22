from pix2pix.options.train_options1 import TrainOptions
from pix2pix.models import create_model
import os
from Setting import project_path
import torch
opt = TrainOptions().parse()

opt.model = 'AdATTACK'
opt.net = 'search'
opt.continue_train = True

AdA_R = create_model(opt)
AdA_R.load_path = os.path.join(project_path,'checkpoints/%s/20_net_R.pth'%opt.model)
# AdA_R.load_state_dict(torch.load(os.path.join(project_path,'checkpoints/%s/20_net_R.pth'%opt.model)))
AdA_R.setup(opt)
AdA_R.train()


# print('AdA.load_path', AdA.load_path)
