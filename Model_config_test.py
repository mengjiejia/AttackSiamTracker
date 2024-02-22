from pix2pix.options.test_options1 import TestOptions
from pix2pix.models import create_model
import os
from Setting import project_path
opt = TestOptions().parse()

opt.model = 'AdATTACK'
opt.net = 'search'

AdA = create_model(opt)
# model provided by authors
# AdA.load_path = os.path.join(project_path,'checkpoints/%s/model.pth'%opt.model)
# my trained model
AdA.load_path = os.path.join(project_path,'checkpoints/%s/30_net_A_attack.pth'%opt.model)
AdA.setup(opt)
AdA.eval()

# print('AdA.load_path', AdA.load_path)
