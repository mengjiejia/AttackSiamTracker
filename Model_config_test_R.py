from pix2pix.options.test_options1 import TestOptions
from pix2pix.models import create_model
import os
from Setting import project_path
opt = TestOptions().parse()

opt.model = 'AdATTACK'
opt.net = 'search'

AdA_R = create_model(opt)
# model provided by authors
# AdA.load_path = os.path.join(project_path,'checkpoints/%s/model.pth'%opt.model)
# my trained model
AdA_R.load_path = os.path.join(project_path,'checkpoints/%s/50_net_R.pth'%opt.model)
AdA_R.setup(opt)
AdA_R.eval()

# print('AdA.load_path', AdA.load_path)
