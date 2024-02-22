from pix2pix.options.train_options1 import TrainOptions
from pix2pix.models import create_model
import os
from Setting import project_path
opt = TrainOptions().parse()

opt.model = 'AdATTACK'
opt.net = 'search'

# recovery model training
# AdA_R = create_model(opt)
# AdA.load_path = os.path.join(project_path,'checkpoints/%s/model.pth'%opt.model)
# AdA_R.setup(opt)

# attack model training
AdA = create_model(opt)
# AdA.load_path = os.path.join(project_path,'checkpoints/%s/model.pth'%opt.model)
AdA.setup(opt)

# print('AdA.load_path', AdA.load_path)
