import torch
from .base_model import BaseModel
from . import networks
import time
from siamRPNPP import SiamRPNPP
from data_utils import normalize

cls_thres = 0.7


class AdATTACKModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):

        parser.set_defaults(norm='batch', net='search', dataset_mode='aligned')
        # replace is_train to True
        if is_train:
            parser.add_argument('--lambda_L2', type=float, default=700, help='weight for L1 loss')

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        if self.isTrain:
            self.model_names = ['A']
        else:
            self.model_names = ['A']
        self.netA = networks.define_NET(opt.net, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionL2 = torch.nn.MSELoss()
            # before
            # self.init_weight_L2 = opt.lambda_L2
            self.init_weight_L2 = 700
            self.init_weight_cls = 1
            self.init_weight_reg = 10
            self.cls_margin = -4
            self.side_margin1 = -5
            self.side_margin2 = -5
            self.weight_L2 = self.init_weight_L2
            self.weight_cls = self.init_weight_cls
            self.weight_reg = self.init_weight_reg
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_A = torch.optim.Adam(self.netA.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_A)
        else:
            self.criterionL2 = torch.nn.MSELoss()
            self.init_weight_L2 = 700
            self.init_weight_cls = 1
            self.init_weight_reg = 10
            self.cls_margin = -4
            self.side_margin1 = -5
            self.side_margin2 = -5
            self.weight_L2 = self.init_weight_L2
            self.weight_cls = self.init_weight_cls
            self.weight_reg = self.init_weight_reg
        '''siamrpn++'''
        self.siam = SiamRPNPP()
        self.siam_A = SiamRPNPP()

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        # self.template = input[0].squeeze(0).cuda()
        # need 4d input for networks in batch size
        self.template = input[0].squeeze(0).cuda()
        self.template = self.template.repeat(15, 1, 1, 1)  # template in batch size
        self.search_clean255 = input[1].squeeze(0).cuda()
        self.search_clean1 = normalize(self.search_clean255)
        self.zhanbi = input[2]

        self.num_search = self.search_clean1.size(0)

    # set input from adv dataset
    # def set_input_adv(self, input):
    #     self.search_attack255 = input[1].squeeze(0).cuda()
    #     self.search_attack1 = normalize(self.search_attack255)

    def set_input_adv(self, img_adv255):
        self.search_attack255 = img_adv255.squeeze(0).cuda()
        self.search_attack1 = normalize(self.search_attack255)

    def backward_A(self):
        self.loss_A_L2 = self.criterionL2(self.search_adv1, self.search_clean1) * self.weight_L2
        attention_mask = (self.score_maps_clean_o > cls_thres)
        num_attention = int(torch.sum(attention_mask))
        if num_attention > 0:
            score_map_adv_att = self.score_maps_adv[attention_mask]
            score_map_clean_att = self.score_maps_clean[attention_mask]
            reg_adv_att = self.reg_res_adv[2:4, attention_mask]
            self.loss_cls = (torch.mean(
                torch.clamp(score_map_adv_att[:, 1] - score_map_clean_att[:, 1], min=-3)) + torch.mean(
                torch.clamp(score_map_clean_att[:, 0] - score_map_adv_att[:, 0], min=-3))) * self.weight_cls
            self.loss_reg = (torch.mean(torch.clamp(reg_adv_att[0, :], min=self.side_margin1)) +
                             torch.mean(torch.clamp(reg_adv_att[1, :], min=self.side_margin2))) * self.weight_reg + (
                                    torch.argmax(score_map_adv_att[:, 1]) - torch.argmin(
                                score_map_clean_att[:, 0])) * 0.001
            self.loss_A = self.loss_A_L2 + self.loss_cls + self.loss_reg
        else:
            self.loss_A = self.loss_A_L2

        self.loss_A.backward()

    # def backward_R(self):
    #     self.loss_R_L2 = self.criterionL2(self.search_rec1, self.search_clean1) * self.weight_L2
    #     attention_mask = (self.score_maps_clean_o > cls_thres)
    #     num_attention = int(torch.sum(attention_mask))
    #     if num_attention > 0:
    #         score_map_rec_att = self.score_maps_rec[attention_mask]
    #         score_map_clean_att = self.score_maps_clean[attention_mask]
    #         reg_rec_att = self.reg_res_rec[0:4, attention_mask]
    #         reg_res_att_clean = self.reg_res_clean[0:4, attention_mask]
    #         self.loss_cls_R = (
    #                                 torch.mean(torch.square(
    #                                     torch.clamp(score_map_rec_att[:, 1] - score_map_clean_att[:, 1], min=-3)))
    #                                 + torch.mean(
    #                             torch.square(torch.clamp(score_map_clean_att[:, 0] - score_map_rec_att[:, 0], min=-3)))
    #                         ) * self.weight_cls
    #         # center point distance and size difference
    #
    #         rec_wh = reg_rec_att[2, :] * reg_rec_att[3, :]
    #         clean_wh = reg_res_att_clean[2, :] * reg_res_att_clean[3, :]
    #
    #         self.loss_reg_R = torch.mean(torch.square(reg_rec_att[0, :] - reg_res_att_clean[0, :]) + torch.square(
    #             reg_rec_att[1, :] - reg_res_att_clean[1, :])) \
    #                         + (torch.mean(torch.abs(rec_wh - clean_wh)))
    #
    #         self.loss_R = self.loss_R_L2 + self.loss_cls_R + self.loss_reg_R
    #     else:
    #         self.loss_R = self.loss_R_L2
    #
    #     self.loss_R.backward()

    def backward_R(self):
        self.loss_R_L2 = self.criterionL2(self.search_rec1, self.search_clean1) * self.weight_L2
        attention_mask = (self.score_maps_clean_o > cls_thres)
        num_attention = int(torch.sum(attention_mask))
        if num_attention > 0:
            score_map_rec_att = self.score_maps_rec[attention_mask]
            score_map_clean_att = self.score_maps_clean[attention_mask]
            reg_rec_att = self.reg_res_rec[0:4, attention_mask]
            reg_res_att_clean = self.reg_res_clean[0:4, attention_mask]
            self.loss_cls_R = (
                                    torch.mean(torch.abs(
                                        torch.clamp(score_map_rec_att[:, 1] - score_map_clean_att[:, 1], min=-3)))
                                    + torch.mean(
                                torch.abs(torch.clamp(score_map_clean_att[:, 0] - score_map_rec_att[:, 0], min=-3)))
                            ) * self.weight_cls
            # center point distance and size difference

            rec_wh = reg_rec_att[2, :] * reg_rec_att[3, :]
            clean_wh = reg_res_att_clean[2, :] * reg_res_att_clean[3, :]

            self.loss_reg_R = torch.mean(torch.square(reg_rec_att[0, :] - reg_res_att_clean[0, :]) + torch.square(
                reg_rec_att[1, :] - reg_res_att_clean[1, :])) \
                            + (torch.mean(torch.abs(rec_wh - clean_wh)))

            self.loss_R = self.loss_R_L2 + self.loss_cls_R + self.loss_reg_R
        else:
            self.loss_R = self.loss_R_L2

        self.loss_R.backward()

    # before target_sz = (255,255)
    def forward(self, target_sz=(287, 287)):
        if self.zhanbi < 0.002:
            block_num = 1
            search512_clean2 = torch.nn.functional.interpolate(self.search_clean1, size=(128, 128), mode='bilinear')
        elif self.zhanbi < 0.02:
            block_num = 2
            search512_clean2 = torch.nn.functional.interpolate(self.search_clean1, size=(64, 64), mode='bilinear')
        else:
            block_num = 3
            search512_clean2 = torch.nn.functional.interpolate(self.search_clean1, size=(32, 32), mode='bilinear')
        search512_adv1 = self.netA(search512_clean2, block_num)

        self.search_adv1 = torch.nn.functional.interpolate(search512_adv1, size=target_sz, mode='bilinear')
        self.search_adv255 = self.search_adv1 * 127.5 + 127.5

    def forward_R(self, target_sz=(287, 287)):
        if self.zhanbi < 0.002:
            block_num = 1
            search512_attack2 = torch.nn.functional.interpolate(self.search_attack1, size=(128, 128), mode='bilinear')
        elif self.zhanbi < 0.02:
            block_num = 2
            search512_attack2 = torch.nn.functional.interpolate(self.search_attack1, size=(64, 64), mode='bilinear')
        else:
            block_num = 3
            search512_attack2 = torch.nn.functional.interpolate(self.search_attack1, size=(32, 32), mode='bilinear')
        search512_rec1 = self.netA(search512_attack2, block_num)

        self.search_rec1 = torch.nn.functional.interpolate(search512_rec1, size=target_sz, mode='bilinear')
        self.search_rec255 = self.search_rec1 * 127.5 + 127.5

    def transform(self, patch_clean1, target_sz=(287, 287)):
        patch512_clean1 = torch.nn.functional.interpolate(patch_clean1, size=(512, 512), mode='bilinear')
        patch512_adv1 = patch512_clean1 + self.netA(patch512_clean1)  # Residual form: G(A)+A
        patch_adv1 = torch.nn.functional.interpolate(patch512_adv1, size=target_sz, mode='bilinear')
        patch_adv255 = patch_adv1 * 127.5 + 127.5
        return patch_adv255

    def optimize_parameters(self):
        with torch.no_grad():
            self.siam.model.template(self.template)
            self.score_maps_clean_o = self.siam.get_heat_map(self.search_clean255, softmax=True)  # (5HWN,),with softmax
        self.forward()
        self.score_maps_adv, self.reg_res_adv = self.siam.get_cls_reg(self.search_adv255,
                                                                      softmax=False)  # (5HWN,2)without softmax,(5HWN,4)
        self.score_maps_clean, self.reg_res_clean = self.siam.get_cls_reg(self.search_clean255, softmax=False)
        self.optimizer_A.zero_grad()
        self.backward_A()
        self.optimizer_A.step()

    def optimize_parameters_R(self):
        with torch.no_grad():
            self.siam.model.template(self.template)
            self.score_maps_clean_o = self.siam.get_heat_map(self.search_clean255, softmax=True)  # (5HWN,),with softmax
        self.forward_R()
        self.score_maps_rec, self.reg_res_rec = self.siam.get_cls_reg(self.search_rec255,
                                                                      softmax=False)  # (5HWN,2)without softmax,(5HWN,4)
        self.score_maps_clean, self.reg_res_clean = self.siam.get_cls_reg(self.search_clean255, softmax=False)
        self.optimizer_A.zero_grad()
        self.backward_R()
        self.optimizer_A.step()

    def calculate_loss(self):
        self.loss_A_L2 = self.criterionL2(self.search_adv1, self.search_clean1) * self.weight_L2
        attention_mask = (self.score_maps_clean_o > cls_thres)
        num_attention = int(torch.sum(attention_mask))
        if num_attention > 0:
            score_map_adv_att = self.score_maps_adv[attention_mask]
            score_map_clean_att = self.score_maps_clean[attention_mask]
            reg_adv_att = self.reg_res_adv[2:4, attention_mask]
            self.loss_cls = (torch.mean(
                torch.clamp(score_map_adv_att[:, 1] - score_map_clean_att[:, 1], min=-3)) + torch.mean(
                torch.clamp(score_map_clean_att[:, 0] - score_map_adv_att[:, 0], min=-3))) * self.weight_cls
            self.loss_reg = (torch.mean(torch.clamp(reg_adv_att[0, :], min=self.side_margin1)) +
                             torch.mean(torch.clamp(reg_adv_att[1, :], min=self.side_margin2))) * self.weight_reg + (
                                    torch.argmax(score_map_adv_att[:, 1]) - torch.argmin(
                                score_map_clean_att[:, 0])) * 0.001
            self.loss_A = self.loss_A_L2 + self.loss_cls + self.loss_reg
        else:
            self.loss_A = self.loss_A_L2

        return self.loss_A

    # before target_sz=(255, 255)
    def get_loss(self, target_sz=(287, 287)):
        with torch.no_grad():
            self.siam.model.template(self.template)
            self.score_maps_clean_o = self.siam.get_heat_map(self.search_clean255, softmax=True)  # (5HWN,),with softmax
        self.forward(target_sz)
        self.score_maps_adv, self.reg_res_adv = self.siam.get_cls_reg(self.search_adv255,
                                                                      softmax=False)  # (5HWN,2)without softmax,(5HWN,4)
        self.score_maps_clean, self.reg_res_clean = self.siam.get_cls_reg(self.search_clean255, softmax=False)
        loss = self.calculate_loss()
        return loss
