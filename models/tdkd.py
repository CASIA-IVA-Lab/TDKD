from models.fdm import FeatureDistillMask
from models.distill_loss import FeatLoss, ProbLoss
import torch.nn as nn


def conv_dw(inp, oup, stride):
    return nn.Sequential(
#        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
#        nn.BatchNorm2d(inp),
#        nn.ReLU(inp),
#
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
#        nn.BatchNorm2d(oup),
#        nn.ReLU(oup),
    )

class Adapt(nn.Module):
	def __init__(self, inchannels=64):
		super(Adapt, self).__init__()
		self.trans = conv_dw(inchannels, inchannels, 1)

	def forward(self, x):
		x = self.trans(x)
		return x


class TDKD(nn.Module):
	def __init__(self):
		super(TDKD, self).__init__()
		self.feat_loss = FeatLoss()
		self.prob_loss = ProbLoss()
		self.fdm = FeatureDistillMask(2, 3, 2)
		self.stage = 3
		self.adpt_loc_list = [Adapt() for i in range(self.stage)]
		self.adpt_cls_list = [Adapt() for i in range(self.stage)]

	def forward(self, stu_output, tea_output, priors_by_layer, priors, targets, pos, neg):

		_, _, _, stu_loc_feat_list, stu_cls_feat_list, stu_loc_list, stu_cls_list, stu_prob = stu_output
		_, tea_prob, _, tea_loc_feat_list, tea_cls_feat_list, tea_loc_list, tea_cls_list = tea_output

		for i in range(self.stage):
			adap_loc = self.adpt_loc_list[i].cuda()
			stu_loc_feat_list[i] = adap_loc(stu_loc_feat_list[i])

			adap_cls = self.adpt_cls_list[i].cuda()
			stu_cls_feat_list[i] = adap_cls(stu_cls_feat_list[i])


	
		loc_list = []
		cls_list = []
		num = int(pos.shape[0])
		#print("pos.shape=", pos.shape)

		pos_part1 = pos[:, 0:12800]
		pos_part2 = pos[:, 12800:16000]
		pos_part3 = pos[:, 16000:]

		cls_part1 = (neg[:, 0:12800] + pos_part1) > 0
		cls_part2 = (neg[:, 12800:16000] + pos_part2) > 0
		cls_part3 = (neg[:, 16000:] + pos_part3) > 0

		loc_list.append(pos_part1.view(num, -1, 2).sum(dim = 2) > 0)
		loc_list.append(pos_part2.view(num, -1, 2).sum(dim = 2) > 0)
		loc_list.append(pos_part3.view(num, -1, 2).sum(dim = 2) > 0)

		cls_list.append(cls_part1.view(num, -1, 2).sum(dim = 2) > 0)
		cls_list.append(cls_part2.view(num, -1, 2).sum(dim = 2) > 0)
		cls_list.append(cls_part3.view(num, -1, 2).sum(dim = 2) > 0)


		#loc_mask_list, cls_mask_list = self.fdm.run(stu_loc_list, stu_cls_list, priors_by_layer, priors, targets)
		feat_loss = self.feat_loss(stu_loc_feat_list, stu_cls_feat_list, tea_loc_feat_list, tea_cls_feat_list, loc_list, cls_list)
		#print("feat_loss=", feat_loss)

		#for i in range(self.stage):
		#	print("loc_mask_list[i].shape=", loc_mask_list[i].shape)
		#	print("loc_list[i].shape=", loc_list[i].shape)
		prob_loss = self.prob_loss(stu_prob, tea_prob)
		#exit()
		return feat_loss, prob_loss

