from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatLoss(nn.Module):

	def __init__(self):
		super(FeatLoss, self).__init__()


	def forward(self, stu_loc_feat_list, stu_cls_feat_list, tea_loc_feat_list, tea_cls_feat_list, loc_mask_list, cls_mask_list):

		stage = len(stu_cls_feat_list)

		cls_distill_loss = 0
		loc_distill_loss = 0

		#all_loc = 0
		for i in range(stage):
			b, c, h, w = tea_cls_feat_list[i].shape
			mask_cls = cls_mask_list[i].unsqueeze(2).repeat(1, 1, c).permute(0, 2, 1)
			mask_loc = loc_mask_list[i].unsqueeze(2).repeat(1, 1, c).permute(0, 2, 1)
			N_cls = max(1, mask_cls.sum() * 2 // c) * 1.0
			N_loc = max(1, mask_loc.sum() * 2 // c) * 1.0
			cls_distill_loss += (torch.pow(stu_cls_feat_list[i].view(b, c, -1) - tea_cls_feat_list[i].view(b, c, -1), 2) * mask_cls.float()).sum() / N_cls
			loc_distill_loss += (torch.pow(stu_loc_feat_list[i].view(b, c, -1) - tea_loc_feat_list[i].view(b, c, -1), 2) * mask_loc.float()).sum() / N_loc
			#all_loc = all_loc + (torch.pow(stu_loc_feat_list[i].view(b, c, -1) - tea_loc_feat_list[i].view(b, c, -1), 2)).sum()

		feat_distill_loss = cls_distill_loss + loc_distill_loss
		return feat_distill_loss



class ProbLoss(nn.Module):

	def __init__(self):
		super(ProbLoss, self).__init__()


	def forward(self, p_s, p_t):


		prob_mask = self.get_prob_mask(p_t)
		N_pos = max(1, prob_mask.sum()) * 1.0
		valid_p_s = p_s[prob_mask].view(-1)
		valid_p_t = p_t[prob_mask].view(-1)

		loss = (valid_p_t - valid_p_s).pow(2)
		loss = loss.sum() / N_pos

		return loss



	def get_prob_mask(self, p_t):
		mask = (p_t > 0.1) * (p_t < 0.9)
		return mask
