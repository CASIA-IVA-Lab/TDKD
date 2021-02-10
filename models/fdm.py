import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils.box_utils import log_sum_exp, jaccard, point_form
from data import WiderFaceDetection, detection_collate, preproc, cfg
import torch.utils.data as data
from models.retinaface import RetinaFace
from layers.functions.prior_box import PriorBox

GPU = cfg['gpu_train']



def match(truths, priors, all_priors, mask_t, idx):

    iou_map = jaccard(point_form(priors), truths)
    iou_map_global = jaccard(point_form(all_priors), truths)
    feature_size = int(iou_map.shape[0])
    max_iou, _ = torch.max(iou_map_global, dim = 0)
    mask_per_img = torch.zeros([feature_size], dtype=torch.int64).cuda()
    
    for k in range(truths.shape[0]):
        if torch.sum(truths[k]) == 0.:
            break
        #if max_iou[k] < 0.2:
        #    continue
        max_iou_per_gt = 0.35 #max_iou[k] * 0.5
        #mask_per_gt = torch.sum((iou_map[:,k] > max_iou_per_gt).view(feature_size, num_anchors), dim=1)
        mask_per_gt = iou_map[:,k] > max_iou_per_gt
        mask_per_gt = mask_per_gt.long()
        mask_per_img += mask_per_gt
    
    mask_per_img = mask_per_img > 0
    mask_t[idx] = mask_per_img

class FeatureDistillMask:

    def __init__(self, num_classes, neg_pos, num_anchors):
        super(FeatureDistillMask, self).__init__()
        self.num_classes = num_classes
        self.negpos_ratio = neg_pos
        self.num_anchors = num_anchors

    def run(self, bbox_regressions_list, classifications_list, priors, all_priors, targets):

        cls_mask_list = []
        loc_mask_list = []
        
        for i in range(len(bbox_regressions_list)):
            prior = priors[i].cuda()
            bbox_regression = bbox_regressions_list[i]
            conf_data = classifications_list[i]
            num = bbox_regression.size(0)
            num_priors = int(prior.size(0))


            mask_t = torch.LongTensor(num, num_priors)


            for idx in range(num):
                truths = targets[idx][:, :4].data
                defaults = prior.data
                match(truths, defaults, all_priors.cuda().data, mask_t, idx)
            if GPU:
                mask_t = mask_t.cuda()


            # Compute max conf across batch for hard negative mining
            batch_conf = conf_data.view(-1, self.num_classes)
            loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, mask_t.view(-1, 1))

            zeros = torch.tensor(0).cuda()
            pos = mask_t > zeros

            # Hard Negative Mining
            loss_c[pos.view(-1, 1)] = 0 # filter out pos boxes for now
            loss_c = loss_c.view(num, -1)
            _, loss_idx = loss_c.sort(1, descending=True)
            _, idx_rank = loss_idx.sort(1)

            num_pos = pos.long().sum(1, keepdim=True)
            num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
            neg = idx_rank < num_neg.expand_as(idx_rank)


            cls_mask = torch.sum((pos + neg).view(num, -1, self.num_anchors), 2) > 0
            loc_mask = torch.sum(pos.view(num, -1, self.num_anchors), 2) > 0

            cls_mask_list.append(cls_mask.detach())
            loc_mask_list.append(loc_mask.detach())

        return loc_mask_list, cls_mask_list




if __name__ == '__main__':

    num_classes = 2
    fdm = FeatureDistillMask(num_classes, 3, 2)

    training_dataset = './data/widerface/train/label.txt'
    rgb_mean = (104, 117, 123) # bgr order
    img_dim = 640
    batch_size = 4
    dataset = WiderFaceDetection( training_dataset,preproc(img_dim, rgb_mean))
    batch_iterator = iter(data.DataLoader(dataset, batch_size, shuffle=True, num_workers=1, collate_fn=detection_collate))
    net = RetinaFace().cuda()

    priorbox = PriorBox(cfg, image_size=(img_dim, img_dim))
    with torch.no_grad():
        priors, priors_by_layer = priorbox.forward()
        priors = priors.cuda()


    for iteration in range(0, 1):

        # load train data
        images, targets = next(batch_iterator)
        images = images.cuda()
        targets = [anno.cuda() for anno in targets]

        # forward
        out = net(images)
        loc_mask_list, cls_mask_list = fdm.run(out, priors_by_layer, targets)
