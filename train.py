from __future__ import print_function
import os
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import argparse
import torch.utils.data as data
from data import WiderFaceDetection, detection_collate, preproc, cfg
from layers.modules import MultiBoxLoss
from layers.functions.prior_box import PriorBox
import time
import datetime
import math
from models.retinaface import RetinaFace
from models.tdkd import TDKD
from models.retinaface_width import RetinaFace as RetinaFaceWidth

parser = argparse.ArgumentParser(description='Retinaface Training')
parser.add_argument('--training_dataset', default='./data/widerface/train/label.txt', help='Training dataset directory')
parser.add_argument('-b', '--batch_size', default=32, type=int, help='Batch size for training')
parser.add_argument('--num_workers', default=4, type=int, help='Number of workers used in dataloading')
parser.add_argument('--ngpu', default=1, type=int, help='gpus')
parser.add_argument('--lr', '--learning-rate', default=2e-2, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--resume_net', default='./base.pth', help='resume net for retraining')
parser.add_argument('--resume_epoch', default=0, type=int, help='resume iter for retraining')
parser.add_argument('-max', '--max_epoch', default=80, type=int, help='max epoch for retraining')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
parser.add_argument('--save_folder', default='./weights/', help='Location to save checkpoint models')
parser.add_argument('--img_dim', default=640, help='Input shape when training')
parser.add_argument('--mode', type=str, required=True, help='training mode: distillation/teacher/student')



args = parser.parse_args()


if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

rgb_mean = (104, 117, 123) # bgr order
num_classes = 2
img_dim = args.img_dim
num_gpu = args.ngpu
num_workers = args.num_workers
batch_size = args.batch_size
momentum = args.momentum
weight_decay = args.weight_decay
initial_lr = args.lr
gamma = args.gamma
max_epoch = args.max_epoch
training_dataset = args.training_dataset
save_folder = args.save_folder
gpu_train = cfg['gpu_train']
mode = args.mode


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


net = None
if mode == 'teacher':
    net = RetinaFaceWidth()
    args.resume_net = None
elif mode == 'student':
    net = RetinaFace()
    args.resume_net = None
elif mode == 'distillation':
    net = RetinaFace()
else:
    print("Input mode does not exit.")
    exit()

teacher_net = RetinaFaceWidth(phase='test')
teacher_net = load_model(teacher_net, './Final_Retinaface_width.pth', False)



#print("Printing net...")
#print(net)

if args.resume_net is not None:
    print('Loading resume network...')
    state_dict = torch.load(args.resume_net)
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        head = k[:7]
        if head == 'module.':
            name = k[7:] # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)

if num_gpu > 1 and gpu_train:
    net = torch.nn.DataParallel(net, device_ids=list(range(num_gpu)))
    teacher_net = torch.nn.DataParallel(teacher_net, device_ids=list(range(num_gpu)))

device = torch.device('cuda:0' if gpu_train else 'cpu')
cudnn.benchmark = True
net = net.to(device)
teacher_net = teacher_net.to(device)

optimizer = optim.SGD(net.parameters(), lr=initial_lr, momentum=momentum, weight_decay=weight_decay)
criterion = MultiBoxLoss(num_classes, 0.35, True, 0, True, 7, 0.35, False)
tdkd = TDKD()
priorbox = PriorBox(cfg, image_size=(img_dim, img_dim))
with torch.no_grad():
    priors, priors_by_layer = priorbox.forward()
    priors = priors.to(device)



def train():
    net.train()
    teacher_net.eval()
    for param in teacher_net.parameters():
        param.requires_grad = False

    epoch = 0 + args.resume_epoch
    print('Loading Dataset...')

    dataset = WiderFaceDetection( training_dataset,preproc(img_dim, rgb_mean))

    epoch_size = math.ceil(len(dataset) / batch_size)
    max_iter = max_epoch * epoch_size

    stepvalues = (55 * epoch_size, 68 * epoch_size)
    step_index = 0

    if args.resume_epoch > 0:
        start_iter = args.resume_epoch * epoch_size
    else:
        start_iter = 0

    for iteration in range(start_iter, max_iter):
        if iteration % epoch_size == 0:
            # create batch iterator
            batch_iterator = iter(data.DataLoader(dataset, batch_size, shuffle=True, num_workers=num_workers, collate_fn=detection_collate))
            if (epoch % 10 == 0 and epoch > 0) or (epoch % 5 == 0 and epoch > 200):
                torch.save(net.state_dict(), save_folder + 'Retinaface_epoch_' + str(epoch) + '.pth')
            epoch += 1

        load_t0 = time.time()
        if iteration in stepvalues:
            step_index += 1
        lr = adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size)

        # load train data
        images, targets = next(batch_iterator)
        images = images.to(device)
        targets = [anno.to(device) for anno in targets]

        # forward
        out = net(images)
        teacher_out = teacher_net(images)



        # backprop
        optimizer.zero_grad()
        loss_l, loss_c, loss_landm, pos, neg = criterion(out, priors, targets)

        #distillation
        feat_loss, prob_loss = tdkd(out, teacher_out, priors_by_layer, priors, targets, pos, neg)

        d_feat = 1.0

        scale = loss_l.detach() / feat_loss.detach()
        d_prob = 50


        if mode == 'teacher' or mode == 'student':
            d_feat = 0
            d_prob = 0
            scale = 0


        loss = cfg['loc_weight'] * loss_l + loss_c + loss_landm + d_feat * scale * feat_loss + d_prob * prob_loss
        loss.backward()
        optimizer.step()
        load_t1 = time.time()
        batch_time = load_t1 - load_t0
        eta = int(batch_time * (max_iter - iteration))
        print('Epoch:{}/{} || Epochiter: {}/{} || Iter: {}/{} || Loc: {:.4f} Cla: {:.4f} Landm: {:.4f} Dfeat: {:.4f} Dprob: {:.4f}|| LR: {:.8f} || Batchtime: {:.4f} s || ETA: {}'
              .format(epoch, max_epoch, (iteration % epoch_size) + 1,
              epoch_size, iteration + 1, max_iter, loss_l.item(), loss_c.item(), loss_landm.item(), feat_loss.item() * d_feat, prob_loss.item() * d_prob, lr, batch_time, str(datetime.timedelta(seconds=eta))))

    torch.save(net.state_dict(), save_folder + 'Final_Retinaface.pth')


def adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size):
    """Sets the learning rate
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    warmup_epoch = -1
    if epoch <= warmup_epoch:
        lr = 1e-6 + (initial_lr-1e-6) * iteration / (epoch_size * warmup_epoch)
    else:
        lr = initial_lr * (gamma ** (step_index))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

if __name__ == '__main__':
    train()
