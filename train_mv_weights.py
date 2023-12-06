# for train
import os
import torch.nn as nn
import time
import torch
import numpy as np
from torch.autograd import Variable
from torch.utils import tensorboard as tb

from MVSketchANet import SketchMViewTripletNet
from MVSketchANet import SketchSubNet
from CustomDataSet_weights import performTrainLoader
from retriveImageFeatures import compute_sketch_feat
from retriveImageFeatures import compute_image_feat
from retriveImageFeatures import computeCustomMAP

# define training parameters
batchSize = 8  # net  batchSize
niter = 1  # net iteration nums
checkpoint_val_step = 1  # net validation  step
checkpoint = 50  # save net model step
img_size = 225  # image size (width*height)
custom_view_nums = 12  # projection of view nums
class_num = 7  # class nums (+1)

outFolderName = 'out\\stdModel_20Class_SketchMV_'  # output folder
curTimeStr = time.strftime('%y%m%d%H%M', time.localtime(time.time()))
outFolderName = outFolderName + curTimeStr
pre_model_path = "preModel\\pre_MV_Esb3D_380.pth"

root = 'dataSet\\StdModel-20Class'  # dataset
data_path = 'dataSet\\StdModel-20Class\\datas'  # view image path
sketch_path = 'dataSet\\StdModel-20Class\\input-sketch'  # free-sketch path
curLogPath = root + '\\logs\\' + curTimeStr  # set tensorboard log path


# random net init params
def feed_random_seed(seed=np.random.randint(1, 10000)):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


# perform net training
def train(epochItem, dataloader, net, optItem, criterionItem):
    accum_loss = 0
    net.train()
    for i, (img_a, imgList_p, imgList_n) in enumerate(dataloader):
        # get input sample(iterm)
        # 1.anchor input
        anc_src = Variable(img_a.cuda())

        # 2.positive input
        N, V, C, H, W = imgList_p.size()
        pos_src = Variable(imgList_p).view(-1, C, H, W).cuda()

        # 3.negetive input
        N, V, C, H, W = imgList_n.size()
        neg_src = Variable(imgList_n).view(-1, C, H, W).cuda()

        net.zero_grad()
        feat_a, feat_p, feat_n = net(anc_src, pos_src, neg_src)

        # compute triple loss
        loss = criterionItem(feat_a, feat_p, feat_n)

        # BP Loss
        loss.backward()
        # traing net
        optItem.step()
        accum_loss += loss.data.item()

        print(f'[{epochItem}][{i}/{len(dataloader)}] loss: {loss.data.item():.4f}')
    return accum_loss / len(dataloader)


if __name__ == '__main__':
    # init params
    os.makedirs('out', exist_ok=True)
    os.makedirs(outFolderName, exist_ok=True)
    os.makedirs(curLogPath, exist_ok=True)
    feed_random_seed()

    # load training dataset
    train_loader = performTrainLoader(batchSize, img_size, custom_view_nums, root, data_path, 'trainImageLabel')

    # init net
    net = SketchSubNet(num_feat=100, num_views=custom_view_nums)
    model = SketchMViewTripletNet(net)

    # load pre-training model
    if os.path.exists(pre_model_path):
        model.load_state_dict(torch.load(pre_model_path))
    model = model.cuda()

    # SGD training
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, weight_decay=0.0005, momentum=0.9)

    # compute L2 distance as loss
    criterion = nn.TripletMarginLoss(margin=1.0, p=2.0)

    # Tensorboard view
    writer = tb.SummaryWriter(curLogPath)
    resume_epoch = 0

    # print training results
    fid = open('{}/loss_and_accurary.txt'.format(outFolderName), 'a')

    best_precise = 0.
    # train
    for epoch in range(resume_epoch + 1, niter + 1):
        # step1
        train_loss = train(epoch, train_loader, model, optimizer, criterion)

        # loss
        print('[%d/%d] train loss: %f' % (epoch, niter, train_loss))
        fid.write('[%d/%d] train loss: %f\n' % (epoch, niter, train_loss))
        writer.add_scalar('train-loss', train_loss, epoch)

        # step2  validation  and save the best model
        if epoch > 0 and (epoch % checkpoint_val_step == 0 or epoch == niter - 1):
            curDataPath = data_path + '\\' + 'val'
            curLabel = 'valImageLabel'

            feat_s, cls_name_s, cls_num_s, path_s, name_s = compute_sketch_feat(root, curDataPath, curLabel, model, net.num_feat)

            feat_p, cls_name_p, cls_num_p, path_p, name_p = compute_image_feat(root, curDataPath, curLabel, model, net.num_feat)

            curMAP, curMapWeights = computeCustomMAP(feat_s, name_s, feat_p, cls_num_p, class_num)
            # change weights from mAP after validation, and select a best negative input for training net
            train_loader.dataset.setWeights(curMapWeights)

            print('[%d/%d] mAp : %.3f' % (epoch, niter, curMAP))
            fid.write('[%d/%d] mAp: %.3f\n' % (epoch, niter, curMAP))
            writer.add_scalar('mAp', curMAP, epoch)

            # save the best mAP  net
            if best_precise < curMAP:
                best_precise = curMAP
                torch.save(model.state_dict(), os.path.join(outFolderName, f'{epoch:03d}_best.pth'))

        # step3. save net
        if epoch % checkpoint == 0:
            torch.save(model.state_dict(), os.path.join(outFolderName, f'{epoch:03d}.pth'))

    # exit
    fid.close()
    print(f'best mAP:{best_precise}')