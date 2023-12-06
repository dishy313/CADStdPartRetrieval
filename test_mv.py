# for test
import os
import numpy as np
import torch
from torch.autograd import Variable

from MVSketchANet import SketchSubNet
from MVSketchANet import SketchMViewTripletNet
from compute_mAp import CalcTopMap
from compute_mAp import CalcTopAcc
from compute_mAp import CalcMap
from CustomDataSet_weights import performLoadTestData


root = 'dataSet\\StdModel-20Class'  # dataset
data_path = 'dataSet\\StdModel-20Class\\datas'  # view image path
sketch_path = 'dataSet\\StdModel-20Class\\input-sketch'  # free-sketch path

num_views = 12
img_size = 225
class_num = 6

currentModelName = 'finalModel'  # model folder
currentModelPath = 'out/{}/SketchNet_MV_Weights_660.pth'.format(currentModelName)  # model path
outFolderPath = 'out_feat/{}'.format(currentModelName)  # output path(log results)


def commitFreeSketchFeature():
    os.makedirs('out_feat', exist_ok=True)
    os.makedirs(outFolderPath, exist_ok=True)

    curDataPath = data_path + '\\' + 'test'
    curLabel = 'testImageLabel'

    cur_init_dataset = performLoadTestData(img_size, root, curDataPath, curLabel, num_views)

    # init net
    net = SketchSubNet(num_feat=100, num_views=num_views)
    model = SketchMViewTripletNet(net)
    # load net
    model.load_state_dict(torch.load(currentModelPath))
    model = model.cuda()
    # val/test
    model.eval()

    out = np.empty((len(cur_init_dataset.sketch_datapath), net.num_feat), dtype=np.float32)
    out_cls_name = []
    out_cls_num = []
    out_path = []
    out_name = []

    for i in range(len(cur_init_dataset.sketch_datapath)):
        sketch_src, sketch_info = cur_init_dataset.getFreeSketchItem(i)
        sketch = Variable(sketch_src.unsqueeze(0)).cuda()
        feat = model.get_sketch_feat(sketch)
        feat = feat.cpu().data.numpy()
        out[i] = feat
        cls_name = sketch_info[0]
        cls_num = sketch_info[1]
        path = sketch_info[2]
        name = sketch_info[3]
        out_cls_name.append(cls_name)
        out_cls_num.append(cls_num)
        out_path.append(path)
        out_name.append(name)
        print(f"[{i}/{len(cur_init_dataset.sketch_datapath)}] ('{cls_name}', '{cls_num}', '{name}') completed")

    # save results(sketch feature)
    np.savez(os.path.join(outFolderPath, 'feat_sketch.npz'), feat_sketch=out, cls_name=out_cls_name, cls_num=out_cls_num,
             path=out_path, name=out_name)
    print('commitFreeSketchFeature done!')


def commitMultiViewImageFeature():
    os.makedirs('out_feat', exist_ok=True)
    os.makedirs(outFolderPath, exist_ok=True)

    curDataPath = data_path + '\\' + 'test'
    curLabel = 'testImageLabel'

    cur_init_dataset = performLoadTestData(img_size, root, curDataPath, curLabel, num_views)

    # init net
    branch_net = SketchSubNet(num_feat=100, num_views=num_views)  # for photography edge
    model = SketchMViewTripletNet(branch_net)

    # loda net
    model.load_state_dict(torch.load(currentModelPath))
    model = model.cuda()
    model.eval()

    out = np.empty((len(cur_init_dataset), branch_net.num_feat), dtype=np.float32)
    out_cls_name = []
    out_cls_num = []
    out_path = []
    out_name = []

    for i in range(len(cur_init_dataset)):
        imgs_src, img_edge_info = cur_init_dataset.getMutilViewImageItem(i)

        V, C, H, W = imgs_src.size()
        img_edge_src = Variable(imgs_src).view(-1, C, H, W).cuda()

        feat = model.get_Image_feat(img_edge_src)
        feat = feat.cpu().data.numpy()
        out[i] = feat
        cls_name = img_edge_info[0]
        cls_num = img_edge_info[1]
        path = img_edge_info[2]
        name = img_edge_info[3]
        out_cls_name.append(cls_name)
        out_cls_num.append(cls_num)
        out_path.append(path)
        out_name.append(name)
        print(f"[{i}/{len(cur_init_dataset)}] ('{cls_name}', '{cls_num}', '{name}') completed")

    # save results(image feature)
    np.savez(os.path.join(outFolderPath, 'feat_photo.npz'), feat=out, cls_name=out_cls_name, cls_num=out_cls_num, path=out_path,
             name=out_name)
    print('commitMultiViewImageFeature done!')

# compute final mAP
def commit_map_infos():
    feat_photo_path = 'out_feat/{}/feat_photo.npz'.format(currentModelName)
    feat_sketch_path = 'out_feat/{}/feat_sketch.npz'.format(currentModelName)
    feat_photo = np.load(feat_photo_path)
    feat_sketch = np.load(feat_sketch_path)

    feat_s = feat_sketch['feat_sketch']
    cls_num_s = feat_sketch['cls_num']

    feat_p = feat_photo['feat']
    cls_num_p = feat_photo['cls_num']

    out_cls_num_s = np.empty((len(feat_s), 1), dtype=np.int_)
    for ii in range(len(feat_s)):
        out_cls_num_s[ii] = int(cls_num_s[ii])

    out_cls_num_p = np.empty((len(feat_p), 1), dtype=np.int_)
    for ii in range(len(feat_p)):
        out_cls_num_p[ii] = int(cls_num_p[ii])

    topk = 5

    topkmap = CalcTopMap(feat_s, feat_p, out_cls_num_s, out_cls_num_p, topk)
    print("map@{}为：{}".format(topk, topkmap))

    map = CalcMap(feat_s, feat_p, out_cls_num_s, out_cls_num_p)
    print("map为：{}".format(map))

    topkacc = CalcTopAcc(feat_s, feat_p, out_cls_num_s, out_cls_num_p, topk)
    print("acc@{}为：{}".format(topk, topkacc))



if __name__ == '__main__':
    commitFreeSketchFeature()
    commitMultiViewImageFeature()
    commit_map_infos()
