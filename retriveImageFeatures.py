# extract image and sketch feature from net

import numpy as np
from torch.autograd import Variable
from CustomDataSet_weights import performLoadValData

num_views = 12
img_size = 225


# extract image feature from net
def compute_image_feat(path_cur, data_path_cur, cur_label, model, num_feat):
    # loda dataset
    cur_dataset = performLoadValData(img_size, path_cur, data_path_cur, cur_label, num_views)
    model.eval()
    out = np.empty((len(cur_dataset), num_feat), dtype=np.float32)

    out_cls_name = []
    out_cls_num = []
    out_path = []
    out_name = []

    for i in range(len(cur_dataset)):
        imgs_src, img_edge_info = cur_dataset.getMutilViewImageItem(i)

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
    return out, out_cls_name, out_cls_num, out_path, out_name


# extract sketch feature from net
def compute_sketch_feat(path_cur, data_path_cur, cur_label, model, num_feat):
    # loda dataset
    cur_dataset = performLoadValData(img_size, path_cur, data_path_cur, cur_label, num_views)

    model.eval()
    out = np.empty((len(cur_dataset.sketch_datapath), num_feat), dtype=np.float32)

    out_cls_name = []
    out_cls_num = []
    out_path = []
    out_name = []

    for i in range(len(cur_dataset.sketch_datapath)):
        sketch_src, sketch_info = cur_dataset.getFreeSketchItem(i)

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
    return out, out_cls_name, out_cls_num, out_path, out_name

# compute mAP value and get a weights from mAP
def computeCustomMAP(input_feat_s, input_name_s, input_feat_p, input_cls_num_p, input_class_num):
    feat_s = np.array(input_feat_s)
    name_s = np.array(input_name_s)
    feat_p = np.array(input_feat_p)
    cls_num_p = np.array(input_cls_num_p)
    AP = []
    mapClassNumsNeg = {}
    mapNegWeights = {}

    for i in range(1, input_class_num):
        tmpListValue = []
        tmpListValueDef = []
        for mm in range(1, input_class_num):
            if mm != i:
                tmpListValue.append(str(mm))
                tmpListValueDef.append(0)
        mapClassNumsNeg[str(i)] = tmpListValue
        mapNegWeights[str(i)] = tmpListValueDef

    countNum = len(feat_s)
    avgNums = int(countNum / input_class_num)
    for i in range(0, countNum):
        sketch = name_s[i]
        dist_l2 = np.sqrt(np.sum(np.square(feat_s[i] - feat_p), 1))
        order = np.argsort(dist_l2)

        order_cls_num_p = cls_num_p[order]

        # compute weights
        curPorcessClass = int(sketch)
        curListNeg = mapClassNumsNeg[str(curPorcessClass)]
        curTmpIndexValue = 0
        for val in order_cls_num_p:
            if str(curPorcessClass) != val:
                # current class num
                curThisIndex = curListNeg.index(val)
                curTmpIndexValue += 1
                mapNegWeights[str(curPorcessClass)][curThisIndex] += 1 - (curTmpIndexValue / len(order_cls_num_p))

        # mAP
        curSketchIndex = int(sketch)
        curNumsAll = 0
        for index in range(0, len(cls_num_p)):
            x = cls_num_p[index]
            curVal = int(x)
            if curVal == curSketchIndex:
                curNumsAll = curNumsAll + 1

        curNumsTP = 0
        for mm in range(0, curNumsAll):
            x = order_cls_num_p[mm]
            curVal = int(x)
            if curVal == curSketchIndex:
                curNumsTP = curNumsTP + 1

        pValue = curNumsTP / curNumsAll
        ap = pValue
        AP.append(pValue)

    # set the value of  weights
    for i in range(1, input_class_num):
        itemValue = mapNegWeights[str(i)]
        new_list = [item / avgNums for item in itemValue]
        mapNegWeights[str(i)] = np.exp(new_list)
    return np.mean(AP), mapNegWeights
