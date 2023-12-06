# custom dataset for training or testing

import os
import os.path
from random import choice
import random

import torch
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class customDataset(Dataset):
    def __init__(self, train=True, root='', data_path='', cur_label='trainImageLabel', view_nums=5, transforms=None):
        self.root = root
        self.labelPath = os.path.join(self.root, cur_label)
        self.imageListPath = data_path
        if train:
            self.sketchListPath = os.path.join(self.root, 'input-sketch')
        else:
            self.sketchListPath = os.path.join(self.root, 'test-sketch')
        self.cacheMap = {}
        self.transforms = transforms
        self.custom_view_nums = view_nums

        class_num = 7           # class num index limit +1
        input_sketch_num = 6                # free-sketch group num index limit +1
        if not train:
            input_sketch_num = 2

        # init weights value
        self.classRelNeg = {}
        self.classRelNeg_Weights = {}
        for i in range(1, class_num):
            tmpListValue = []
            tmpListValueDef = []
            for mm in range(1, class_num):
                if mm != i:
                    tmpListValue.append(mm)
                    tmpListValueDef.append(1)  # set 1, the same probability to select
            self.classRelNeg[str(i)] = tmpListValue
            self.classRelNeg_Weights[str(i)] = tmpListValueDef

        for i in range(1, class_num):
            self.cacheMap[str(i)] = []
        file = open(self.labelPath)
        tmpIdNum = 0
        for line in file:
            tmpIdNum = tmpIdNum + 1
        file.close()

        tmpIndexlist = []
        for divisors in range(tmpIdNum + 1):
            if divisors % view_nums == 0:
                if divisors != tmpIdNum:
                    tmpIndexlist.append(divisors)

        file = open(self.labelPath)
        curTmpIdNum = 0
        for line in file:
            if curTmpIdNum in tmpIndexlist:
                sketch_cls = line.split()[0]
                img_path = line.split()[1][:-4] + '.png'
                img_cls = img_path.split('/')[0]
                img_name = img_path.split('/')[1][:-4]
                img_path = os.path.join(self.imageListPath, img_path)
                # check image valid
                if os.path.exists(img_path):
                    self.cacheMap[sketch_cls].append((img_path, img_cls, img_name))
            curTmpIdNum = curTmpIdNum + 1
        file.close()

        self.datapath = []
        for i in range(1, class_num):
            item = str(i)
            for fn in self.cacheMap[item]:
                # f[0]: file path
                # f[1]: class num
                # f[2]: file name
                self.datapath.append((fn[1], item, fn[0], fn[2]))

        self.sketch_datapath = []
        for i in range(1, input_sketch_num):
            for j in range(1, class_num):
                cls_name = self.cacheMap[str(j)][0][1]
                abs_path = os.path.join(self.sketchListPath, str(i), (str(j) + '.png'))
                self.sketch_datapath.append((cls_name, str(j), abs_path, str(j)))

    # select index from weights
    def weight_choice(self, weight):
        t = random.uniform(0, sum(weight) - 1)
        for i, val in enumerate(weight):
            t -= val
            if t < 0:
                return i

    def setWeights(self, curValues):
        self.classRelNeg_Weights = curValues

    def selectFromMapWeights(self, curClassPosNum):
        # curClassPosNum, positive sample index
        curListSel = self.classRelNeg[str(curClassPosNum)]
        curListSelWeights = self.classRelNeg_Weights[str(curClassPosNum)]
        indexSel = self.weight_choice(curListSelWeights)
        return curListSel[indexSel]

    def __getitem__(self, idx):
        # process sample in dataset
        anc_fn = self.datapath[idx]
        cls_num_a = anc_fn[1]
        anc_path = anc_fn[2]
        anc_name = anc_fn[3]

        #  input1 : anchor image
        # select a anchor image from class num
        sketch_path = self.selectFreeSketchItem(cls_num_a, self.sketchListPath)
        img_a = Image.open(sketch_path).convert('RGB')
        img_a = self.transforms(img_a)

        #  input2 : positive image
        pos_path = anc_path
        pos_name = anc_name
        imageList_p = []
        self.retriveAllImagesOfView(pos_path, pos_name, imageList_p)
        imgs_p = []
        for iPathIndex in imageList_p:
            im = Image.open(iPathIndex).convert('RGB')
            if self.transforms:
                im = self.transforms(im)
            imgs_p.append(im)

        #  input3 : positive image
        cls_num_n = str(self.selectFromMapWeights(int(cls_num_a)))
        neg_fn = self.cacheMap[cls_num_n][choice([i for i in range(0, len(self.cacheMap[cls_num_n]))])]
        neg_name = neg_fn[2]
        neg_path = neg_fn[0]
        imageList_n = []
        # get the same image from views
        self.retriveAllImagesOfView(neg_path, neg_name, imageList_n)
        imgs_n = []
        for iPathIndex in imageList_n:
            im = Image.open(iPathIndex).convert('RGB')
            if self.transforms:
                im = self.transforms(im)
            imgs_n.append(im)

        return img_a, torch.stack(imgs_p), torch.stack(imgs_n)

    def retriveAllImagesOfView(self, iPath, iName, oImagePaths):
        curTmpPath = iPath.split('/')[0]
        position = iName.rfind('-')
        strName = iName[0:position]
        strName = strName + '-'
        for i in range(0, self.custom_view_nums):
            strNameCur = strName + str(i)
            strCurPath = curTmpPath + '\\' + strNameCur + '.png'
            oImagePaths.append(strCurPath)

    def __len__(self):
        return len(self.datapath)

    def getMutilViewImageItem(self, idx):
        img_edge = self.datapath[idx]
        path = img_edge[2]
        name = img_edge[3]
        imgs = []
        self.retriveAllImagesOfView(path, name, imgs)
        imgs_cur = []
        for iPathIndex in imgs:
            im = Image.open(iPathIndex).convert('RGB')
            if self.transforms:
                im = self.transforms(im)
            imgs_cur.append(im)

        return torch.stack(imgs_cur), img_edge

    def getFreeSketchItem(self, idx):
        sketch = self.sketch_datapath[idx]
        path = sketch[2]
        sketch_src = Image.open(path).convert('RGB')
        sketch_src = self.transforms(sketch_src)
        return sketch_src, sketch

    def selectFreeSketchItem(self, cls_num, sketch_set_path):
        user_select = str(random.randint(1, 5))
        sketch_path = os.path.join(sketch_set_path, user_select, (cls_num + '.png'))
        return sketch_path


# get a training dataset
def performTrainLoader(batchSize, img_size, viewNums, root, dataPath, curLabel):
    # image transformation
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        normalize
    ])

    # train set
    trainPath = dataPath + "\\" + 'train'
    train_loader = DataLoader(
        customDataset(train=True, transforms=transform, root=root, data_path=trainPath, cur_label=curLabel,
                      view_nums=viewNums),
        batch_size=batchSize, shuffle=True, num_workers=os.cpu_count(), pin_memory=True)
    print(f'train set: {len(train_loader.dataset)}')

    return train_loader


# get a validation  dataset
def performLoadValData(img_size, root, dataPath, curLabel, viewNum):
    # image transformation
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        normalize
    ])
    return customDataset(train=True, root=root, data_path=dataPath, cur_label=curLabel, transforms=transform,
                         view_nums=viewNum)

# get a test  dataset
def performLoadTestData(img_size, root, dataPath, curLabel, viewNum):
    # image transformation
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        normalize
    ])
    return customDataset(train=False, root=root, data_path=dataPath, cur_label=curLabel, transforms=transform,
                         view_nums=viewNum)
