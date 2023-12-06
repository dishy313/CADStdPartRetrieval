# reference link :https://blog.csdn.net/pengchengliu/article/details/119418007
import numpy as np

def CalcDist(B1, B2):
    # compute L2 diatance
    dist = np.sqrt(np.sum(np.square(B1 - B2), 1))
    return dist

def CalcTopMap(qB, rB, queryL, retrievalL, topk):
    num_query = queryL.shape[0]
    topkmap = 0
    for iter in range(num_query):
        gnd = np.empty(len(retrievalL), dtype=np.float32)
        for ii in range(len(retrievalL)):
            if queryL[iter] == retrievalL[ii]:
                gnd[ii] = 1.0
            else:
                gnd[ii] = 0.0

        distance = CalcDist(qB[iter, :], rB)
        ind = np.argsort(distance)
        gnd = gnd[ind]

        tgnd = gnd[0:topk]
        tsum = np.sum(tgnd).astype(int)
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, tsum)

        tindex = np.asarray(np.where(tgnd == 1)) + 1.0
        topkmap_ = np.mean(count / (tindex))
        topkmap += topkmap_
    topkmap = topkmap / num_query
    return topkmap


def CalcMap(qB, rB, queryL, retrievalL):
    num_query = queryL.shape[0]
    map = 0
    for iter in range(num_query):
        gnd = np.empty(len(retrievalL), dtype=np.float32)
        for ii in range(len(retrievalL)):
            if queryL[iter] == retrievalL[ii]:
                gnd[ii] = 1.0
            else:
                gnd[ii] = 0.0

        tsum = np.sum(gnd).astype(int)
        if tsum == 0:
            continue
        distance = CalcDist(qB[iter, :], rB)
        ind = np.argsort(distance)
        gnd = gnd[ind]
        count = np.linspace(1, tsum, tsum)

        tindex = np.asarray(np.where(gnd == 1)) + 1.0
        map_ = np.mean(count / (tindex))
        map = map + map_
    map = map / num_query

    return map

def CalcTopAcc(qB, rB, queryL, retrievalL, topk):
    num_query = queryL.shape[0]
    topkacc = 0
    for iter in range(num_query):
        gnd = np.empty(len(retrievalL), dtype=np.float32)
        for ii in range(len(retrievalL)):
            if queryL[iter] == retrievalL[ii]:
                gnd[ii] = 1.0
            else:
                gnd[ii] = 0.0

        distance = CalcDist(qB[iter, :], rB)
        ind = np.argsort(distance)
        gnd = gnd[ind]
        tgnd = gnd[0:topk]
        tsum = np.sum(tgnd)
        if tsum == 0:
            continue
        topkacc += tsum / topk
    topkacc = topkacc / num_query
    return topkacc