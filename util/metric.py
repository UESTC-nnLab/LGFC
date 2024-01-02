import  numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from skimage import measure
from sklearn.metrics import auc
import threading

class SegmentationMetricTPFNFP(object):
    """Computes pixAcc and mIoU metric scroes
    """

    def __init__(self, nclass):
        self.nclass = nclass
        self.lock = threading.Lock()
        self.reset()

    def update(self, labels, preds):
        def evaluate_worker(self, label, pred):
            tp, fp, fn = batch_tp_fp_fn(pred, label, self.nclass)
            with self.lock:
                self.total_tp += tp
                self.total_fp += fp
                self.total_fn += fn
            return

        if isinstance(preds, torch.Tensor):
            evaluate_worker(self, labels, preds)
        elif isinstance(preds, (list, tuple)):
            threads = [threading.Thread(target=evaluate_worker,
                                        args=(self, label, pred),
                                        )
                       for (label, pred) in zip(labels, preds)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        else:
            raise NotImplemented

    def get_all(self):
        return self.total_tp, self.total_fp, self.total_fn

    def get(self):
        return get_miou_prec_recall_fscore(self.total_tp, self.total_fp, self.total_fn)

    def reset(self):
        self.total_tp = 0
        self.total_fp = 0
        self.total_fn = 0
        return



def get_miou_prec_recall_fscore(total_tp, total_fp, total_fn):
    miou = 1.0 * total_tp / (np.spacing(1) + total_tp + total_fp + total_fn)
    prec = 1.0 * total_tp / (np.spacing(1) + total_tp + total_fp)
    recall = 1.0 * total_tp / (np.spacing(1) + total_tp + total_fn)
    fscore = 2.0 * prec * recall / (np.spacing(1) + prec + recall)

    return miou, prec, recall, fscore


def batch_tp_fp_fn(output, target, nclass):
    """Batch Intersection of Union
    Args:
        predict: input 4D tensor
        target: label 3D tensor
        nclass: number of categories (int)
    """

    mini = 1
    maxi = nclass
    nbins = nclass

    predict = (output.cpu().detach().numpy() > 0).astype('int64')  # P
    target = target.cpu().numpy().astype('int64')  # T
    intersection = predict * (predict == target)  # TP

    # areas of intersection and union
    area_inter, _ = np.histogram(intersection, bins=nbins, range=(mini, maxi))
    area_pred, _ = np.histogram(predict, bins=nbins, range=(mini, maxi))
    area_lab, _ = np.histogram(target, bins=nbins, range=(mini, maxi))

    # areas of TN FP FN
    area_tp = area_inter[0]
    area_fp = area_pred[0] - area_inter[0]
    area_fn = area_lab[0] - area_inter[0]

    # area_union = area_pred + area_lab - area_inter
    assert area_tp <= (area_tp + area_fn + area_fp)
    return area_tp, area_fp, area_fn


class ROCMetric():
    def __init__(self, nclass=1, bins=10):  
        super(ROCMetric, self).__init__()
        self.nclass = nclass
        self.bins = bins
        self.reset()
    """
    更新评估指标,preds是预测结果,labels是标签
    """
    def update(self, preds, labels):
        for iBin in range(self.bins+1): #循环10次
            score_thresh = (iBin + 0.0) / self.bins
            # score_thresh = 0.5
            # print(iBin, "-th, score_thresh: ", score_thresh)

            i_tp, i_pos, i_fp, i_neg, i_class_pos = Cal_tp_pos_fp_neg(preds, labels, self.nclass,score_thresh)
            self.tp_arr[iBin]   += i_tp
            self.pos_arr[iBin]  += i_pos
            self.fp_arr[iBin]   += i_fp
            self.neg_arr[iBin]  += i_neg
            self.class_pos[iBin]+=i_class_pos

    def get(self):
        
        tp_rates    = self.tp_arr / (self.pos_arr + 0.001)  ##### pos 代表真值为正，neg代表真值为负
        fp_rates    = self.fp_arr / (self.neg_arr + 0.001)

        return tp_rates, fp_rates #### tp_rates 代表Recall 值， fp_rate 代表 false positive rate

    def reset(self):
        """Resets the internal evaluation result to initial state."""
        self.tp_arr   = np.zeros([11])
        self.pos_arr  = np.zeros([11])
        self.fp_arr   = np.zeros([11])
        self.neg_arr  = np.zeros([11])
        self.class_pos= np.zeros([11])

class PD_FA():
    def __init__(self, nclass, bins):  
        super(PD_FA, self).__init__()
        self.nclass = nclass
        self.bins = bins
        self.image_area_total = []
        self.image_area_match = []
        self.match_index = []
        self.FA = np.zeros(self.bins+1)
        self.PD = np.zeros(self.bins + 1)
        self.nIoU = np.zeros(self.bins + 1)
        self.target= np.zeros(self.bins + 1)
    def update(self, preds, labels):
        labels = np.array((labels).cpu()).astype('int64') 

        b,c,h,w = labels.shape

        labelss = labels 
        labelss = np.reshape(labelss, (b*c*256, 256))
        label = measure.label(labelss, connectivity=2) 
        coord_label = measure.regionprops(label) 
        for iBin in range(self.bins+1):
            # score_thresh = (iBin ) / self.bins #根据当前索引计算得到的阈值
            score_thresh = 0.5
            predit = np.array((preds > score_thresh).cpu()).astype('int64') 
            predits = np.reshape(predit, (b*c*256, 256))


            image = measure.label(predits, connectivity=2)
            coord_image = measure.regionprops(image)


            self.target[iBin]    += len(coord_label)
            self.image_area_total = []
            self.image_area_match = []
            self.distance_match   = []
            self.dismatch         = []
            self.IoU = 0
            for K in range(len(coord_image)):
                area_image = np.array(coord_image[K].area)
                self.image_area_total.append(area_image)

            self.sum_match=0
            self.match_index=[]
            for m in range(len(coord_image)):
                centroid_image = np.array(list(coord_image[m].centroid))

                area_image = np.array(coord_image[m].area)

                for i in range(len(coord_label)):
                    centroid_label = np.array(list(coord_label[i].centroid))
                    distance = np.linalg.norm(centroid_image - centroid_label)
                    if distance < 0.5*coord_label[i].equivalent_diameter:
                        self.distance_match.append(distance)
                        self.image_area_match.append(area_image)
                        self.match_index.append(i)
                        self.sum_match=self.sum_match+1

                        intersection = np.sum(np.array(image==m+1)*np.array(label==i+1))
                        label_sum = np.sum(np.array(coord_label[i].area))
                        pred_sum = np.sum(area_image)
                        self.IoU += intersection/(label_sum+pred_sum-intersection)
                        # del coord_image[m]
                        break

            self.match_index= list(set(self.match_index))

            self.dismatch = [x for x in self.image_area_total if x not in self.image_area_match]
            self.FA[iBin]+=np.sum(self.dismatch)
            self.PD[iBin]+=len(self.match_index)
            self.nIoU[iBin]+=self.IoU


    def get(self,img_num):

        Final_FA =  self.FA / ((256 * 256) * img_num)
        Final_PD =  self.PD /self.target
        Final_nIoU = self.nIoU / self.target

        return Final_FA,Final_PD,Final_nIoU


    def reset(self):
        self.FA  = np.zeros([self.bins+1])
        self.PD  = np.zeros([self.bins+1])
        self.nIoU= np.zeros([self.bins+1])
        self.target = np.zeros(self.bins + 1)

class MIoU():
    def __init__(self):
        super(MIoU, self).__init__()
        self.reset()

    def update(self, preds, labels):
        correct, labeled = batch_pix_accuracy(preds, labels)
        inter, union = batch_intersection_union(preds, labels)
        self.total_correct += correct
        self.total_label += labeled
        self.total_inter += inter
        self.total_union += union

    def get(self):
        pixAcc = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        mIoU = IoU.mean()
        return float(pixAcc), mIoU

    def reset(self):
        self.total_inter = 0
        self.total_union = 0
        self.total_correct = 0
        self.total_label = 0

def batch_pix_accuracy(output, target):
    output = output.cpu().detach().numpy()
    target = target.cpu().detach().numpy()

    predict = (output > 0).astype('int64')  # P
    pixel_labeled = np.sum(target > 0)  # T
    pixel_correct = np.sum((predict == target) * (target > 0))  # TP
    assert pixel_correct <= pixel_labeled
    return pixel_correct, pixel_labeled
   
def batch_intersection_union(output, target):
    mini = 1
    maxi = 1
    nbins = 1

    predict = (output.cpu().detach().numpy() > 0).astype('int64')  # P
    target = target.cpu().numpy().astype('int64')  # T
    intersection = predict * (predict == target)  # TP

    # areas of intersection and union
    area_inter, _ = np.histogram(intersection, bins=nbins, range=(mini, maxi))
    area_pred, _ = np.histogram(predict, bins=nbins, range=(mini, maxi))
    area_lab, _ = np.histogram(target, bins=nbins, range=(mini, maxi))
    area_union = area_pred + area_lab - area_inter
    assert (area_inter <= area_union).all()
    return area_inter, area_union


def Cal_tp_pos_fp_neg(output, target, nclass, score_thresh):
    mini = 1
    maxi = 1 
    nbins = 1 

    predict = (torch.sigmoid(output).cpu().detach().numpy() > score_thresh).astype('int64') # P
    target = target.cpu().detach().numpy().astype('int64')  # T
    intersection = predict * (predict == target) # TP
    tp = intersection.sum()
    fp = (predict * (predict != target)).sum()  # FP
    tn = ((1 - predict) * (predict == target)).sum()  # TN
    fn = ((predict != target) * (1 - predict)).sum()   # FN
    pos = tp + fn
    neg = fp + tn
    class_pos= tp+fp
    return tp, pos, fp, neg, class_pos

