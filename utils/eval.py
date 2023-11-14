import os
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from PIL import Image
import json
from datasets.cityscapes_Dataset import name_classes

np.seterr(divide='ignore', invalid='ignore')

synthia_set_16 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 15, 17, 18]
synthia_set_13 = [0, 1, 2, 6, 7, 8, 10, 11, 12, 13, 15, 17, 18]
synthia_set_16_to_13 = [0, 1, 2, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

class Eval():
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)
        self.ignore_index = None
        self.synthia = True if num_class == 16 else False


    def Pixel_Accuracy(self):
        if np.sum(self.confusion_matrix) == 0:
            print("Attention: pixel_total is zero!!!")
            PA = 0
        else:
            PA = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()

        return PA

    def Mean_Pixel_Accuracy(self, out_16_13=False):
        MPA = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        if self.synthia:
            MPA_16 = np.nanmean(MPA[:self.ignore_index])
            MPA_13 = np.nanmean(MPA[synthia_set_16_to_13])
            return MPA_16, MPA_13
        if out_16_13:
            MPA_16 = np.nanmean(MPA[synthia_set_16])
            MPA_13 = np.nanmean(MPA[synthia_set_13])
            return MPA_16, MPA_13
        MPA = np.nanmean(MPA[:self.ignore_index])

        return MPA

    def Mean_Intersection_over_Union(self, out_16_13=False):
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        if self.synthia:
            MIoU_16 = np.nanmean(MIoU[:self.ignore_index])
            MIoU_13 = np.nanmean(MIoU[synthia_set_16_to_13])
            return MIoU_16, MIoU_13
        if out_16_13:
            MIoU_16 = np.nanmean(MIoU[synthia_set_16])
            MIoU_13 = np.nanmean(MIoU[synthia_set_13])
            return MIoU_16, MIoU_13
        MIoU = np.nanmean(MIoU[:self.ignore_index])

        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self, out_16_13=False):
        FWIoU = np.multiply(np.sum(self.confusion_matrix, axis=1), np.diag(self.confusion_matrix))
        FWIoU = FWIoU / (np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                         np.diag(self.confusion_matrix))
        if self.synthia:
            FWIoU_16 = np.sum(i for i in FWIoU if not np.isnan(i)) / np.sum(self.confusion_matrix)
            FWIoU_13 = np.sum(i for i in FWIoU[synthia_set_16_to_13] if not np.isnan(i)) / np.sum(self.confusion_matrix)
            return FWIoU_16, FWIoU_13
        if out_16_13:
            FWIoU_16 = np.sum(i for i in FWIoU[synthia_set_16] if not np.isnan(i)) / np.sum(self.confusion_matrix)
            FWIoU_13 = np.sum(i for i in FWIoU[synthia_set_13] if not np.isnan(i)) / np.sum(self.confusion_matrix)
            return FWIoU_16, FWIoU_13
        FWIoU = sum(i for i in FWIoU if not np.isnan(i)) / np.sum(self.confusion_matrix)

        return FWIoU

    def Mean_Precision(self, out_16_13=False):
        Precision = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=0)
        if self.synthia:
            Precision_16 = np.nanmean(Precision[:self.ignore_index])
            Precision_13 = np.nanmean(Precision[synthia_set_16_to_13])
            return Precision_16, Precision_13
        if out_16_13:
            Precision_16 = np.nanmean(Precision[synthia_set_16])
            Precision_13 = np.nanmean(Precision[synthia_set_13])
            return Precision_16, Precision_13
        Precision = np.nanmean(Precision[:self.ignore_index])
        return Precision
    
    def Print_Every_class_Eval(self, out_16_13=False):
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        MPA = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Precision = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=0)
        Class_ratio = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        Pred_retio = np.sum(self.confusion_matrix, axis=0) / np.sum(self.confusion_matrix)
        print('===>Everyclass:\t' + 'MPA\t' + 'MIoU\t' + 'PC\t' + 'Ratio\t' + 'Pred_Retio')
        if out_16_13: MIoU = MIoU[synthia_set_16]
        for ind_class in range(len(MIoU)):
            pa = str(round(MPA[ind_class] * 100, 2)) if not np.isnan(MPA[ind_class]) else 'nan'
            iou = str(round(MIoU[ind_class] * 100, 2)) if not np.isnan(MIoU[ind_class]) else 'nan'
            pc = str(round(Precision[ind_class] * 100, 2)) if not np.isnan(Precision[ind_class]) else 'nan'
            cr = str(round(Class_ratio[ind_class] * 100, 2)) if not np.isnan(Class_ratio[ind_class]) else 'nan'
            pr = str(round(Pred_retio[ind_class] * 100, 2)) if not np.isnan(Pred_retio[ind_class]) else 'nan'
            print('===>' + name_classes[ind_class] + ':\t' + pa + '\t' + iou + '\t' + pc + '\t' + cr + '\t' + pr)

    def Get_class_ratio(self):
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        Class_ratio = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        Pred_retio = np.sum(self.confusion_matrix, axis=0) / np.sum(self.confusion_matrix)
        return MIoU, Class_ratio, Pred_retio

    # generate confusion matrix
    def __generate_matrix(self, gt_image, pre_image):

        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        # assert the size of two images are same
        assert gt_image.shape == pre_image.shape

        self.confusion_matrix += self.__generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

def softmax(k, axis=None):
    exp_k = np.exp(k)
    return exp_k / np.sum(exp_k, axis=axis, keepdims=True)


def reliability_diagram(y_true, pred_classes, confidences, n_bins):
    bin_dic_acc = {}
    bin_dic_conf = {}
    #   ordered array: looping over its leangth we name thekeys of all the dictionaries to maintain order correspondence
    #   even if the hash map for dictionaries gets shuffled
    min_bins_borders = np.arange(0., 1., 1. / float(n_bins))
    #   the keys of the dict correspond to the increasing order: beans from left to right have oncreasing index
    for min_border_idx in range(len(min_bins_borders)):
        bin_dic_acc[min_border_idx] = []
        bin_dic_conf[min_border_idx] = []

    # if DEBUG:
    #     for i_ter in range(len(min_bins_borders)):
    #         if i_ter + 1 < len(min_bins_borders):
    #             print("sx_border ---> ", min_bins_borders[i_ter], " ### dx_border ---> ", min_bins_borders[i_ter + 1])
    #         else:
    #             print("sx_border ---> ", min_bins_borders[i_ter], " ### dx_border ---> ", 1.)

    list_acc_per_bin = compute_bin_acc(y_true, pred_classes, confidences, bin_dic_acc=bin_dic_acc,
                                          min_bins_borders=min_bins_borders)

    list_conf_per_bin, elements_in_bin = compute_bin_conf(confidences,
                                                             min_bins_borders=min_bins_borders,
                                                             bin_dic_conf=bin_dic_conf)

    return [min_bins_borders, list_acc_per_bin, list_conf_per_bin, elements_in_bin]





def plot_reliabiblity_diagram(y_true, pred_classes, confidences, n_bins, rel_diag_folder=None):
    x_data = None

    min_bins_borders, list_acc_per_bin, list_conf_per_bin, elements_in_bin = reliability_diagram(y_true=y_true,
                                                                                                 pred_classes=pred_classes,
                                                                                                 confidences=confidences,
                                                                                                 n_bins=n_bins)
    if x_data is None:
        x_data = min_bins_borders

    y_data = list_acc_per_bin
    conf_per_bin_list = list_conf_per_bin
    elements_in_bin_list = elements_in_bin

    # if DEBUG:
    #     print(len(y_data))
    #     print(len(conf_per_bin_list))
    #     print(len(elements_in_bin_list))

    # ECE = compute_ECE(y_data=y_data, conf_per_bin_list=conf_per_bin_list, elements_in_bin_list=elements_in_bin_list,
    #                      y_true=y_true)

    fig, ax = plt.subplots()

    y_data_acc = y_data
    # if DEBUG:
    #     a = np.array(y_data).reshape((len(y_data), len(y_data[0])))
    #     b = np.array(conf_per_bin_list).reshape((len(y_data), len(y_data[0])))
    #     print(y_data[0])
    #     print(a.shape)
    #     print(y_data_acc)

    y_data_conf = conf_per_bin_list

    y_data_gap_top = []
    y_data_gap_bottom = []

    for i_ter in range(len(y_data)):
        max_ = max(y_data_conf[i_ter], y_data_acc[i_ter])
        min_ = min(y_data_conf[i_ter], y_data_acc[i_ter])
        y_data_gap_top.append(max_ - min_)
        y_data_gap_bottom.append(min_)

    # ax.bar(x_data, height=y_data_gap_top, bottom=y_data_gap_bottom, width=1. / float(n_bins), color=(1.0, 0.0, 0.0, 0.2),
    #        label='Gap', hatch="///", edgecolor='black', align='edge')
    ax.bar(x_data, height=y_data_acc, width=1. / float(n_bins), color=(0.0, 0.0, 1.0, 0.2), label='Output',
           edgecolor='black', align='edge')

    ax.set_xlabel("Confidence intervals")
    ax.set_ylabel("Accuracy")

    # ax.legend(["mIOU=35.6"], loc='best')

    ax.text(0.05, 0.9, "mIOU = 35.6")

    plt.xlim(0, 1)

    # plt.plot([0, 1], [0, 1], linestyle='--', color='r', linewidth=2)
    # plt.show()
    plt.savefig("reliability diagram.pdf")

    # if rel_diag_folder is None:
    #     plt.show()
    # else:
    #     if rel_diag_folder[-1] != "/":
    #         rel_diag_folder += "/"
    #     try:
    #         os.makedirs(rel_diag_folder)
    #     except FileExistsError:
    #         # directory already exists
    #         pass
    #     plt.savefig(rel_diag_folder + "RED.png", dpi=300)



def compute_bin_acc(y_true, pred_classes, confidences, min_bins_borders, bin_dic_acc):
    for row_idx in range(y_true.shape[0]):
        confidence = confidences[row_idx]
        pred_class = pred_classes[row_idx]
        # pred_class = np.argmax(y_prob[row_idx, :])
        # confidence = np.max(y_prob[row_idx, :])
        real_class = y_true[row_idx]

        for min_border_idx in range(len(min_bins_borders)):
            if min_border_idx + 1 < len(min_bins_borders):
                if min_bins_borders[min_border_idx] < confidence <= min_bins_borders[min_border_idx + 1]:
                    if pred_class == real_class:
                        #   element in the dictionary always corresponding to the bin since it is its key
                        bin_dic_acc[min_border_idx].append(1)
                    else:
                        bin_dic_acc[min_border_idx].append(0)
                    break
            else:
                if pred_class == real_class:
                    bin_dic_acc[len(min_bins_borders) - 1].append(1)
                else:
                    bin_dic_acc[len(min_bins_borders) - 1].append(0)

    list_acc_per_bin = []

    for key in range(len(bin_dic_acc)):
        tmp = bin_dic_acc[key]
        if len(tmp) == 0:
            list_acc_per_bin.append(0)
        else:
            list_acc_per_bin.append(float(np.mean(np.array(tmp).reshape(len(tmp), 1), axis=0)[0]))

    return list_acc_per_bin

def compute_bin_conf(confidences, min_bins_borders, bin_dic_conf):
    """
    :param confidences
    :param min_bins_borders: separtor coordinates for the bins
    :param bin_dic_conf: dictionary of the confidence bin by bin
    :return: list of the confidence bin by bin and list of elements for each bin
    """
    #   loop over the soft-probabilities for each sample
    for confidence in confidences:
        #   get the highest element as the confidence
        # confidence = np.max(soft_probabilities_matrix[row_idx, :])

        #   min_bins_borders in ascending order
        #   check which bin the confidence belongs to
        #   assign each confidence to its bin
        for min_border_idx in range(len(min_bins_borders)):
            if min_border_idx + 1 < len(min_bins_borders):
                if min_bins_borders[min_border_idx] < confidence <= min_bins_borders[min_border_idx + 1]:
                    bin_dic_conf[min_border_idx].append(confidence)
                    break
            else:
                bin_dic_conf[len(min_bins_borders) - 1].append(confidence)

    #   list of list: each element is a bin and each bin can have elements inside
    list_conf_per_bin = []
    elements_in_bin = []

    for key in range(len(bin_dic_conf)):
        tmp = bin_dic_conf[key]
        elements_in_bin.append(len(tmp))
        if len(tmp) == 0:
            list_conf_per_bin.append(0)
        else:
            list_conf_per_bin.append(float(np.mean(np.array(tmp).reshape(len(tmp), 1), axis=0)[0]))

    return [list_conf_per_bin, elements_in_bin]

def build_palette(data_src):
    # palette
    if data_src == 'gta' or data_src == 'cityscapes':
        # gta:
        palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
                   220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
                   0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
    elif data_src == 'synthia':
        # synthia:
        palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 64, 64, 128, 153, 153, 153, 250, 170, 30, 220,
                   220, 0,
                   107, 142, 35, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 60, 100, 0, 0, 230, 119, 11, 32]

    zero_pad = 256 * 3 - len(palette)
    for i in range(zero_pad):
        palette.append(0)
    return palette

def colorize_mask(mask, data_src):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(build_palette(data_src))
    return new_mask

def save_preds(preds, data_src, name, method):
    output_col = colorize_mask(preds[0], data_src)
    output_col.save('./%s/%s_color_%s.png' % ('saved_images', name.split('.')[0], method))


def build_eval_info(class_16, logger, current_epoch):
    # get eval result
    if class_16:
        def val_info(Eval, name):
            PA = Eval.Pixel_Accuracy()
            MPA_16, MPA_13 = Eval.Mean_Pixel_Accuracy()
            MIoU_16, MIoU_13 = Eval.Mean_Intersection_over_Union()
            FWIoU_16, FWIoU_13 = Eval.Frequency_Weighted_Intersection_over_Union()
            PC_16, PC_13 = Eval.Mean_Precision()
            print("########## Eval{} ############".format(name))

            logger.info(
                '\nEpoch:{:.3f}, {} PA:{:.3f}, MPA_16:{:.3f}, MIoU_16:{:.3f}, FWIoU_16:{:.3f}, PC_16:{:.3f}'.format(
                    current_epoch, name, PA, MPA_16,
                    MIoU_16, FWIoU_16, PC_16))
            logger.info(
                '\nEpoch:{:.3f}, {} PA:{:.3f}, MPA_13:{:.3f}, MIoU_13:{:.3f}, FWIoU_13:{:.3f}, PC_13:{:.3f}'.format(
                    current_epoch, name, PA, MPA_13,
                    MIoU_13, FWIoU_13, PC_13))
            return PA, MPA_13, MIoU_13, FWIoU_13
    else:
        def val_info(Eval, name):
            PA = Eval.Pixel_Accuracy()
            MPA = Eval.Mean_Pixel_Accuracy()
            MIoU = Eval.Mean_Intersection_over_Union()
            FWIoU = Eval.Frequency_Weighted_Intersection_over_Union()
            PC = Eval.Mean_Precision()
            print("########## Eval{} ############".format(name))

            logger.info(
                '\nEpoch:{:.3f}, {} PA1:{:.3f}, MPA1:{:.3f}, MIoU1:{:.3f}, FWIoU1:{:.3f}, PC:{:.3f}'.format(
                    current_epoch, name, PA, MPA,
                    MIoU, FWIoU, PC))
            return PA, MPA, MIoU, FWIoU

    return val_info
