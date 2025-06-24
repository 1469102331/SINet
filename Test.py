# -*- coding: utf-8 -*-

import time

import numpy as np

import torch
from scipy.io import loadmat
from Utils import *
import os
import hdf5storage as h5
from torch import nn
# from Compare_Model.TFNet import ResTFNet
# from Compare_Model.SSRNET import SSRNET
# from Compare_Model.MIMO import Net
from calculate_metrics import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import hdf5storage

device = torch.device(f'cuda:{0}' if torch.cuda.is_available() else 'cpu')


def get_edge(data):
    rs = np.zeros_like(data)
    N = data.shape[0]
    for i in range(N):
        if len(data.shape) == 3:
            rs[i, :, :] = data[i, :, :] - cv2.boxFilter(data[i, :, :], -1, (5, 5))
        else:
            rs[i, :, :, :] = data[i, :, :, :] - cv2.boxFilter(data[i, :, :, :], -1, (5, 5))
    return rs


dataset='CAVE'
test_path = r'./Data/CAVE/Test/HSI/'
root= "./checkpoint_cave/"



# dataset = 'Harvard'
# test_path = r'./Data/Harvard/Test/HSI/'
# root = "./checkpoint_harvard/"

imglist = os.listdir(test_path)

from SINet import Net
model = Net(31, 3).cuda()
model_name = 'SINet'


print(model_name)
path = root + model_name
save_path = './test_save/' + model_name + '/' + dataset + '/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

test_epoch = 1000
load_ckpt = (torch.load(os.path.join(path, "model_%04d.pth" % test_epoch), map_location=device))
# load_weights_dict = {k: v for k, v in load_ckpt['parameter'].items()
#                                       if model.state_dict()[k].numel() == v.numel()}
load_weights_dict = {
    k: v
    for k, v in load_ckpt['parameter'].items()
    if k in model.state_dict() and model.state_dict()[k].numel() == v.numel()
}

# 可选：打印缺失的键（未被加载的键）
missing_keys = [k for k in model.state_dict() if k not in load_weights_dict]
print(f"Missing keys: {missing_keys}")
load_weights_dict = load_ckpt['parameter']
# model.load_state_dict(load_weights_dict)
model.load_state_dict(load_weights_dict, strict=False)

R = create_F()
R_inv = np.linalg.pinv(R)
R_inv = torch.Tensor(R_inv)
R = torch.Tensor(R)

RMSE = []
training_size = 64

stride1 = 32
PSF = fspecial('gaussian', 8, 3)
downsample_factor = 8

loss_func = nn.L1Loss(reduction='mean').cuda()


def reconstruction(net2, HSI_LR, MSI, HRHSI, downsample_factor, training_size, stride, val_loss):
    index_matrix = torch.zeros((HSI_LR.shape[1], MSI.shape[2], MSI.shape[3])).cuda()
    abundance_t = torch.zeros((HSI_LR.shape[1], MSI.shape[2], MSI.shape[3])).cuda()
    a = []
    for j in range(0, MSI.shape[2] - training_size + 1, stride):
        a.append(j)
    a.append(MSI.shape[2] - training_size)
    b = []
    for j in range(0, MSI.shape[3] - training_size + 1, stride):
        b.append(j)
    b.append(MSI.shape[3] - training_size)
    for j in a:
        for k in b:
            temp_hrms = MSI[:, :, j:j + training_size, k:k + training_size]
            temp_lrhs = HSI_LR[:, :, int(j / downsample_factor):int((j + training_size) / downsample_factor),
                        int(k / downsample_factor):int((k + training_size) / downsample_factor)]
            temp_hrhs = HRHSI[:, :, j:j + training_size, k:k + training_size]
            with torch.no_grad():
                out = net2(temp_lrhs, temp_hrms)
                #                 out = net2(temp_hrms,temp_lrhs) #PSRT
                #                 outputs3, outputs2, out = net2(temp_hrms,temp_lrhs) # MIMO
                #                 out, out_spat, out_spec, edge_spat1, edge_spat2, edge_spec = model(temp_lrhs,temp_hrms) #SSRNET
                #                 loss_temp = loss_func(out, temp_hrhs.cuda())
                #                 val_loss.update(loss_temp)
                HSI = out.squeeze()
                # 去掉维数为一的维度
                HSI = torch.clamp(HSI, 0, 1)
                abundance_t[:, j:j + training_size, k:k + training_size] = abundance_t[:, j:j + training_size,
                                                                           k:k + training_size] + HSI
                index_matrix[:, j:j + training_size, k:k + training_size] = 1 + index_matrix[:, j:j + training_size,
                                                                                k:k + training_size]
    val_loss = 0
    HSI_recon = abundance_t / index_matrix
    assert torch.isnan(HSI_recon).sum() == 0
    return HSI_recon, val_loss


val_loss = AverageMeter()
SAM = Loss_SAM_gpu()
RMSE = Loss_RMSE()
PSNR = Loss_PSNR()
sam = AverageMeter()
rmse = AverageMeter()
psnr = AverageMeter()
uiqi_total = 0
k = 0
ergas_total = 0
ssim_total = 0
with torch.no_grad():
    for i in range(0, len(imglist)):

        print(i + 1)
        img = h5.loadmat(test_path + imglist[i])
        if dataset == 'Harvard':
            img1 = img["ref"]
            img1 = img1 / img1.max()
        if dataset == 'CAVE':
            img1 = img['b']
     
        time1 = time.time()
        HRHSI = torch.Tensor(np.transpose(img1, (2, 0, 1)))

        w, h = int(HRHSI.shape[1] / downsample_factor), int(HRHSI.shape[2] / downsample_factor)
        HRHSI = HRHSI[:, :w * downsample_factor, :h * downsample_factor]

        MSI = torch.tensordot(torch.Tensor(R), HRHSI, dims=([1], [0]))
        HSI_LR = Gaussian_downsample(HRHSI, PSF, downsample_factor)
        MSI_1 = torch.unsqueeze(MSI, 0)
        HSI_LR1 = torch.unsqueeze(torch.Tensor(HSI_LR), 0)  # 加维度 (b,c,h,w)
        # 计算val_loss用的，防止出错单独拿出来
        HRHSI = HRHSI.to(device)
        HSI_LR1 = HSI_LR1.to(device)
        MSI_1 = MSI_1.to(device)
        to_fet_loss_hr_hsi = torch.unsqueeze(torch.Tensor(HRHSI), 0)

        prediction, val_loss = reconstruction(model, HSI_LR1, MSI_1, to_fet_loss_hr_hsi, downsample_factor,
                                              training_size, stride1, val_loss)

        Fuse = prediction.squeeze().cpu().detach().numpy()
        print(Fuse.shape)
        faker_hyper = np.transpose(Fuse, (1, 2, 0))
        gt = HRHSI.cpu().detach().numpy()
        gt = np.transpose(gt, (1, 2, 0))
        test_data_path = os.path.join(save_path + str(i))
        hdf5storage.savemat(test_data_path, {'fak': faker_hyper}, format='7.3')
        hdf5storage.savemat(test_data_path, {'rea': gt}, format='7.3')

        ergas = calc_ergas_gpu(HRHSI.cuda(), prediction)
        ergas_total = ergas_total + ergas
        uiqi = calc_uiqi_gpu(HRHSI.cuda(), prediction)
        uiqi_total = uiqi_total + uiqi
        k = k + 1
        #         sam.update(SAM(np.transpose(HRHSI.detach().numpy(),(1, 2, 0)),np.transpose(prediction.squeeze().detach().numpy(),(1, 2, 0))))
        sam.update(SAM(HRHSI.permute(1, 2, 0), prediction.squeeze().permute(1, 2, 0)))

        rmse.update(RMSE(HRHSI.permute(1, 2, 0), prediction.squeeze().permute(1, 2, 0)))
        psnr.update(PSNR(HRHSI.permute(1, 2, 0), prediction.squeeze().permute(1, 2, 0)))
    #
    print(i)
    print('val ergas:', ergas_total / k)
    print('val uiqi:', uiqi_total / k)
    print("val PSNR:", psnr.avg.cpu().detach().numpy())
    print(" RMSE:", rmse.avg.cpu().detach().numpy())
    print("  SAM:", sam.avg.cpu().detach().numpy())

