import os
import numpy as np
import torch
from hdf5storage import loadmat
from torch import nn
from skimage.metrics import structural_similarity as ssim_cpu
from torchmetrics.image import StructuralSimilarityIndexMeasure as ssim_gpu
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def calc_uiqi(img_tgt, img_fus):
    # 将输入的图像张量转换为 NumPy 数组并移到 CPU 上
    img_tgt = img_tgt.squeeze(0).data.cpu().numpy()
    img_fus = img_fus.squeeze(0).data.cpu().numpy()
    # 去除单维度
    img_tgt = np.squeeze(img_tgt)
    img_fus = np.squeeze(img_fus)

    if img_tgt.ndim == 2:  # 单波段图像
        # 获取数据范围
        data_range = img_tgt.max() - img_tgt.min()
        uiqi_value = ssim_cpu(img_tgt, img_fus, gaussian_weights=True, sigma=1.5, data_range=data_range)
    elif img_tgt.ndim == 3:  # 多波段图像
        num_bands = img_tgt.shape[0]
        uiqi_sum = 0
        for band in range(num_bands):
            # 获取当前波段的数据范围
            data_range = img_tgt[band, :, :].max() - img_tgt[band, :, :].min()
            # 计算每个波段的 SSIM 作为 UIQI
            uiqi_sum += ssim_cpu(img_tgt[band, :, :], img_fus[band, :, :], gaussian_weights=True, sigma=1.5, data_range=data_range)
        uiqi_value = uiqi_sum / num_bands
    else:
        raise ValueError("输入图像的维度必须是 2 或 3")

    return uiqi_value

def calc_uiqi_gpu(img_tgt, img_fus):
    # 确保输入张量在 GPU 上
    device = img_tgt.device
    img_tgt = img_tgt.to(device)
    img_fus = img_fus.to(device)

    if img_tgt.ndim == 2:  # 单波段图像
        ssim_metric = ssim_gpu().to(device)
        uiqi_value = ssim_metric(img_tgt.unsqueeze(0).unsqueeze(0), img_fus.unsqueeze(0).unsqueeze(0))
    elif img_tgt.ndim == 3:  # 多波段图像
        num_bands = img_tgt.shape[0]
        uiqi_sum = 0
        for band in range(num_bands):
            ssim_metric = ssim_gpu().to(device)
            uiqi_sum += ssim_metric(img_tgt[band].unsqueeze(0).unsqueeze(0), img_fus[band].unsqueeze(0).unsqueeze(0))
        uiqi_value = uiqi_sum / num_bands
    else:
        raise ValueError("输入图像的维度必须是 2 或 3")

    return uiqi_value.item()

def calc_ergas(img_tgt, img_fus):
    scale = 8
    img_tgt = img_tgt.squeeze(0).data.cpu().numpy()
    img_fus = img_fus.squeeze(0).data.cpu().numpy()
    img_tgt = np.squeeze(img_tgt)
    img_fus = np.squeeze(img_fus)
    img_tgt = img_tgt.reshape(img_tgt.shape[0], -1)
    img_fus = img_fus.reshape(img_fus.shape[0], -1)

    rmse = np.mean((img_tgt-img_fus)**2, axis=1)
    rmse = rmse**0.5
    mean = np.mean(img_tgt, axis=1)

    ergas = np.mean((rmse/mean)**2)
    ergas = 100/scale*ergas**0.5

    return ergas

def calc_ergas_gpu(img_tgt, img_fus, scale=8):
    # 确保输入张量在 GPU 上
    device = img_tgt.device
    img_tgt = img_tgt.squeeze(0)
    img_fus = img_fus.squeeze(0)

    # 将图像张量展平为二维张量 (C, H*W)
    img_tgt = img_tgt.reshape(img_tgt.shape[0], -1)
    img_fus = img_fus.reshape(img_fus.shape[0], -1)

    # 计算均方根误差 (RMSE)
    rmse = torch.mean((img_tgt - img_fus) ** 2, dim=1)
    rmse = torch.sqrt(rmse)

    # 计算目标图像的均值
    mean = torch.mean(img_tgt, dim=1)

    # 计算 ERGAS
    ergas = torch.mean((rmse / mean) ** 2)
    ergas = (100 / scale) * torch.sqrt(ergas)

    return ergas.item()  # 返回一个标量值

class Loss_PSNR(nn.Module):
    def __init__(self):
        super(Loss_PSNR, self).__init__()

    def forward(self, im_true, im_fake, data_range=255):
        Itrue = im_true.clamp(0., 1.)*data_range
        Ifake = im_fake.clamp(0., 1.)*data_range
        err=Itrue-Ifake
        err=torch.pow(err,2)
        err = torch.mean(err,dim=0)
        err = torch.mean(err,dim=0)

        psnr = 10. * torch.log10((data_range ** 2) / err)
        psnr=torch.mean(psnr)
        return psnr
    
class Loss_PSNR1(nn.Module):
    def __init__(self):
        super(Loss_PSNR1, self).__init__()

    def forward(self, im_true, im_fake, data_range=255):
        # 确保输入的形状一致
        assert im_true.shape == im_fake.shape, "Input shapes must match"
        
        # 将输入值缩放到 [0, data_range] 范围
        Itrue = im_true.clamp(0., 1.) * data_range
        Ifake = im_fake.clamp(0., 1.) * data_range
        
        # 计算误差平方
        err = Itrue - Ifake
        err = torch.pow(err, 2)
        
        # 计算均方误差（MSE）
        mse = torch.mean(err, dim=[1, 2, 3])  # 沿通道和空间维度计算均值，保留批量维度
        
        # 计算 PSNR
        psnr = 10. * torch.log10((data_range ** 2) / mse)
        
        # 返回整个批量的平均 PSNR
        return torch.mean(psnr)

class Loss_RMSE(nn.Module):
    def __init__(self):
        super(Loss_RMSE, self).__init__()

    def forward(self, outputs, label):
        assert outputs.shape == label.shape
        error = outputs.clamp(0., 1.)*255- label.clamp(0., 1.)*255
        sqrt_error = torch.pow(error,2)
        rmse = torch.sqrt(torch.mean(sqrt_error.contiguous().view(-1)))
        return rmse
    
class Loss_RMSE1(nn.Module):
    def __init__(self):
        super(Loss_RMSE1, self).__init__()

    def forward(self, outputs, label):
        # 确保输入的形状一致
        assert outputs.shape == label.shape, "Input shapes must match"
        
        # 将输入值缩放到 [0, 255] 范围
        outputs = outputs.clamp(0., 1.) * 255
        label = label.clamp(0., 1.) * 255
        
        # 计算误差平方
        error = outputs - label
        sqrt_error = torch.pow(error, 2)
        
        # 计算均方根误差（RMSE）
        rmse = torch.sqrt(torch.mean(sqrt_error, dim=[1, 2, 3]))  # 沿通道和空间维度计算均值，保留批量维度
        
        # 返回整个批量的平均 RMSE
        return torch.mean(rmse)

class Loss_SAM(nn.Module):
    def __init__(self):
        super(Loss_SAM, self).__init__()
        self.eps=2.2204e-16
    def forward(self,im1, im2):
        assert im1.shape == im2.shape
        H,W,C=im1.shape
        im1 = np.reshape(im1,( H*W,C))
        im2 = np.reshape(im2,(H*W,C))
        core=np.multiply(im1, im2)
        mole = np.sum(core, axis=1)
        im1_norm = np.sqrt(np.sum(np.square(im1), axis=1))
        im2_norm = np.sqrt(np.sum(np.square(im2), axis=1))
        deno = np.multiply(im1_norm, im2_norm)
        sam = np.rad2deg(np.arccos(((mole+self.eps)/(deno+self.eps)).clip(-1,1)))
        return np.mean(sam)

class Loss_SAM_gpu(nn.Module):
    def __init__(self):
        super(Loss_SAM_gpu, self).__init__()
        self.eps = 2.2204e-16

    def forward(self, im1, im2):
        # 确保输入张量在 GPU 上
        device = im1.device
        assert im1.shape == im2.shape, "Input images must have the same shape"

        # 获取图像的维度
        H, W, C = im1.shape

        # 将图像张量展平为二维张量 (H*W, C)
        im1 = im1.reshape(H * W, C)  # 使用 reshape 替代 view
        im2 = im2.reshape(H * W, C)  # 使用 reshape 替代 view

        # 计算分子
        core = torch.mul(im1, im2)
        mole = torch.sum(core, dim=1)

        # 计算分母
        im1_norm = torch.sqrt(torch.sum(torch.square(im1), dim=1))
        im2_norm = torch.sqrt(torch.sum(torch.square(im2), dim=1))
        deno = torch.mul(im1_norm, im2_norm)

        # 计算 SAM
        sam = torch.acos(torch.clamp((mole + self.eps) / (deno + self.eps), -1, 1))
        sam = torch.rad2deg(sam)

        # 返回 SAM 的平均值
        return torch.mean(sam)
    
class Loss_SAM1(nn.Module):
    def __init__(self):
        super(Loss_SAM1, self).__init__()
        self.eps = 2.2204e-16

    def forward(self, im1, im2):
        # 确保输入的形状是 [batch_size, C, H, W]
        assert im1.shape == im2.shape, "Input shapes must match"
        batch_size, C, H, W = im1.shape

        im1 = im1.transpose(0, 2, 3, 1).reshape(batch_size, H * W, C)
        im2 = im2.transpose(0, 2, 3, 1).reshape(batch_size, H * W, C)

        # 计算分子
        core = np.multiply(im1, im2)
        mole = np.sum(core, axis=2)

        # 计算分母
        im1_norm = np.sqrt(np.sum(np.square(im1), axis=2))  # 计算 im1 的范数
        im2_norm = np.sqrt(np.sum(np.square(im2), axis=2))
        deno = np.multiply(im1_norm, im2_norm)

        # 计算 SAM
        sam = np.arccos(np.clip((mole + self.eps) / (deno + self.eps), -1, 1))  # 计算夹角
        sam = np.rad2deg(sam)  # 将弧度转换为度

        # 计算平均 SAM 损失
        sam_loss = np.mean(sam, axis=1)  # 沿像素维度计算平均值
        return np.mean(sam_loss)  # 返回整个批量的平均 SAM 损失

if __name__ == '__main__':
    SAM=Loss_SAM()
    RMSE=Loss_RMSE()
    PSNR=Loss_PSNR()
    psnr_list=[]
    sam_list=[]
    sam=AverageMeter()
    rmse=AverageMeter()
    psnr=AverageMeter()
    path = 'D:\LYY\YJX_fusion\model_save\\fusion_model_v9_1\cavee_test/'
    imglist = os.listdir(path)

    for i in range(0, len(imglist)):
        img = loadmat(path + imglist[i])
        lable = img["rea"]
        recon = img["fak"]
        sam_temp=SAM(lable,recon)
        psnr_temp=PSNR(torch.Tensor(lable), torch.Tensor(recon))
        sam.update(sam_temp)
        rmse.update(RMSE(torch.Tensor(lable),torch.Tensor(recon)))
        psnr.update(psnr_temp)
        psnr_list.append(psnr_temp)
        sam_list.append(sam_temp)
    print(sam.avg)
    print(rmse.avg)
    print(psnr.avg)
    print(psnr_list)
    print(sam_list)
