from torch import optim

from scipy.io import loadmat
from thop import profile, clever_format
import os
from calculate_metrics import *
import torch

from multiprocessing import Pool
from torch.utils.data import Dataset, DataLoader
from data_load import DATAprocess
from torch import nn
from tqdm import tqdm
import time
import pandas as pd
import torch.utils.data as data
from Utils import *
import math
import numpy as np
import hdf5storage as h5
from torch.cuda import is_available as is_cuda_available

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径
        print("训练文件夹为：{}".format(path))
    else:
        print('已存在{}'.format(path))


if __name__ == '__main__':
    # 路径参数

    from SINet import Net


    model = Net(31, 3).cuda()
    model_name = 'SINet'

    root = "./checkpoint_cave/"
    dataset = 'CAVE'
    data_path = './Data/CAVE/Train/HSI/'
    test_path = './Data/CAVE/Test/HSI/'

    path = root + model_name
    print(path)
    mkdir(os.path.join(root, model_name))

    # 训练参数
    loss_func = nn.L1Loss(reduction='mean').cuda()
    R = create_F()
    PSF = fspecial('gaussian', 8, 3)
    downsample_factor = 8
    training_size = 64  # 训练和测试裁剪块的大小
    LR = 1e-4
    EPOCH = 1000
    weight_decay = 1e-8
    BATCH_SIZE = 16

    psnr_optimal = 44
    rmse_optimal = 1.5
    val_epoch = 20  # 前50epoch不测试
    val_interval = 10  # 每隔val_interval epoch测试一次
    # checkpoint_interval = 40

    device = torch.device(f'cuda:{0}' if torch.cuda.is_available() else 'cpu')




    # set the number of parallel GPUs
    print("===> Setting GPU")
    #     model = dataparallel(model ,1)

    # 测试参数量
    a,b=64,64
    hsi = torch.randn(2, 31,a//8,b//8).cuda()
    msi = torch.randn(2, 3, a,b).cuda()
    flops, params,DIC= profile(model, inputs=(hsi,msi),ret_layer_info=True)
    # flops, params,DIC= profile(model, inputs=(msi,hsi),ret_layer_info=True)
    flops, params= clever_format([flops, params], "%.3f")
    print(flops, params)

    # 数据集处理
    if dataset == 'CAVE':
        stride = 32
        stride1 = 32
        train_data = DATAprocess(data_path, R, training_size, stride, downsample_factor, PSF, num=20,
                                      dataset='cave')

    if dataset == 'Harvard':
        stride = 32  # 滑动窗口方式裁剪，stride
        stride1 = 60
        train_data = DATAprocess(data_path, R, training_size, stride, downsample_factor, PSF, num=30,
                                      dataset='harvard')

    print('len(dataset):', len(train_data))
    train_loader = data.DataLoader(dataset=train_data, num_workers=4, batch_size=BATCH_SIZE, shuffle=True,
                                   pin_memory=True)


    # 模型初始化
    # Initialize weight
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            nn.init.xavier_uniform_(layer.weight)
        if isinstance(layer, nn.ConvTranspose2d):
            nn.init.xavier_uniform_(layer.weight)

    optimizer = optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-8)

    initial_epoch = findLastCheckpoint(save_dir=root + model_name)

    loss_eval = 999
    if initial_epoch > 0:
        print('resuming by loading epoch %04d' % initial_epoch)
        load_ckpt = torch.load(os.path.join(root + model_name, 'model_%04d.pth' % initial_epoch))
        load_weights_dict = {
            k: v
            for k, v in load_ckpt['parameter'].items()
            if k in model.state_dict() and model.state_dict()[k].numel() == v.numel()
        }

        # 可选：打印缺失的键（未被加载的键）
        missing_keys = [k for k in model.state_dict() if k not in load_weights_dict]
        print(f"Missing keys: {missing_keys}")
        load_weights_dict = load_ckpt['parameter']

        model.load_state_dict(load_weights_dict, strict=False)
        optimizer.load_state_dict(load_ckpt['optimizer'])  # 加载优化器状态
    #         scheduler.load_state_dict(load_ckpt['scheduler'])

    # 创建excel
    df = pd.DataFrame(columns=['epoch', 'lr', 'train_loss', 'val_psnr', 'val_sam'])  # 列名
    excel_name = dataset + '_record.csv'
    excel_path = os.path.join(path, excel_name)
    if initial_epoch == 0:
        df.to_csv(excel_path, index=False)
        step = 0

    for epoch in range(initial_epoch + 1, EPOCH + 1):
        model.train()
        loss_all = []
        loop = tqdm(train_loader, total=len(train_loader))
        for HR, RGB, LR in loop:
            lr = optimizer.param_groups[0]['lr']
            output = model(LR.cuda(),RGB.cuda())

            loss = loss_func(output, HR.cuda())
            loss_temp = loss
            loss_all.append(np.array(loss_temp.detach().cpu().numpy()))
            optimizer.zero_grad()
            loss.backward()


            optimizer.step()
            loop.set_description(f'Epoch [{epoch}/{EPOCH}]')
            loop.set_postfix({'loss': '{0:1.8f}'.format(np.mean(loss_all)), "lr": '{0:1.6f}'.format(lr)})
        #         scheduler.step()

        if ((epoch % val_interval == 0) and (epoch >= val_epoch)) or epoch == EPOCH or epoch == 1:
            model.eval()
            checkpoint = {'parameter': model.state_dict(),
                          'optimizer': optimizer.state_dict(),
                          #                           'scheduler': scheduler.state_dict(),
                          'epoch': epoch,
                          'loss': loss_eval
                          }
            torch.save(checkpoint, root + model_name + '/model_%04d.pth' % (epoch))
            val_loss = AverageMeter()
            SAM = Loss_SAM_gpu()
            RMSE = Loss_RMSE()
            PSNR = Loss_PSNR()
            sam = AverageMeter()
            rmse = AverageMeter()
            psnr = AverageMeter()

            imglist = os.listdir(test_path)
            with torch.no_grad():
                for i in range(0, len(imglist) - 8):
                    print(i)
                    img = h5.loadmat(test_path + imglist[i])
                    if dataset == 'Harvard':
                        img1 = img["ref"]
                        img1 = img1 / img1.max()
                    if dataset == 'CAVE':
                        img1 = img['b']
                        # img1 = img1 / img1.max()
                    print(img1.shape)
                    HRHSI = torch.Tensor(np.transpose(img1, (2, 0, 1)))

                    w, h = int(HRHSI.shape[1] / downsample_factor), int(HRHSI.shape[2] / downsample_factor)
                    HRHSI = HRHSI[:, :w * downsample_factor, :h * downsample_factor]

                    MSI = torch.tensordot(torch.Tensor(R), HRHSI, dims=([1], [0]))
                    HSI_LR = Gaussian_downsample(HRHSI, PSF, downsample_factor)
                    MSI_1 = torch.unsqueeze(MSI, 0)
                    HSI_LR1 = torch.unsqueeze(torch.Tensor(HSI_LR), 0)  # 加维度 (b,c,h,w)
                    # 计算val_loss用的，防止出错单独拿出来
                    to_fet_loss_hr_hsi = torch.unsqueeze(torch.Tensor(HRHSI), 0)
                    HRHSI = HRHSI.to(device)
                    HSI_LR1 = HSI_LR1.to(device)
                    MSI_1 = MSI_1.to(device)

                    prediction, val_loss = reconstruction(model, HSI_LR1, MSI_1, to_fet_loss_hr_hsi, downsample_factor,
                                                          training_size, stride1, val_loss)
                    sam.update(SAM(HRHSI.permute(1, 2, 0), prediction.squeeze().permute(1, 2, 0)))
                    psnr.update(PSNR(HRHSI.permute(1, 2, 0), prediction.squeeze().permute(1, 2, 0)))

                                     
                print(f"Epoch {epoch}, Val PSNR: {psnr.avg}, RMSE: {rmse.avg}, SAM: {sam.avg}")
                val_list = [epoch, lr, np.mean(loss_all), psnr.avg.cpu().detach().numpy(),
                            sam.avg.cpu().detach().numpy()]

                val_data = pd.DataFrame([val_list])
                val_data.to_csv(excel_path, mode='a', header=False, index=False)  # mode设为a,就可以向csv文件追加数据了
                time.sleep(0.1)


