import hdf5storage as h5

from Utils import *
import numpy as np
import torch
from torch.utils.data import Dataset
from multiprocessing import Pool


class RealDATAProcess3(Dataset):
    def __init__(self, LR, msi, HR, training_size, stride, downsample_factor):
        """
        :param path:
        :param R: 光谱响应矩阵
        :param training_size:
        :param stride:
        :param downsample_factor:
        :param PSF: 高斯模糊核
        :param num: cave=32个场景
        """
        train_hrhs = []
        train_lrhs = []
        train_hrms = []
        HSI_LR = LR
        MSI = msi
        HRHSI = HR
        for j in range(0, HRHSI.shape[1] - training_size + 1, stride):
            for k in range(0, HRHSI.shape[2] - training_size + 1, stride):
                # if (j+training_size)>800 and k<400:
                #     pass
                # else:
                temp_hrhs = HRHSI[:, j:j + training_size, k:k + training_size]
                temp_hrms = MSI[:, j:j + training_size, k:k + training_size]

                temp_lrhs = HSI_LR[:, int(j / downsample_factor):int((j + training_size) / downsample_factor),
                            int(k / downsample_factor):int((k + training_size) / downsample_factor)]
                # print(temp_hrhs.shape,temp_lrhs.shape,temp_hrms.shape)
                temp_hrhs = temp_hrhs.astype(np.float32)
                temp_lrhs = temp_lrhs.astype(np.float32)
                temp_hrms = temp_hrms.astype(np.float32)
                train_hrhs.append(temp_hrhs)
                train_lrhs.append(temp_lrhs)
                train_hrms.append(temp_hrms)

        train_hrhs = torch.Tensor(np.array(train_hrhs))
        train_lrhs = torch.Tensor(np.array(train_lrhs))
        train_hrms = torch.Tensor(np.array(train_hrms))

        # print(train_hrhs.shape, train_hrms.shape)
        self.train_hrhs_all = train_hrhs
        self.train_lrhs_all = train_lrhs
        self.train_hrms_all = train_hrms

    def __getitem__(self, index):
        train_hrhs = self.train_hrhs_all[index, :, :, :]
        train_lrhs = self.train_lrhs_all[index, :, :, :]
        train_hrms = self.train_hrms_all[index, :, :, :]
        # print(train_hrhs.shape, train_hrms.shape,train_lrhs.shape)
        return train_hrhs, train_lrhs, train_hrms

    def __len__(self):
        return self.train_hrhs_all.shape[0]


class RealDATAProcess2(Dataset):
    def __init__(self, hsi, msi, training_size, stride, downsample_factor, PSF):
        """
        :param path:
        :param R: 光谱响应矩阵
        :param training_size:
        :param stride:
        :param downsample_factor:
        :param PSF: 高斯模糊核
        :param num: cave=32个场景
        """
        train_hrhs = []
        train_lrhs = []
        train_hrms = []

        # hwc-chw
        HRHSI = np.transpose(hsi, (2, 0, 1))
        msi = np.transpose(msi, (2, 0, 1))

        HSI_LR = Gaussian_downsample(HRHSI, PSF, downsample_factor)
        MSI = Gaussian_downsample(msi, PSF, downsample_factor)

        for j in range(0, HRHSI.shape[1] - training_size + 1, stride):
            for k in range(0, HRHSI.shape[2] - training_size + 1, stride):
                # if (j+training_size)>800 and k<400:
                #     pass
                # else:
                temp_hrhs = HRHSI[:, j:j + training_size, k:k + training_size]
                temp_hrms = MSI[:, j:j + training_size, k:k + training_size]

                temp_lrhs = HSI_LR[:, int(j / downsample_factor):int((j + training_size) / downsample_factor),
                            int(k / downsample_factor):int((k + training_size) / downsample_factor)]
                # print(temp_hrhs.shape,temp_lrhs.shape,temp_hrms.shape)
                temp_hrhs = temp_hrhs.astype(np.float32)
                temp_lrhs = temp_lrhs.astype(np.float32)
                temp_hrms = temp_hrms.astype(np.float32)
                train_hrhs.append(temp_hrhs)
                train_lrhs.append(temp_lrhs)
                train_hrms.append(temp_hrms)

        train_hrhs = torch.Tensor(np.array(train_hrhs))
        train_lrhs = torch.Tensor(np.array(train_lrhs))
        train_hrms = torch.Tensor(np.array(train_hrms))

        # print(train_hrhs.shape, train_hrms.shape)
        self.train_hrhs_all = train_hrhs
        self.train_lrhs_all = train_lrhs
        self.train_hrms_all = train_hrms

    def __getitem__(self, index):
        train_hrhs = self.train_hrhs_all[index, :, :, :]
        train_lrhs = self.train_lrhs_all[index, :, :, :]
        train_hrms = self.train_hrms_all[index, :, :, :]
        # print(train_hrhs.shape, train_hrms.shape,train_lrhs.shape)
        return train_hrhs, train_hrms, train_lrhs

    def __len__(self):
        return self.train_hrhs_all.shape[0]


class RealDATAProcess(Dataset):
    def __init__(self, hsi, msi, training_size, stride, downsample_factor, PSF):
        """
        :param path:
        :param R: 光谱响应矩阵
        :param training_size:
        :param stride:
        :param downsample_factor:
        :param PSF: 高斯模糊核
        :param num: cave=32个场景
        """
        train_hrhs = []
        train_lrhs = []
        train_hrms = []

        HRHSI = hsi
        # hwc-chw
        HSI_LR = Gaussian_downsample(hsi, PSF, downsample_factor)
        MSI = Gaussian_downsample(msi, PSF, downsample_factor)

        for j in range(0, HRHSI.shape[1] - training_size + 1, stride):
            for k in range(0, HRHSI.shape[2] - training_size + 1, stride):
                temp_hrhs = HRHSI[:, j:j + training_size, k:k + training_size]
                temp_hrms = MSI[:, j:j + training_size, k:k + training_size]

                temp_lrhs = HSI_LR[:, int(j / downsample_factor):int((j + training_size) / downsample_factor),
                            int(k / downsample_factor):int((k + training_size) / downsample_factor)]
                # print(temp_hrhs.shape,temp_lrhs.shape,temp_hrms.shape)
                temp_hrhs = temp_hrhs.astype(np.float32)
                temp_lrhs = temp_lrhs.astype(np.float32)
                temp_hrms = temp_hrms.astype(np.float32)
                train_hrhs.append(temp_hrhs)
                train_lrhs.append(temp_lrhs)
                train_hrms.append(temp_hrms)

        train_hrhs = torch.Tensor(np.array(train_hrhs))
        train_lrhs = torch.Tensor(np.array(train_lrhs))
        train_hrms = torch.Tensor(np.array(train_hrms))

        # print(train_hrhs.shape, train_hrms.shape)
        self.train_hrhs_all = train_hrhs
        self.train_lrhs_all = train_lrhs
        self.train_hrms_all = train_hrms

    def __getitem__(self, index):
        train_hrhs = self.train_hrhs_all[index, :, :, :]
        train_lrhs = self.train_lrhs_all[index, :, :, :]
        train_hrms = self.train_hrms_all[index, :, :, :]
        # print(train_hrhs.shape, train_hrms.shape,train_lrhs.shape)
        return train_hrhs, train_hrms, train_lrhs

    def __len__(self):
        return self.train_hrhs_all.shape[0]


class CAVEHSIDATAprocess(Dataset):
    def __init__(self, path, R, training_size, stride, downsample_factor, PSF, num):
        """
        :param path:
        :param R: 光谱响应矩阵
        :param training_size:
        :param stride:
        :param downsample_factor:
        :param PSF: 高斯模糊核
        :param num: cave=32个场景
        """
        imglist = os.listdir(path)
        train_hrhs = []
        train_hrms = []
        train_lrhs = []

        for i in range(num):
            img = h5.loadmat(path + imglist[i])
            #             print(img)
            img1 = img["b"]

            HRHSI = np.transpose(img1, (2, 0, 1))
            # hwc-chw
            HSI_LR = Gaussian_downsample(HRHSI, PSF, downsample_factor)
            MSI = np.tensordot(R, HRHSI, axes=([1], [0]))
            for j in range(0, HRHSI.shape[1] - training_size + 1, stride):
                for k in range(0, HRHSI.shape[2] - training_size + 1, stride):
                    temp_hrhs = HRHSI[:, j:j + training_size, k:k + training_size]
                    temp_hrms = MSI[:, j:j + training_size, k:k + training_size]
                    temp_lrhs = HSI_LR[:, int(j / downsample_factor):int((j + training_size) / downsample_factor),
                                int(k / downsample_factor):int((k + training_size) / downsample_factor)]
                    # print(temp_hrhs.shape,temp_lrhs.shape,temp_hrms.shape)
                    temp_hrhs = temp_hrhs.astype(np.float32)
                    temp_hrms = temp_hrms.astype(np.float32)
                    temp_lrhs = temp_lrhs.astype(np.float32)
                    train_hrhs.append(temp_hrhs)
                    train_hrms.append(temp_hrms)
                    train_lrhs.append(temp_lrhs)
        train_hrhs = torch.Tensor(np.array(train_hrhs))
        train_hrms = torch.Tensor(np.array(train_hrms))
        train_lrhs = torch.Tensor(np.array(train_lrhs))

        # print(train_hrhs.shape, train_hrms.shape)
        self.train_hrhs_all = train_hrhs
        self.train_hrms_all = train_hrms
        self.train_lrhs_all = train_lrhs

    def __getitem__(self, index):
        train_hrhs = self.train_hrhs_all[index, :, :, :]
        train_hrms = self.train_hrms_all[index, :, :, :]
        train_lrhs = self.train_lrhs_all[index, :, :, :]
        # print(train_hrhs.shape, train_hrms.shape,train_lrhs.shape)
        return train_hrhs, train_hrms, train_lrhs

    def __len__(self):
        return self.train_hrhs_all.shape[0]


if __name__ == '__main__':
    print('hello world')


class HarvardHSIDATAprocess(Dataset):
    def __init__(self, path, R, training_size, stride, downsample_factor, PSF, num):
        """
        :param path:
        :param R: 光谱响应矩阵
        :param training_size:
        :param stride:
        :param downsample_factor:
        :param PSF: 高斯模糊核
        :param num: cave=32个场景
        """
        imglist = os.listdir(path)
        train_hrhs = []
        train_hrms = []
        train_lrhs = []

        for i in range(num):
            img = h5.loadmat(path + imglist[i])
            img1 = img["ref"]
            img1 = img1 / img1.max()

            HRHSI = np.transpose(img1, (2, 0, 1))
            # hwc-chw
            HSI_LR = Gaussian_downsample(HRHSI, PSF, downsample_factor)
            MSI = np.tensordot(R, HRHSI, axes=([1], [0]))
            for j in range(0, HRHSI.shape[1] - training_size + 1, stride):
                for k in range(0, HRHSI.shape[2] - training_size + 1, stride):
                    temp_hrhs = HRHSI[:, j:j + training_size, k:k + training_size]
                    temp_hrms = MSI[:, j:j + training_size, k:k + training_size]
                    temp_lrhs = HSI_LR[:, int(j / downsample_factor):int((j + training_size) / downsample_factor),
                                int(k / downsample_factor):int((k + training_size) / downsample_factor)]
                    # print(temp_hrhs.shape,temp_lrhs.shape,temp_hrms.shape)
                    temp_hrhs = temp_hrhs.astype(np.float32)
                    temp_hrms = temp_hrms.astype(np.float32)
                    temp_lrhs = temp_lrhs.astype(np.float32)
                    train_hrhs.append(temp_hrhs)
                    train_hrms.append(temp_hrms)
                    train_lrhs.append(temp_lrhs)
        train_hrhs = torch.Tensor(np.array(train_hrhs))
        train_hrms = torch.Tensor(np.array(train_hrms))
        train_lrhs = torch.Tensor(np.array(train_lrhs))

        # print(train_hrhs.shape, train_hrms.shape)
        self.train_hrhs_all = train_hrhs
        self.train_hrms_all = train_hrms
        self.train_lrhs_all = train_lrhs

    def __getitem__(self, index):
        train_hrhs = self.train_hrhs_all[index, :, :, :]
        train_hrms = self.train_hrms_all[index, :, :, :]
        train_lrhs = self.train_lrhs_all[index, :, :, :]
        # print(train_hrhs.shape, train_hrms.shape,train_lrhs.shape)
        return train_hrhs, train_hrms, train_lrhs

    def __len__(self):
        return self.train_hrhs_all.shape[0]


class ICVLDATAprocess(Dataset):
    def __init__(self, path, R, training_size, stride, downsample_factor, PSF, num):
        """
        :param path:
        :param R: 光谱响应矩阵
        :param training_size:
        :param stride:
        :param downsample_factor:
        :param PSF: 高斯模糊核
        :param num: cave=32个场景
        """
        imglist = os.listdir(path)
        train_hrhs = []
        train_hrms = []
        train_lrhs = []

        for i in range(num):
            img = h5.loadmat(path + imglist[i])
            img1 = img["HSI"]
            img1 = img1 / img1.max()
            HRHSI = np.transpose(img1, (2, 0, 1))

            w, h = int(HRHSI.shape[1] // downsample_factor), int(HRHSI.shape[2] // downsample_factor)
            HRHSI = HRHSI[:, :w * downsample_factor, :h * downsample_factor]
            # hwc-chw
            HSI_LR = Gaussian_downsample(HRHSI, PSF, downsample_factor)
            MSI = np.tensordot(R, HRHSI, axes=([1], [0]))
            for j in range(0, HRHSI.shape[1] - training_size + 1, stride):
                for k in range(0, HRHSI.shape[2] - training_size + 1, stride):
                    temp_hrhs = HRHSI[:, j:j + training_size, k:k + training_size]
                    temp_hrms = MSI[:, j:j + training_size, k:k + training_size]
                    temp_lrhs = HSI_LR[:, int(j / downsample_factor):int((j + training_size) / downsample_factor),
                                int(k / downsample_factor):int((k + training_size) / downsample_factor)]
                    # print(temp_hrhs.shape,temp_lrhs.shape,temp_hrms.shape)
                    temp_hrhs = temp_hrhs.astype(np.float32)
                    temp_hrms = temp_hrms.astype(np.float32)
                    temp_lrhs = temp_lrhs.astype(np.float32)
                    train_hrhs.append(temp_hrhs)
                    train_hrms.append(temp_hrms)
                    train_lrhs.append(temp_lrhs)
        train_hrhs = torch.Tensor(np.array(train_hrhs))
        train_hrms = torch.Tensor(np.array(train_hrms))
        train_lrhs = torch.Tensor(np.array(train_lrhs))

        # print(train_hrhs.shape, train_hrms.shape)
        self.train_hrhs_all = train_hrhs
        self.train_hrms_all = train_hrms
        self.train_lrhs_all = train_lrhs

    def __getitem__(self, index):
        train_hrhs = self.train_hrhs_all[index, :, :, :]
        train_hrms = self.train_hrms_all[index, :, :, :]
        train_lrhs = self.train_lrhs_all[index, :, :, :]
        # print(train_hrhs.shape, train_hrms.shape,train_lrhs.shape)
        return train_hrhs, train_hrms, train_lrhs

    def __len__(self):
        return self.train_hrhs_all.shape[0]


class KAISTDATAprocess(Dataset):
    def __init__(self, path, R, training_size, stride, downsample_factor, PSF, num):
        """
        :param path:
        :param R: 光谱响应矩阵
        :param training_size:
        :param stride:
        :param downsample_factor:
        :param PSF: 高斯模糊核
        :param num: cave=32个场景
        """
        imglist = os.listdir(path)
        train_hrhs = []
        train_hrms = []
        train_lrhs = []

        for i in range(num):
            img = h5.loadmat(path + imglist[i])
            img1 = img["hsi"]
            img1 = img1[:, :, 3:]
            img1 = img1 / img1.max()
            #             print(img1.shape)

            HRHSI = np.transpose(img1, (2, 0, 1))

            w, h = int(HRHSI.shape[1] // downsample_factor), int(HRHSI.shape[2] // downsample_factor)
            HRHSI = HRHSI[:, :w * downsample_factor, :h * downsample_factor]
            # hwc-chw
            HSI_LR = Gaussian_downsample(HRHSI, PSF, downsample_factor)
            MSI = np.tensordot(R, HRHSI, axes=([1], [0]))
            for j in range(0, HRHSI.shape[1] - training_size + 1, stride):
                for k in range(0, HRHSI.shape[2] - training_size + 1, stride):
                    temp_hrhs = HRHSI[:, j:j + training_size, k:k + training_size]
                    temp_hrms = MSI[:, j:j + training_size, k:k + training_size]
                    temp_lrhs = HSI_LR[:, int(j / downsample_factor):int((j + training_size) / downsample_factor),
                                int(k / downsample_factor):int((k + training_size) / downsample_factor)]
                    # print(temp_hrhs.shape,temp_lrhs.shape,temp_hrms.shape)
                    temp_hrhs = temp_hrhs.astype(np.float16)
                    temp_hrms = temp_hrms.astype(np.float16)
                    temp_lrhs = temp_lrhs.astype(np.float16)
                    train_hrhs.append(temp_hrhs)
                    train_hrms.append(temp_hrms)
                    train_lrhs.append(temp_lrhs)
        train_hrhs = torch.Tensor(np.array(train_hrhs))
        train_hrms = torch.Tensor(np.array(train_hrms))
        train_lrhs = torch.Tensor(np.array(train_lrhs))

        # print(train_hrhs.shape, train_hrms.shape)
        self.train_hrhs_all = train_hrhs
        self.train_hrms_all = train_hrms
        self.train_lrhs_all = train_lrhs

    def __getitem__(self, index):
        train_hrhs = self.train_hrhs_all[index, :, :, :]
        train_hrms = self.train_hrms_all[index, :, :, :]
        train_lrhs = self.train_lrhs_all[index, :, :, :]
        # print(train_hrhs.shape, train_hrms.shape,train_lrhs.shape)
        return train_hrhs, train_hrms, train_lrhs

    def __len__(self):
        return self.train_hrhs_all.shape[0]


def process_image(img_path, R, training_size, stride, downsample_factor, PSF, dataset='kaist'):
    """函数的作用是对单张高光谱图像进行预处理，生成训练所需的高分辨率高光谱图像（HRHSI）、低分辨率高光谱图像（HSI_LR）和多光谱图像（MSI）的训练样本"""
    img = h5.loadmat(img_path)
    # keys = list(img.keys())
    # print("Keys in the .mat file:", keys)
    if dataset == 'cave':
        img1 = img['b']
    if dataset == 'kaist':
        img1 = img["HSI"]
        img1 = img1 / img1.max()
    if dataset == 'harvard':
        img1 = img["ref"]
        img1 = img1 / img1.max()
    if dataset == 'pavia':
        img1 = img["data"]

    HRHSI = np.transpose(img1, (2, 0, 1))
    # w, h = int(HRHSI.shape[1] // downsample_factor), int(HRHSI.shape[2] // downsample_factor)
    # HRHSI = HRHSI[:, :w * downsample_factor, :h * downsample_factor]

    HSI_LR = Gaussian_downsample(HRHSI, PSF, downsample_factor)
    MSI = np.tensordot(R, HRHSI, axes=([1], [0]))

    train_hrhs = []
    train_hrms = []
    train_lrhs = []

    for j in range(0, HRHSI.shape[1] - training_size + 1, stride):
        for k in range(0, HRHSI.shape[2] - training_size + 1, stride):
            temp_hrhs = HRHSI[:, j:j + training_size, k:k + training_size]
            temp_hrms = MSI[:, j:j + training_size, k:k + training_size]
            temp_lrhs = HSI_LR[:, int(j / downsample_factor):int((j + training_size) / downsample_factor),
                        int(k / downsample_factor):int((k + training_size) / downsample_factor)]

            temp_hrhs = temp_hrhs.astype(np.float16)
            temp_hrms = temp_hrms.astype(np.float16)
            temp_lrhs = temp_lrhs.astype(np.float16)

            train_hrhs.append(temp_hrhs)
            train_hrms.append(temp_hrms)
            train_lrhs.append(temp_lrhs)

    return train_hrhs, train_hrms, train_lrhs


class DATAprocess(Dataset):
    def __init__(self, path, R, training_size, stride, downsample_factor, PSF, num, dataset='kaist'):
        imglist = os.listdir(path)
        self.train_hrhs_all = []
        self.train_hrms_all = []
        self.train_lrhs_all = []

        with Pool(processes=os.cpu_count()) as pool:
            results = pool.starmap(process_image, [
                (os.path.join(path, imglist[i]), R, training_size, stride, downsample_factor, PSF, dataset) for i in
                range(num)])

        for result in results:
            train_hrhs, train_hrms, train_lrhs = result
            self.train_hrhs_all.extend(train_hrhs)
            self.train_hrms_all.extend(train_hrms)
            self.train_lrhs_all.extend(train_lrhs)

        self.train_hrhs_all = torch.Tensor(np.array(self.train_hrhs_all))
        self.train_hrms_all = torch.Tensor(np.array(self.train_hrms_all))
        self.train_lrhs_all = torch.Tensor(np.array(self.train_lrhs_all))

    def __getitem__(self, index):
        train_hrhs = self.train_hrhs_all[index, :, :, :]
        train_hrms = self.train_hrms_all[index, :, :, :]
        train_lrhs = self.train_lrhs_all[index, :, :, :]
        return train_hrhs, train_hrms, train_lrhs

    def __len__(self):
        return len(self.train_hrhs_all)



