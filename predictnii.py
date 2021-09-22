import os
import torch
import numpy as np
import pandas as pd
import cv2
from utils import file_name_path, load_itkfilewithtruncation, resize_image_itk, resize_image_itkwithsize, removesmallConnectedCompont, morphologicaloperation, getRangeImageDepth, outputbmp
import SimpleITK as sitk
import torch3dModel
import torch2dModel
import time


def test_3ddice(input, modelpath):
    model = torch3dModel.get_net(training=False)
    model.cuda()
    model.load_state_dict(torch.load(modelpath))
    model.eval()
    Z,H,W = input.shape
    C = 1
    image = np.reshape(input, (1, C, Z, H, W))
    # 三维数据主动标准化并转化为tensor
    image = image.astype(np.float32)
    nimage = np.multiply(image, 1.0 / 255)
    timage = torch.from_numpy(nimage)
    data = timage.cuda()
    output = model(data)
    output = (output.cpu()).detach().numpy()
    return output

def test_2ddice(input, modelpath):
    model = torch2dModel.get_net(training=False)
    model.cuda()
    model.load_state_dict(torch.load(modelpath))
    model.eval()
    H,W = input.shape
    C = 1
    image = np.reshape(input, (1,C, H, W))
    # 三维数据主动标准化并转化为tensor
    image = image.astype(np.float32)
    nimage = np.multiply(image, 1.0 / 255)
    timage = torch.from_numpy(nimage)
    data = timage.cuda()
    output = model(data)
    output = (output.cpu()).detach().numpy()
    return output

#step 一：模型预测

def predict(file):
    depth_z = 16
    height_h = 128
    width_w = 128
    kits_path = "C:/CSUProgram/Kits/data/kits"
    image_name = "imaging.nii.gz"

    """
    load itk image,change z Spacing value to 1,and save image ,liver mask ,tumor mask
    """
    # step1 predict 3d kindey
    path_list = file_name_path(kits_path)
    #kits_subset_path = kits_path + "/" + str(path_list[subsetindex]) + "/"
    # file_image = kits_subset_path + image_name
    file_image = file
    print(file_image)
    # 1 load itk image and truncate value with upper and lower
    center = 35  # 窗位
    width = 350  # 窗宽
    min = (2 * center - width) / 2.0 + 0.5
    max = (2 * center + width) / 2.0 + 0.5
    dFactor = 255.0 / (max - min)
    src = load_itkfilewithtruncation(file_image, max, min, dFactor)
    originSize = src.GetSize()
    originSpacing = src.GetSpacing()
    thickspacing, widthspacing, heightspacing = originSpacing[0], originSpacing[1], originSpacing[2]
    if originSize[1] != 512 or originSize[2] != 512:
        _, sub_src = resize_image_itkwithsize(src, newSize=(originSize[0], 512, 512),
                                            originSize=[originSize[0], originSize[1],originSize[2]],
                                            originSpcaing=(thickspacing, widthspacing, heightspacing),
                                            resamplemethod=sitk.sitkLinear)
    # 2 change z spacing >1.0 to 1.0
    if thickspacing > 1.0:
        _, src = resize_image_itk(src, newSpacing=(1.0, widthspacing, widthspacing),
                                originSpcaing=(thickspacing, widthspacing, widthspacing),
                                resamplemethod=sitk.sitkLinear)

    # 3 get resample array(image and segmask)
    srcimg = sitk.GetArrayFromImage(src)
    srcimg = np.swapaxes(srcimg, 0, 2)
    index = np.shape(srcimg)[0]
    k_3d_pd_arr = np.zeros(np.shape(srcimg), np.uint8)
    _, H, W = srcimg.shape
    print(k_3d_pd_arr.shape)

    print("step 1-1 begin")
    last_depth = 0
    for depth in range(0, index // depth_z, 1):
        for height in range(0,H//height_h,1):
            for width in range(0,W//width_w,1):
                patch_xs = srcimg[depth * depth_z:(depth + 1) * depth_z, height * height_h:(height+1) * height_h, width * width_w:(width+1) * width_w]
                patch_pd = test_3ddice(patch_xs,modelpath="C:\CSUProgram\Kits\log\kidney\\3d\\model.pth")
                np.reshape(patch_pd,(depth_z, height_h, width_w))
                k_3d_pd_arr[depth * depth_z:(depth + 1) * depth_z, height * height_h:(height+1) * height_h, width * width_w:(width+1) * width_w] = patch_pd
        last_depth = depth
    if index != depth_z * last_depth:
        for height in range(0,H//height_h,1):
            for width in range(0,W//width_w,1):
                patch_xs = srcimg[(index - depth_z):index, height * height_h:(height+1) * height_h, width * width_w:(width+1) * width_w]
                patch_pd = test_3ddice(patch_xs,modelpath="C:\CSUProgram\Kits\log\kidney\\3d\\model.pth")
                k_3d_pd_arr[(index - depth_z):index, height * height_h:(height+1) * height_h, width * width_w:(width+1) * width_w] = patch_pd

    ys_pd_sitk = sitk.GetImageFromArray(k_3d_pd_arr)
    k_3d_pd_arr = removesmallConnectedCompont(ys_pd_sitk, 0.2)
    k_3d_pd_arr = np.clip(k_3d_pd_arr, 0, 255).astype('uint8')

    # outputbmp(k_3d_pd_arr,"result3dkidney")

    print("prediction 3d kindey end:", np.shape(k_3d_pd_arr))
    # step2 predict 2d fine kindey

    index = np.shape(srcimg)[0]
    print("step 1-2 begin")
    k_2d_pd_arr = np.zeros(np.shape(srcimg), np.uint8)

    for z in range(index):
        if np.max(k_3d_pd_arr[z]) != 0:
            result = test_2ddice(srcimg[z], modelpath="C:\CSUProgram\Kits\log\kidney\\2d\\model.pth")[0,0]
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(15, 15))
            closedresult = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
            k_2d_pd_arr[z, :, :] = closedresult

    mask = k_2d_pd_arr.copy()
    mask[k_2d_pd_arr > 0] = 255
    resultk = np.clip(mask, 0, 255).astype('uint8')

    # outputbmp(resultk, "result2dkidney")

    print("prediction 2d kindey end:", np.shape(resultk))

    '''
    #step3  predict 3d fine tumor
    depth_z = 16
    print("step 1-3 begin")
    fine_t_pd_arr = np.empty((index, 512, 512), np.uint8)

    last_depth = 0
    for depth in range(0, index // depth_z, 1):
        for height in range(0, H // height_h, 1):
            for width in range(0, W // width_w, 1):
                patch_xs = srcimg[depth * depth_z:(depth + 1) * depth_z, height * height_h:(height + 1) * height_h, width * width_w:(width + 1) * width_w]
                patch_pd = test_3ddice(patch_xs, modelpath="C:\CSUProgram\Kits\log\\tumor\\3d\\model.pth")
                np.reshape(patch_pd, (depth_z, height_h, width_w))
                fine_t_pd_arr[depth * depth_z:(depth + 1) * depth_z, height * height_h:(height + 1) * height_h, width * width_w:(width + 1) * width_w] = patch_pd
        last_depth = depth
    if index != depth_z * last_depth:
        for height in range(0, H // height_h, 1):
            for width in range(0, W // width_w, 1):
                patch_xs = srcimg[(index - depth_z):index, height * height_h:(height + 1) * height_h, width * width_w:(width + 1) * width_w]
                patch_pd = test_3ddice(patch_xs, modelpath="C:\CSUProgram\Kits\log\\tumor\\3d\\model.pth")
                fine_t_pd_arr[(index - depth_z):index, height * height_h:(height + 1) * height_h,width * width_w:(width + 1) * width_w] = patch_pd

    fine_t_pd_arr = np.clip(fine_t_pd_arr, 0, 255).astype('uint8')
    startposition,endposition = getRangeImageDepth(k_3d_pd_arr)
    # get expand roi
    startposition = startposition - 5
    endposition = endposition + 5
    if startposition < 0:
        startposition = 0
    if endposition >= index:
        endposition = index
    print("start,end",startposition,endposition)
    t_3d_pd_arr = np.zeros((index, 512, 512), np.uint8)
    t_3d_pd_arr[startposition:endposition] = fine_t_pd_arr[startposition:endposition]

    result_path = "C:\CSUProgram\Kits\data\\testdata\\result\\result3dtumor\\"
    result_array = np.array(t_3d_pd_arr)
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    for i in range(np.shape(result_array)[0]):
        cv2.imwrite(result_path + "/" + str(i) + ".bmp", result_array[i])

    print("prediction 3d tumor end:", np.shape(t_3d_pd_arr))
    '''
    t_3d_pd_arr = np.empty((index, 512, 512), np.uint8)

    # step4 predict 2d fine tumor
    print("step 1-4 begin")
    t_2d_pd_arr = np.zeros(np.shape(srcimg), np.uint8)

    for z in range(index):
        if np.max(k_3d_pd_arr[z]) != 0:
            result = test_2ddice(srcimg[z], modelpath="C:\CSUProgram\Kits\log\\tumor\\2d\\model.pth")[0,0]
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
            closedresult = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
            t_2d_pd_arr[z, :, :] = closedresult

    mask = t_2d_pd_arr.copy()
    mask[t_2d_pd_arr > 0] = 255
    resultt = np.clip(mask, 0, 255).astype('uint8')

    # outputbmp(resultt, "result2dtumor")

    print("prediction 2d tumor end:", np.shape(resultt))

    return srcimg, k_3d_pd_arr, resultk, t_3d_pd_arr, resultt

#step 二：结果处理

# step 1
def removekidneysmallobj(fine_k, k_2d, t_3d, t_2d):
    """
    去除3d,2dVnet肾脏结果的小物体
    :return:
    """
    print("step 2-1",np.array(fine_k).shape)
    kindey3d_array = np.array(fine_k)
    index = kindey3d_array.shape[0]
    kindey3d_array = np.reshape(kindey3d_array, (index, 512, 512))
    kindey3d_sitk = sitk.GetImageFromArray(kindey3d_array)
    kindey3d_array = removesmallConnectedCompont(kindey3d_sitk, 0.2)
    kindey3d_array = np.clip(kindey3d_array, 0, 255).astype('uint8')

    kindey2d_array = np.array(k_2d)
    kindey2d_array = np.reshape(kindey2d_array, (index, 512, 512))
    kindey2d_sitk = sitk.GetImageFromArray(kindey2d_array)
    kindey2d_array = removesmallConnectedCompont(kindey2d_sitk, 0.2)
    kindey2d_array = np.clip(kindey2d_array, 0, 255).astype('uint8')
    return kindey3d_array, kindey2d_array

# step 2
def kindey2d3doverlap(fine_k, k_2d, t_3d, t_2d):
    """
    求2d和3dVnet的肾脏区域的交集结果
    :return:
    """
    k_3d, k_2d = removekidneysmallobj(fine_k, k_2d, t_3d, t_2d)
    print("step 2-2")
    print("2d tumor shape:", np.array(k_2d).shape)
    print("3d tumor shape:", np.array(k_3d).shape)
    index = np.array(k_3d).shape[0]
    tumor3d_array = np.array(k_3d)
    tumor3d_array = np.reshape(tumor3d_array, (index, 512, 512))
    tumor2d_array = np.array(k_2d)
    tumor2d_array = np.reshape(tumor2d_array, (index, 512, 512))
    tumor_array = tumor3d_array & tumor2d_array
    tumor_array_sitk = sitk.GetImageFromArray(tumor_array)
    tumor_array = removesmallConnectedCompont(tumor_array_sitk, 0.2)
    tumor_array = np.clip(tumor_array, 0, 255).astype('uint8')
    return tumor_array

# step 3
def kindey2d3dmerge(fine_k, k_2d, t_3d, t_2d):
    """
    求2d和3d交集的区域分别与2d和3d相连接的区域都保留下来
    :return:
    """
    print("step 2-3")
    k_3d, k_2d = removekidneysmallobj(fine_k, k_2d, t_3d, t_2d)
    k_23d = kindey2d3doverlap(fine_k, k_2d, t_3d, t_2d)
    index = np.array(k_3d).shape[0]
    kindey3d_array = np.array(k_3d)
    kindey3d_array = np.reshape(kindey3d_array, (index, 512, 512))
    kindey2d_array = np.array(k_2d)
    kindey2d_array = np.reshape(kindey2d_array, (index, 512, 512))
    kindey2d3d_array = np.array(k_23d)
    kindey2d3d_array = np.reshape(kindey2d3d_array, (index, 512, 512))
    kindey_array = np.zeros((index, 512, 512), int)
    for z in range(index):
        kindey2d3d = kindey2d3d_array[z]
        if np.max(kindey2d3d) != 0:
            kindey2d3dlabels, kindey2d3dout = cv2.connectedComponents(kindey2d3d)
            kindey3d = kindey3d_array[z]
            kindey3dlabels, kindey3dout = cv2.connectedComponents(kindey3d)
            kindey2d = kindey2d_array[z]
            kindey2dlabels, kindey2dout = cv2.connectedComponents(kindey2d)

            for i in range(1, kindey2d3dlabels):
                kindey2d3doutmask = np.zeros(kindey2d3dout.shape, int)
                kindey2d3doutmask[kindey2d3dout == i] = 255
                for j in range(1, kindey3dlabels):
                    kindey3doutmask = np.zeros(kindey3dout.shape, int)
                    kindey3doutmask[kindey3dout == j] = 255
                    if cv2.countNonZero(kindey2d3doutmask & kindey3doutmask):
                        kindey_array[z] = kindey_array[z] + kindey3doutmask
                for k in range(1, kindey2dlabels):
                    kindey2doutmask = np.zeros(kindey2dout.shape, int)
                    kindey2doutmask[kindey2dout == k] = 255
                    if cv2.countNonZero(kindey2d3doutmask & kindey2doutmask):
                        kindey_array[z] = kindey_array[z] + kindey2doutmask

    kindey2d_array_sitk = sitk.GetImageFromArray(kindey_array)
    kindey2d_array = removesmallConnectedCompont(kindey2d_array_sitk, 0.2)
    kindey_array = np.clip(kindey2d_array, 0, 255).astype('uint8')
    return kindey_array


# step 4
def remove2d3dtumorsmallobj(fine_k, k_2d, t_3d, t_2d):
    """
    去除2d和3dVnet的肿瘤的小物体
    :return:
    """
    print("step 2-4")
    print("2d tumor shape:",np.array(t_2d).shape)
    print("3d tumor shape:", np.array(t_3d).shape)
    index = np.array(t_3d).shape[0]
    tumor3d_array = np.array(t_3d)
    tumor3d_array = np.reshape(tumor3d_array, (index, 512, 512))
    tumor2d_array = np.array(t_2d)
    tumor2d_array = np.reshape(tumor2d_array, (index, 512, 512))
    tumor2d_array_sitk = sitk.GetImageFromArray(tumor2d_array)
    tumor2d_array = removesmallConnectedCompont(tumor2d_array_sitk, 0.2)
    tumor3d_array_sitk = sitk.GetImageFromArray(tumor3d_array)
    tumor3d_array = removesmallConnectedCompont(tumor3d_array_sitk, 0.2)
    tumor2d_array = np.clip(tumor2d_array, 0, 255).astype('uint8')
    tumor3d_array = np.clip(tumor3d_array, 0, 255).astype('uint8')
    return tumor3d_array, tumor2d_array

# step 5
def tumor2d3doverlap(fine_k, k_2d, t_3d, t_2d):
    """
    求2d和3dVnet的肿瘤区域的交集结果
    :return:
    """
    t_3d, t_2d = remove2d3dtumorsmallobj(fine_k, k_2d, t_3d, t_2d)
    print("step 2-5")
    print("2d tumor shape:", np.array(t_2d).shape)
    print("3d tumor shape:", np.array(t_3d).shape)
    index = np.array(t_3d).shape[0]
    tumor3d_array = np.array(t_3d)
    tumor3d_array = np.reshape(tumor3d_array, (index, 512, 512))
    tumor2d_array = np.array(t_2d)
    tumor2d_array = np.reshape(tumor2d_array, (index, 512, 512))
    tumor_array = tumor3d_array & tumor2d_array
    tumor_array_sitk = sitk.GetImageFromArray(tumor_array)
    tumor_array = removesmallConnectedCompont(tumor_array_sitk, 0.2)
    tumor_array = np.clip(tumor_array, 0, 255).astype('uint8')
    return tumor_array


# step 6
def tumor2d3dmerge(fine_k, k_2d, t_3d, t_2d):
    """
    求2d和3d交集的区域分别与2d和3d相连接的区域都保留下来
    :return:
    """
    print("step 2-6")
    t_3d, t_2d = remove2d3dtumorsmallobj(fine_k, k_2d, t_3d, t_2d)
    t_23d = tumor2d3doverlap(fine_k, k_2d, t_3d, t_2d)
    index = np.array(t_3d).shape[0]
    tumor3d_array = np.array(t_3d)
    tumor3d_array = np.reshape(tumor3d_array, (index, 512, 512))
    tumor2d_array = np.array(t_2d)
    tumor2d_array = np.reshape(tumor2d_array, (index, 512, 512))
    tumor2d3d_array = np.array(t_23d)
    tumor2d3d_array = np.reshape(tumor2d3d_array, (index, 512, 512))
    tumor_array = np.zeros((index, 512, 512), int)
    for z in range(index):
        tumor2d3d = tumor2d3d_array[z]
        if np.max(tumor2d3d) != 0:
            tumor2d3dlabels, tumor2d3dout = cv2.connectedComponents(tumor2d3d)
            tumor3d = tumor3d_array[z]
            tumor3dlabels, tumor3dout = cv2.connectedComponents(tumor3d)
            tumor2d = tumor2d_array[z]
            tumor2dlabels, tumor2dout = cv2.connectedComponents(tumor2d)

            for i in range(1, tumor2d3dlabels):
                tumor2d3doutmask = np.zeros(tumor2d3dout.shape, int)
                tumor2d3doutmask[tumor2d3dout == i] = 255
                for j in range(1, tumor3dlabels):
                    tumor3doutmask = np.zeros(tumor3dout.shape, int)
                    tumor3doutmask[tumor3dout == j] = 255
                    if cv2.countNonZero(tumor2d3doutmask & tumor3doutmask):
                        tumor_array[z] = tumor_array[z] + tumor3doutmask
                for k in range(1, tumor2dlabels):
                    tumor2doutmask = np.zeros(tumor2dout.shape, int)
                    tumor2doutmask[tumor2dout == k] = 255
                    if cv2.countNonZero(tumor2d3doutmask & tumor2doutmask):
                        tumor_array[z] = tumor_array[z] + tumor2doutmask

    tumor2d_array_sitk = sitk.GetImageFromArray(tumor_array)
    tumor2d_array = removesmallConnectedCompont(tumor2d_array_sitk, 0.2)
    tumor_array = np.clip(tumor2d_array, 0, 255).astype('uint8')
    return tumor_array


# step 7
def tumor2d3dinkidney(fine_k, k_2d, t_3d, t_2d):
    """
    保留肾脏区域范围内的肿瘤2d和3d结果
    :return:
    """
    print("step 2-7")
    k = kindey2d3dmerge(fine_k, k_2d, t_3d, t_2d)
    t_3d, t_2d = remove2d3dtumorsmallobj(fine_k, k_2d, t_3d, t_2d)
    index = np.array(k).shape[0]

    kidneys_array = np.array(k)
    kidneys_array = np.reshape(kidneys_array, (index, 512, 512))
    kidneys_array_sitk = sitk.GetImageFromArray(kidneys_array)
    kidneys_array = morphologicaloperation(kidneys_array_sitk, 5, "dilate")

    tumor3d_array = np.array(t_3d)
    tumor3d_array = np.reshape(tumor3d_array, (index, 512, 512))

    tumor2d_array = np.array(t_2d)
    tumor2d_array = np.reshape(tumor2d_array, (index, 512, 512))

    tumor3d_array = tumor3d_array & kidneys_array
    tumor2d_array = tumor2d_array & kidneys_array

    tumor_array = tumor3d_array | tumor2d_array
    tumor_array_sitk = sitk.GetImageFromArray(tumor_array)
    tumor_array = removesmallConnectedCompont(tumor_array_sitk, 0.1)
    tumor_array = np.clip(tumor_array, 0, 255).astype('uint8')
    return tumor_array


# step 8
def tumormodifyallmerge(fine_k, k_2d, t_3d, t_2d):
    """
    将肾脏区域内的2d和3d肿瘤区域并集结果与2d和3d肿瘤交集结果相加
    :return:
    """
    print("step 2-8")
    images = tumor2d3dinkidney(fine_k, k_2d, t_3d, t_2d)
    masks = tumor2d3dmerge(fine_k, k_2d, t_3d, t_2d)
    index = np.array(images).shape[0]

    tumor3d_array = np.array(images)
    tumor3d_array = np.reshape(tumor3d_array, (index, 512, 512))
    tumor2d_array = np.array(masks)
    tumor2d_array = np.reshape(tumor2d_array, (index, 512, 512))
    tumor_array = tumor2d_array + tumor3d_array
    tumor_array = np.clip(tumor_array, 0, 255).astype('uint8')
    return tumor_array

# 计算左右肿瘤最长径和面积：
def calMaxlenAndArea(tumor):
    # 闭运算去除内部空洞
    _, binary = cv2.threshold(tumor, 127, 255, cv2.THRESH_BINARY)
    bin = np.array(binary)
    k = np.ones((10, 10), np.uint8)
    close = cv2.morphologyEx(bin, cv2.MORPH_CLOSE, k, iterations=3)
    contours, hierarchy = cv2.findContours(close, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    number = len(contours)
    # print('subject number', number)
    leftnum = 0
    leftarea = []
    leftmaxlen = 0
    rightnum = 0
    rightarea = []
    rightmaxlen = 0
    for c in range(len(contours)):
        # 取出一条轮廓
        cnt = contours[c]
        # 分别计算轮廓面积与最长轴长度
        if (contours[c][0][0][0] < int((tumor.shape[0]) / 2)):
            leftnum += 1
            # 求面积
            cnt_area = cv2.contourArea(cnt)
            real_area = cnt_area * 0.09199219 * 0.09199219
            leftarea.append(real_area)
            # 获得最小外接圆直径
            center, radius = cv2.minEnclosingCircle(cnt)
            leftmaxlen = 2 * radius * 0.09199219
        else:
            rightnum += 1
            # 求面积
            cnt_area = cv2.contourArea(cnt)
            real_area = cnt_area * 0.09199219 * 0.09199219
            rightarea.append(real_area)
            # 获得最小外接圆直径
            center, radius = cv2.minEnclosingCircle(cnt)
            rightmaxlen = 2 * radius * 0.09199219
    if (leftnum == 0 or rightnum == 0):
        if (leftnum == 0):
            leftarea.append(0)
            leftmaxlen = 0
        else:
            rightarea.append(0)
            rightmaxlen = 0
    return leftarea, leftmaxlen, rightarea, rightmaxlen


# step 9
def outputresult():
    """
    将最后肾脏结果和肿瘤结果输出成nii
    :return:
    """
    kits_path = "C:\CSUProgram\Kits\data\kits"
    kidney_path = "C:\CSUProgram\Kits\data\\testdata\\kidneypred"
    tumor_path = "C:\CSUProgram\Kits\data\\testdata\\tumorpred"
    image_name = "imaging.nii.gz"
    result_path = "C:\CSUProgram\Kits\data\\testdata\\result"

    height, width = 512, 512
    """
    load itk image,change z Spacing value to 1,and save image ,liver mask ,tumor mask
    :return:None
    """
    # step2 get all train image
    path_list = file_name_path(kits_path)
    read = open("kidneyrange.txt", 'r')
    # step3 get signal train image and mask
    for subsetindex in range(len(path_list)):
        line = read.readline()
        line = line.split(',')
        casename = line[0]
        start = int(line[1])
        end = int(line[2][0:-1])
        kits_subset_path = kits_path + "/" + str(path_list[subsetindex]) + "/"
        file_image = kits_subset_path + image_name

        kidney_mask_path = kidney_path + "/" + str(path_list[subsetindex]) + "/"
        tumor_mask_path = tumor_path + "/" + str(path_list[subsetindex]) + "/"
        index = 0
        kidneylist = []
        tumorlist = []
        for _ in os.listdir(kidney_mask_path):
            image = cv2.imread(kidney_mask_path + "/" + str(index) + ".bmp", cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(tumor_mask_path + "/" + str(index) + ".bmp", cv2.IMREAD_GRAYSCALE)
            kidneylist.append(image)
            tumorlist.append(mask)
            index += 1

        kidneyarray = np.array(kidneylist)
        kidneyarray = np.reshape(kidneyarray, (index, height, width))
        tumorarray = np.array(tumorlist)
        tumorarray = np.reshape(tumorarray, (index, height, width))
        outmask = np.zeros((index, height, width), np.uint8)
        outmask[kidneyarray == 255] = 1
        outmask[tumorarray == 255] = 2
        # 1 load itk image and truncate value with upper and lower and get rang kideny region
        src = load_itkfilewithtruncation(file_image, 300, -200)
        originSize = src.GetSize()
        originSpacing = src.GetSpacing()
        thickspacing, widthspacing = originSpacing[0], originSpacing[1]
        outmask = np.swapaxes(outmask, 0, 2)
        mask_sitk = sitk.GetImageFromArray(outmask)
        mask_sitk.SetSpacing((1.0, widthspacing, widthspacing))
        mask_sitk.SetOrigin(src.GetOrigin())
        mask_sitk.SetDirection(src.GetDirection())
        # 2 change z spacing >1.0 to originspacing
        if thickspacing > 1.0:
            _, mask_sitk = resize_image_itk(mask_sitk, newSpacing=(thickspacing, widthspacing, widthspacing),
                                            originSpcaing=(1.0, widthspacing, widthspacing),
                                            resamplemethod=sitk.sitkLinear)
        else:
            mask_sitk.SetSpacing(originSpacing)
        src_mask_array = np.zeros((originSize[0], height, width), np.uint8)
        mask_array = sitk.GetArrayFromImage(mask_sitk)
        mask_array = np.swapaxes(mask_array, 0, 2)
        # make sure the subregion have same size
        if (end - start) != np.shape(mask_array)[0]:
            start = start
            end = start + np.shape(mask_array)[0]
            if end > originSize[0]:
                end = originSize[0]
                start = end - np.shape(mask_array)[0]
        src_mask_array[start:end] = mask_array
        src_mask_array = np.swapaxes(src_mask_array, 0, 2)
        mask_itk = sitk.GetImageFromArray(src_mask_array)
        mask_itk.SetSpacing(originSpacing)
        mask_itk.SetOrigin(src.GetOrigin())
        mask_itk.SetDirection(src.GetDirection())
        mask_name = result_path + "/" + "prediction" + casename[4:10] + ".nii.gz"
        sitk.WriteImage(mask_itk, mask_name)

def resultoutkindey(fine_k, k_2d, t_3d, t_2d):
    result = kindey2d3dmerge(fine_k, k_2d, t_3d, t_2d)
    outputbmp(result, "resultkidney")

def resultouttumor(fine_k, k_2d, t_3d, t_2d):
    # result = tumormodifyallmerge(fine_k, k_2d, t_3d, t_2d)
    result = t_2d
    outputbmp(result, "resulttumor")

def Proresult(src,resultk, resultt):
    Z, H, W = src.shape
    srck0t0 = np.zeros((Z, H, W, 3), np.uint8)
    srck1t0 = np.zeros((Z, H, W, 3), np.uint8)
    srck0t1 = np.zeros((Z, H, W, 3), np.uint8)
    srck1t1 = np.zeros((Z, H, W, 3), np.uint8)
    leftMaxlenList = []
    leftAreaList = []
    rightMaxlenList = []
    rightAreaList = []
    print("shape",srck0t0.shape)
    for z in range(src.shape[0]):
        srck0t0[z] = cv2.cvtColor(src[z],cv2.COLOR_GRAY2RGB)
        srck1t0[z] = cv2.cvtColor(src[z],cv2.COLOR_GRAY2RGB)
        srck0t1[z] = cv2.cvtColor(src[z],cv2.COLOR_GRAY2RGB)
        srck1t1[z] = cv2.cvtColor(src[z],cv2.COLOR_GRAY2RGB)
        resultkZ = resultk[z]
        resulttZ = resultt[z]
        leftarea, leftmaxlen, rightarea, rightmaxlen = calMaxlenAndArea(resulttZ)
        leftMaxlenList.append(leftmaxlen)
        leftAreaList.append(leftarea)
        rightMaxlenList.append(rightmaxlen)
        rightAreaList.append(rightarea)
        # print("copy shape:",srck1t1.shape)
        # k1t0
        srck1t0[z, :, :][resultkZ != 0] = [144, 180, 75]
        # k0t1
        srck0t1[z, :, :][resulttZ != 0] = [58, 84, 204]
        # k1t1
        srck1t1[z, :, :][resultkZ != 0] = [144, 180, 75]
        srck1t1[z, :, :][resulttZ != 0] = [58, 84, 204]

    print("area list:",len(rightAreaList))
    print(srck1t1.shape)
    # outputbmp(srck0t0, "resultk0t0")
    # outputbmp(srck1t0, "resultk1t0")
    # outputbmp(srck0t1, "resultk0t1")
    # outputbmp(srck1t1, "resultk1t1")
    return resultk, resultt, srck0t0, srck1t0, srck0t1, srck1t1, leftMaxlenList, leftAreaList, rightMaxlenList, rightAreaList

def changeWL(window,level,file,resultk, resultt):
    center = level  # 窗位
    width = window  # 窗宽
    min = (2 * center - width) / 2.0 + 0.5
    max = (2 * center + width) / 2.0 + 0.5
    dFactor = 255.0 / (max - min)
    src = load_itkfilewithtruncation(file, max, min, dFactor)
    src = sitk.GetArrayFromImage(src)
    src = np.swapaxes(src, 0, 2)
    print(src.shape)
    Z, H, W = src.shape
    srck0t0 = np.zeros((Z, H, W, 3), np.uint8)
    srck1t0 = np.zeros((Z, H, W, 3), np.uint8)
    srck0t1 = np.zeros((Z, H, W, 3), np.uint8)
    srck1t1 = np.zeros((Z, H, W, 3), np.uint8)

    for z in range(src.shape[0]):
        srck0t0[z] = cv2.cvtColor(src[z],cv2.COLOR_GRAY2RGB)
        srck1t0[z] = cv2.cvtColor(src[z],cv2.COLOR_GRAY2RGB)
        srck0t1[z] = cv2.cvtColor(src[z],cv2.COLOR_GRAY2RGB)
        srck1t1[z] = cv2.cvtColor(src[z],cv2.COLOR_GRAY2RGB)
        resultkZ = resultk[z]
        resulttZ = resultt[z]
        # print("copy shape:",srck1t1.shape)
        # k1t0
        srck1t0[z, :, :][resultkZ != 0] = [144, 180, 75]
        # k0t1
        srck0t1[z, :, :][resulttZ != 0] = [58, 84, 204]
        # k1t1
        srck1t1[z, :, :][resultkZ != 0] = [144, 180, 75]
        srck1t1[z, :, :][resulttZ != 0] = [58, 84, 204]
    return srck0t0, srck1t0, srck0t1, srck1t1
