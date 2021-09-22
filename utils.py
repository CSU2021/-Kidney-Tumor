from __future__ import print_function, division
import SimpleITK as sitk
import numpy as np
import cv2
import os
from glob import glob


def file_name_path(file_dir, dir=True, file=False):
    """
    get root path,sub_dirs,all_sub_files
    :param file_dir:
    :return:
    """
    for root, dirs, files in os.walk(file_dir):
        if len(dirs) and dir:
            # print("sub_dirs:", dirs)
            return dirs
        if len(files) and file:
            # print("files:", files)
            return files


def save_file2csv(file_dir, file_name):
    """
    save file path to csv
    :param file_dir:preprocess data path
    :param file_name:output csv name
    :return:
    """
    out = open(file_name, 'w')
    image = "Image"
    mask = "Mask"
    file_image_dir = file_dir + "/" + image
    file_mask_dir = file_dir + "/" + mask
    file_paths = file_name_path(file_image_dir, dir=False, file=True)
    out.writelines("Image,Mask" + "\n")
    for index in range(len(file_paths)):
        out_file_image_path = file_image_dir + "/" + file_paths[index]
        out_file_mask_path = file_mask_dir + "/" + file_paths[index]
        out.writelines(out_file_image_path + "," + out_file_mask_path + "\n")

def getImagexy(image_src, fixedvalue=255):
    """
    :param image:
    :return:(x,y)start
    """
    # startposition, endposition = np.where(image)[0][[0, -1]]
    image = image_src.copy()
    image[image_src == fixedvalue] = 255
    image[image_src != fixedvalue] = 0
    print(image.shape)
    xposition = 0
    yposition = 0
    for z in range(image.shape[0]):
        for y in range(image.shape[1]):
            for x in range(image.shape[2]):
                notzeroflag = image[z, y, x]
                if notzeroflag:
                    xposition = x
                    yposition = y
                    break
    if xposition < 64:
        xposition = 64
    if yposition < 64:
        yposition = 64
    return xposition, yposition

def getRangeImageDepth(image):
    """
    :param image:
    :return:rang of image depth
    """
    # startposition, endposition = np.where(image)[0][[0, -1]]
    fistflag = True
    startposition = 0
    endposition = 0
    print("z:",image.shape[0])
    for z in range(image.shape[0]):
        notzeroflag = np.max(image[z, :, :])
        if notzeroflag and fistflag:
            startposition = z
            fistflag = False
        if notzeroflag:
            endposition = z
    return startposition, endposition

def load_itkfilewithtruncation(filename, upper=200, lower=-200, dFactor=0.72):
    """
    load mhd files,set truncated value range and normalization 0-255
    :param filename:
    :param upper:
    :param lower:
    :return:
    """
    srcitkimage = sitk.Cast(sitk.ReadImage(filename), sitk.sitkFloat32)
    srcitkimagearray = sitk.GetArrayFromImage(srcitkimage)
    srcitkimagearray = srcitkimagearray - lower
    srcitkimagearray = np.trunc(srcitkimagearray * dFactor)
    srcitkimagearray[srcitkimagearray < 0.0] = 0
    srcitkimagearray[srcitkimagearray > 255.0] = 255
    # 2,get tructed outside of liver value image
    sitktructedimage = sitk.GetImageFromArray(srcitkimagearray)
    origin = np.array(srcitkimage.GetOrigin())
    spacing = np.array(srcitkimage.GetSpacing())
    sitktructedimage.SetSpacing(spacing)
    sitktructedimage.SetOrigin(origin)
    # 3 normalization value to 0-255
    rescalFilt = sitk.RescaleIntensityImageFilter()
    rescalFilt.SetOutputMaximum(255)
    rescalFilt.SetOutputMinimum(0)
    itkimage = rescalFilt.Execute(sitk.Cast(sitktructedimage, sitk.sitkFloat32))
    return itkimage

def resize_image_itkwithsize(itkimage, newSize, originSize, originSpcaing, resamplemethod=sitk.sitkNearestNeighbor):
    """
    image resize withe sitk resampleImageFilter
    :param itkimage:
    :param newSize:such as [1,1,1]
    :param resamplemethod:
    :return:
    """
    resampler = sitk.ResampleImageFilter()
    originSize = np.array(originSize)
    newSize = np.array(newSize)
    factor = originSize / newSize
    newSpacing = factor * originSpcaing
    resampler.SetReferenceImage(itkimage)
    resampler.SetOutputSpacing(newSpacing.tolist())
    resampler.SetSize(newSize.tolist())
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(resamplemethod)
    itkimgResampled = resampler.Execute(itkimage)
    if resamplemethod == sitk.sitkNearestNeighbor:
        itkimgResampled = sitk.Threshold(itkimgResampled, 0, 1.0, 255)
    imgResampled = sitk.GetArrayFromImage(itkimgResampled)
    return imgResampled, itkimgResampled

def resize_image_itk(itkimage, newSpacing, originSpcaing, resamplemethod=sitk.sitkNearestNeighbor):
    """
    image resize withe sitk resampleImageFilter
    :param itkimage:
    :param newSpacing:such as [1,1,1]
    :param resamplemethod:
    :return:
    """
    newSpacing = np.array(newSpacing, float)
    # originSpcaing = itkimage.GetSpacing()
    resampler = sitk.ResampleImageFilter()
    originSize = itkimage.GetSize()
    origin = itkimage.GetOrigin()
    origindirection = itkimage.GetDirection()
    factor = newSpacing / originSpcaing
    newSize = originSize / factor
    newSize = newSize.astype(np.int)
    resampler.SetReferenceImage(itkimage)
    resampler.SetOutputSpacing(newSpacing.tolist())
    resampler.SetSize(newSize.tolist())
    resampler.SetOutputOrigin(origin)
    resampler.SetOutputDirection(origindirection)
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(resamplemethod)
    itkimgResampled = resampler.Execute(itkimage)
    if resamplemethod == sitk.sitkNearestNeighbor:
        itkimgResampled = sitk.Threshold(itkimgResampled, 0, 1.0, 255)
    imgResampled = sitk.GetArrayFromImage(itkimgResampled)
    return imgResampled, itkimgResampled


def removesmallConnectedCompont(sitk_maskimg, rate=0.5):
    cc = sitk.ConnectedComponent(sitk_maskimg)
    stats = sitk.LabelIntensityStatisticsImageFilter()
    stats.Execute(cc, sitk_maskimg)
    maxlabel = 0
    maxsize = 0
    for l in stats.GetLabels():
        size = stats.GetPhysicalSize(l)
        if maxsize < size:
            maxlabel = l
            maxsize = size

    not_remove = []
    for l in stats.GetLabels():
        size = stats.GetPhysicalSize(l)
        if size > maxsize * rate:
            not_remove.append(l)
    labelmaskimage = sitk.GetArrayFromImage(cc)
    outmask = labelmaskimage.copy()
    outmask[labelmaskimage != maxlabel] = 0
    for i in range(len(not_remove)):
        outmask[labelmaskimage == not_remove[i]] = 255
    return outmask


def getLargestConnectedCompont(sitk_maskimg):
    cc = sitk.ConnectedComponent(sitk_maskimg)
    stats = sitk.LabelIntensityStatisticsImageFilter()
    stats.Execute(cc, sitk_maskimg)
    maxlabel = 0
    maxsize = 0
    for l in stats.GetLabels():
        size = stats.GetPhysicalSize(l)
        if maxsize < size:
            maxlabel = l
            maxsize = size

    labelmaskimage = sitk.GetArrayFromImage(cc)
    outmask = labelmaskimage.copy()
    outmask[labelmaskimage == maxlabel] = 255
    outmask[labelmaskimage != maxlabel] = 0
    return outmask


def morphologicaloperation(sitk_maskimg, kernelsize, name='open'):
    if name == 'open':
        morphoimage = sitk.BinaryMorphologicalOpening(sitk_maskimg, [kernelsize, kernelsize, kernelsize])
        labelmaskimage = sitk.GetArrayFromImage(morphoimage)
        outmask = labelmaskimage.copy()
        outmask[labelmaskimage == 1.0] = 255
        return outmask
    if name == 'close':
        morphoimage = sitk.BinaryMorphologicalClosing(sitk_maskimg, [kernelsize, kernelsize, kernelsize])
        labelmaskimage = sitk.GetArrayFromImage(morphoimage)
        outmask = labelmaskimage.copy()
        outmask[labelmaskimage == 1.0] = 255
        return outmask
    if name == 'dilate':
        morphoimage = sitk.BinaryDilate(sitk_maskimg, [kernelsize, kernelsize, kernelsize])
        labelmaskimage = sitk.GetArrayFromImage(morphoimage)
        outmask = labelmaskimage.copy()
        outmask[labelmaskimage == 1.0] = 255
        return outmask
    if name == 'erode':
        morphoimage = sitk.BinaryErode(sitk_maskimg, [kernelsize, kernelsize, kernelsize])
        labelmaskimage = sitk.GetArrayFromImage(morphoimage)
        outmask = labelmaskimage.copy()
        outmask[labelmaskimage == 1.0] = 255
        return outmask



def save_npy2csv(path, name, labelnum=1):
    """
    this is for classify,save label+filepath into csv
    """
    out = open(name, 'w')
    file_list = glob(path + "*.npy")
    out.writelines("index" + "," + "filename" + "\n")
    for index in range(len(file_list)):
        out.writelines(str(labelnum) + "," + file_list[index] + "\n")

def outputbmp(result, path):
    result_path = "C:\CSUProgram\Kits\data\\testdata\\result\\" + str(path) + "\\"
    result_array = np.array(result)
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    for i in range(np.shape(result_array)[0]):
        cv2.imwrite(result_path + "/" + str(i) + ".bmp", result_array[i])

# gettestiamge()
# getmaxsizeimage()
# save_npy2csv("G:\Data\LIDC\LUNA16\classsification\\1_aug\\", "nodel_positive.csv", 1)



