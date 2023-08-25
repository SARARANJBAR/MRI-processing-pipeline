""" scripts for image operations"""


import SimpleITK as sitk

# Authors: Sara Ranjbar <ranjbar.sara@mayo.edu>


def load_image(imagepath):
    return sitk.ReadImage(imagepath)


def save_image(image, outpath):
    sitk.WriteImage(image, outpath)
