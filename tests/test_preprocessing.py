"""
Script for unit testing preprocessing pipeline
"""
import os
import math
import pytest
import numpy as np
import SimpleITK as sitk
from sklearn.pipeline import Pipeline

from ..src.edema_maps_pipeline.processing import preprocessing
from ..src.edema_maps_pipeline import preprocessing_pipline
from ..src.edema_maps_pipeline.config.core import DATA_DIR, config
from ..src.edema_maps_pipeline.processing.data_manager import load_image
from ..src.edema_maps_pipeline.processing.data_manager import save_image

# Authors: Sara Ranjbar <ranjbar.sara@mayo.edu>


@pytest.fixture()
def test_image_file():
    print(DATA_DIR)
    impath = os.path.join(DATA_DIR, config.app_config.test_image_file)
    return load_image(impath)


@pytest.fixture()
def test_brainmask_file():
    impath = os.path.join(DATA_DIR, config.app_config.test_brainmask_file)
    return load_image(impath)


def test_preprocessing_step1_pipeline(test_image_file):

    # this is an example case.
    # if you have an image with any spacing or size and you
    # pass it on to forward processing pipeline (up to
    # normalization and skullstripping), what will you get?

    # Given
    target_spacing = preprocessing_pipline.MLSPACING
    target_size = preprocessing_pipline.MLSIZE

    # When
    result = preprocessing_pipline.image_prep.transform(test_image_file)

    # Then
    assert isinstance(result, sitk.Image)
    assert list(result.GetSpacing()) == target_spacing
    assert list(result.GetSize()) == target_size

    # save result
    tag = '_preprocessed.nii.gz'
    savename = config.app_config.test_image_file.split('.')[0] + tag
    savepath = os.path.join(DATA_DIR, savename)
    save_image(result, savepath)


def test_preprocessing_on_mask(test_brainmask_file):

    # this is an example case.
    # if you have a brain mask with any spacing or size and you
    # pass it on to forward processing pipeline,
    # what will you get?

    # Given
    target_spacing = preprocessing_pipline.MLSPACING
    target_size = preprocessing_pipline.MLSIZE

    # When
    result = preprocessing_pipline.mask_prep.transform(test_brainmask_file)

    # Then
    assert isinstance(result, sitk.Image)
    assert list(result.GetSpacing()) == target_spacing
    assert list(result.GetSize()) == target_size

    # save result
    tag = '_preprocessed.nii.gz'
    savename = config.app_config.test_brainmask_file.split('.')[0] + tag
    savepath = os.path.join(DATA_DIR, savename)
    save_image(result, savepath)


def test_preprocessing_step1n2_pipeline(test_image_file,
                                        test_brainmask_file):

    # this is an example case.
    # if you have an image with any spacing or size and you pass it on
    # to forward processing pipeline  (step1) and then send it to
    # preprocessing step 2 (\normalization and skullstripping), will you get
    # the correct result?
    # step2 has to have a separate pipeline, because mask will come from the
    # skullstripping model.

    # Given
    img_st1 = preprocessing_pipline.image_prep.transform(test_image_file)
    brain_st1 = preprocessing_pipline.image_prep.transform(test_brainmask_file)

    # When
    # Test for step 2 processing: normalization + skullstripping

    image_prep_st2 = Pipeline([
        ('normalize', preprocessing.IntensityNormalizer(mask=brain_st1)),
        ('skullstrip', preprocessing.ImageMasker(mask=brain_st1))
        ])

    result = image_prep_st2.transform(img_st1)
    result_np = sitk.GetArrayFromImage(result)
    mask_np = sitk.GetArrayFromImage(brain_st1)
    mask_locations = np.where(mask_np > 0)

    vol_samples = result_np[mask_locations]

    # Then
    assert isinstance(result, sitk.Image)
    assert math.isclose((np.mean(vol_samples)), 0.0, abs_tol=0.00001)
    assert math.isclose((np.std(vol_samples)), 1.0, abs_tol=0.00001)

    # save result
    tag = '_final.nii.gz'
    savename = config.app_config.test_image_file.split('.')[0] + tag
    savepath = os.path.join(DATA_DIR, savename)
    save_image(result, savepath)
