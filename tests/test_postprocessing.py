"""
Script for unittesting postprocessing pipeline
"""
import os
import pytest

import SimpleITK as sitk
from sklearn.pipeline import Pipeline
from ..src.pipeline.processing import preprocessing
from ..src.pipeline.config.core import DATA_DIR, config
from ..src.pipeline.processing.data_manager import *

# Authors: Sara Ranjbar <ranjbar.sara@mayo.edu>


@pytest.fixture()
def sample_ml_output():
    impath = os.path.join(DATA_DIR, config.app_config.test_mloutput_file)
    return load_image(impath)


def test_postptocessing(sample_ml_output):

    # this is an example case.
    # if you have an image with base spacing below,
    # and you pass it on to forward processing pipeline,
    # after resampler it will have the intermediate_size

    # here we are trying to go backwards, meaning if
    # an ml output image has spacing of 1,1,2 and size of
    # 280,280,112 , how would we go back to original image
    # that was fed into the preprocessing pipeline

    # well, we would need to resize back to the size after resampling
    # and then resample back to the original spacing

    # Given
    # original image spacing of image in patient view
    base_spacing = [0.42, 0.42, 2.0]
    base_size = [512, 512, 90]

    # and size of said image after resampling to 1,1,2
    interim_size = [220, 220, 89]

    # When
    rev_pip = Pipeline([
        ('rev_resizer', preprocessing.Resizer(interim_size)),
        ('rev_resampler', preprocessing.Resampler(base_spacing)),
        ('rev_resizer2', preprocessing.Resizer(base_size)),
        ])

    result = rev_pip.transform(sample_ml_output)

    # Then
    assert isinstance(result, sitk.Image)
    assert list(result.GetSpacing()) == base_spacing
    assert list(result.GetSize()) == base_size

    # save result
    tag = '_postprocessed.nii.gz'
    savename = config.app_config.test_mloutput_file.split('.')[0] + tag
    savepath = os.path.join(DATA_DIR, savename)
    save_image(result, savepath)
