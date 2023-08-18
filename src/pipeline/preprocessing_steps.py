"""
Image and Mask preprocessing pipelines for applying ML models.
sklearn.pipeline is used for pipelining preprocessing steps.
calling pipeline.transform will run the transform method in all components.

This allows for easier saving, and therefore reproducibility of the
preprocessing pipeline
"""
# Authors: Sara Ranjbar <ranjbar.sara@mayo.edu>


from sklearn.pipeline import Pipeline
from .processing import preprocessing as pp
# from src.pipeline.config.core import config

# Preprocessing params
MLSPACING = [1., 1., 2.]
MLSIZE = [280, 280, 112]


# preprocessing steps when you have a brain mask (prior to feeding into models)
image_prep = Pipeline(
    [
        ('resampler', pp.Resampler(MLSPACING)),
        ('resizer', pp.Resizer(MLSIZE)),
        ('N4', pp.N4BiasCorrector()),
        ('denoiser', pp.Denoiser())
    ]
    )

# preprocessing steps when you dont have a brain mask
mask_prep = Pipeline(
    [
        ('resampler', pp.Resampler(MLSPACING)),
        ('resizer', pp.Resizer(MLSIZE)),
    ]
    )
