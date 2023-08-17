"""
Here, we use sklearn and its transformer classes to define classes for
MRI processing components.
Each transformer has some parameters, a fit method, and a transform method.

Inheritting from base class TransformerMinin, gives them a fit_transform
method as well. Inheritting from BaseEstimator allows easier initiation of
parameters.
--------------------------------------------------------------------------

each class follows this template:

class TransformerTemplate(BaseEstimator, TransformerMixin):
    # template transformer class
    def __init__(self, variables):
        # do variable checks
        self.variables = variables

    def fit(self, X, y=None):
        # we need this step to fit the sklearn pipeline
        # dont do anything here, but if you need to, here is where fitting
        # happens.
        return self

    def transform(self, X):
        # input has to be called X to match sklearn estimator syntax but can be
        # of anytype
        # i have a check input is stkimg at the top of all transform methods.

        if not isinstance(X, sitk.Image):
            raise ValueError('input to Resampler class should be a sitk image
            instance.')

        # write method here
        X_t = X

        # return transformed data
        return X_t

"""
import numpy as np
import SimpleITK as sitk
import os

from sklearn.base import BaseEstimator, TransformerMixin

# Authors:  Sara Ranjbar <ranjbar.sara@mayo.edu>
#           Kyle W. Singleton <singleton.kyle@mayo.edu>


class Resampler(BaseEstimator, TransformerMixin):
    # resampler class for adjusting image spacing

    def __init__(self, targetspacing):
        # raise error if it spacing is not any of the recognized types:
        # numpy array, numpy nd array, or list
        if not isinstance(targetspacing, np.ndarray):
            if not isinstance(targetspacing, list):
                raise ValueError('spacing should be a numpy array or list')

        # easier later if this is an array
        self.targetspacing = np.array(targetspacing)

        if len(targetspacing) != 3:
            raise ValueError('targetspacing should have 3 elements')

        # interpretor type is decided later based on input
        self.interpolator = None

    def _set_interpolator_for_image(self, X):
        # interpolator type depends on input

        # first find out if image is a mask
        # we dont tend to have a mask with 10 unique values
        nunique = np.unique(sitk.GetArrayFromImage(X).flatten())

        if len(nunique) < 10:
            self.interpolator = sitk.sitkNearestNeighbor
        else:
            self.interpolator = sitk.sitkBSpline

    def fit(self, X, y=None):
        # we need this step to fit the sklearn pipeline
        # dont do anything here
        return self

    def transform(self, X):
        # method-style functions here
        # will be applied to an input.
        # input has to be called X to match sklearn estimator syntax

        if isinstance(X, sitk.Image) is False:
            raise ValueError('Resampler input should be a sitkimage instance.')

        # Replace any in-plane spacing values of 0 with original image spacing
        self.targetspacing = np.where(self.targetspacing == 0, X.GetSpacing(),
                                      self.targetspacing)

        # set interpolator type based on image type (mask or mr)
        self._set_interpolator_for_image(X)

        orig_image_size = np.array(X.GetSize())
        new_image_size = np.multiply(orig_image_size,
                                     np.divide(X.GetSpacing(),
                                               self.targetspacing))
        new_image_size = new_image_size.astype(int).tolist()

        image_origin = np.array(X.GetOrigin())
        image_direction = np.array(X.GetDirection())
        image_pixel_type = X.GetPixelID()

        rif = sitk.ResampleImageFilter()
        rif.SetOutputSpacing(self.targetspacing)
        rif.SetOutputDirection(image_direction)
        rif.SetSize(new_image_size)
        rif.SetOutputOrigin(image_origin)
        rif.SetOutputPixelType(image_pixel_type)
        rif.SetInterpolator(self.interpolator)

        resampled_X = rif.Execute(X)

        # return the updated version of the input image
        return resampled_X


class Denoiser(BaseEstimator, TransformerMixin):
    # denoiser class
    def __init__(self, time_step=0.125, iterations=4):

        if time_step < 0.000001:
            raise ValueError("time_step is too small - %f" % time_step)
        self.time_step = time_step

        if iterations < 1:
            raise ValueError("denoise_sitk_image: iterations must be >= 1")
        self.niter = iterations

    def fit(self, X, y=None):
        # we need this step to fit the sklearn pipeline
        # dont do anything here, no fitting required
        return self

    def transform(self, X):
        # input has to be called X to match sklearn estimator syntax

        if not isinstance(X, sitk.Image):
            raise ValueError('Denoiser input should be a sitkimage instance.')

        X_t = sitk.CurvatureFlow(image1=X,
                                 timeStep=self.time_step,
                                 numberOfIterations=self.niter)
        return X_t


class N4BiasCorrector(BaseEstimator, TransformerMixin):
    # Bias corrector transformer class
    def __init__(self, shrink=4):

        if shrink < 1.0:
            raise ValueError("shrink factor too small: %f" % shrink)
        self.shrink = shrink

    def fit(self, X, y=None):
        # we need this step to fit the sklearn pipeline
        # dont do anything here, no fitting required
        return self

    def transform(self, X):
        # input has to be called X to match sklearn estimator syntax

        if not isinstance(X, sitk.Image):
            raise ValueError('N4BiasCorrector input should be a sitkimage.')

        pixel_id = X.GetPixelID()

        shrink_factors = [int(self.shrink)] * X.GetDimension()
        small_image = sitk.Shrink(X, shrink_factors)

        # Subtract off the minimum value in the volume.
        # This ensures that the smallest value is 0. Pixel intensities less
        # than 0 cause problems for the N4 filter.
        stats = sitk.StatisticsImageFilter()
        stats.Execute(X)
        vmin = stats.GetMinimum()
        small_image = small_image - vmin

        # Mask small_image to remove background voxels:
        # Setup an Otsu threshold filter to separate foreground (eg: head,
        # value = 255)
        # from background (eg: air, value = 0) by default, the Otsu threshold
        # returns the inverse of what we want. To fix this we have to reverse
        # the Inside and Outside values from their defaults of 1 and 0,
        # respectively, to 0 and 1, respectively. The procedural interface to
        # the Otsu filter does not seem to support setting Inside and Outside
        # values, even though the documentation suggests it should. So, use
        # the object oriented interface.
        otif = sitk.OtsuThresholdImageFilter()
        otif.SetInsideValue(0)
        otif.SetOutsideValue(1)
        small_mask = otif.Execute(small_image)
        small_image = sitk.Cast(small_image, sitk.sitkFloat32)

        # Perform the N4 intensity non-uniformity correction
        corrected_small_image = sitk.N4BiasFieldCorrection(small_image,
                                                           small_mask)
        small_bias_field = sitk.Subtract(small_image, corrected_small_image)

        # Expand the bias field and remove it from the original passed image
        bias_field = sitk.Resample(small_bias_field, X)
        X_f = sitk.Cast(X, sitk.sitkFloat32)
        X_t = sitk.Subtract(X_f, bias_field)
        X_t = sitk.Clamp(X_t, pixel_id, lowerBound=0.0)
        return X_t


class Resizer(BaseEstimator, TransformerMixin):
    # Image resizing class
    # Applies cropping and padding to resize an image to specific size.
    def __init__(self, targetsize, fillmode='constant'):
        """
        Parameters
        ----------
        targetsize : list of 3 integers [int, int, int]
            List of X, Y, Z integer values specifying the size of the
            output image.

        fill_mode : string, default='constant'
            Set fill pattern used when images are padded.
            Options: 'constant', 'edge'

        Returns
        ----------
        None
        """
        if not isinstance(targetsize, list):
            if not isinstance(targetsize, np.ndarray):
                raise ValueError('targetsize should be a numpy array or list')

        # easier if it is a list
        self.targetsize = list(targetsize)

        if len(targetsize) != 3:
            raise ValueError('targetsize should have 3 elements')

        self.targetsize = [int(a) for a in targetsize]

        if fillmode not in ['constant', 'edge']:
            # not sure if this should be valueError or keyError or NameError
            raise KeyError('%s unknown fill mode.' % fillmode)

        self.fillmode = fillmode

    def _crop_array_to_size(self, X):
        """
        Cropping utility used by transform.
        Returns
        ----------
        X : numpy array
            A cropped version of the input X.
        """
        base_size = X.shape

        ncrop = []
        for i in range(3):
            halfcrop = int((base_size[i] - self.targetsize[i])/2)
            if halfcrop <= 0:
                crop_factor = (0, 0)
            else:
                if base_size[i] % 2 == 0:  # even size, use crop_factor as is
                    crop_factor = (halfcrop, halfcrop)
                else:
                    # odd, add 1 to one size
                    crop_factor = (halfcrop, halfcrop + 1)
            ncrop.append(crop_factor)

        X_c = X[ncrop[0][0]: base_size[0] - ncrop[0][1],
                ncrop[1][0]: base_size[1] - ncrop[1][1],
                ncrop[2][0]: base_size[2] - ncrop[2][1]]

        return X_c

    def _pad_array_to_size(self, X):
        """
        Padding utility used by transform.
        Returns
        ----------
        X_p : numpy array
            A padded version of the input X.

        """
        base_size = X.shape
        npad = []

        for i in range(3):  # for 3 planes
            halfpad = int((self.targetsize[i] - base_size[i]) / 2)
            if halfpad <= 0:
                pad = (0, 0)
            else:
                if base_size[i] % 2 == 0:
                    pad = (halfpad, halfpad)
                else:
                    # add 1 to makeup for the difference
                    pad = (halfpad, halfpad + 1)
            npad.append(pad)

        if self.fillmode == 'constant':
            X_p = np.pad(X, pad_width=npad, mode='constant', constant_values=0)
        else:
            # fillmode must be 'edge'
            X_p = np.pad(X, pad_width=npad, mode='edge')

        return X_p

    def fit(self, X, y=None):
        # we need this step to fit the sklearn pipeline
        return self

    def transform(self, X):
        # input has to be called X to match sklearn estimator syntax
        # i have a check input is stkimg at the top of all transform methods.

        if not isinstance(X, sitk.Image):
            raise ValueError('Resizer input should be a sitk image.')

        X_np = sitk.GetArrayFromImage(X).T
        base_size = X_np.shape

        if base_size == self.targetsize:
            # do nothing
            return X

        # crop needs to be done before padding
        # everything is done in numpy space
        X_np = self._crop_array_to_size(X_np)
        X_np = self._pad_array_to_size(X_np)

        X_t = sitk.GetImageFromArray(X_np.T)
        X_t.SetSpacing(X.GetSpacing())
        X_t.SetDirection(X.GetDirection())

        return X_t


class IntensityNormalizer(BaseEstimator, TransformerMixin):
    # intensity normalization class for z-scoring

    def __init__(self, mask):

        if not isinstance(mask, sitk.Image):
            raise ValueError('mask should be a sitk image instance.')
        self.mask = mask

    def fit(self, X, y=None):
        # we need this step to fit the sklearn pipeline
        # dont do anything here, no fitting required
        return self

    def transform(self, X):
        # Normalize the voxel values of an image inside the mask
        # Non-zero voxels in the provided mask define the "population" of
        # intensities used to set the mean and standard deviation estimates.

        # Image and mask dimensions [z,y,x] must match.
        # input has to be called X to match sklearn estimator syntax
        if not isinstance(X, sitk.Image):
            raise ValueError('Normalizer input should be a sitk image.')

        if X.GetSize() != self.mask.GetSize():
            print("*** Normalization Failed: Image & Mask sizes are different.\
                  Returning 'None'")
            print("*** Image size: {}".format(X.GetSize()))
            print("*** Mask size : {}".format(self.mask.GetSize()))
            return None

        mask_np = sitk.GetArrayFromImage(self.mask)
        mask_locations = np.where(mask_np > 0)

        X_np = sitk.GetArrayFromImage(X)
        vol_samples = X_np[mask_locations]
        mean = np.mean(vol_samples)
        std = np.std(vol_samples)

        # Note: 'mean' is a float. This operation converts the voxel type of
        # X_np to float
        X_np = X_np - mean
        if std > 0:
            X_np = X_np / std

        X_t = sitk.GetImageFromArray(X_np)
        X_t.CopyInformation(X)

        return X_t


class ImageMasker(BaseEstimator, TransformerMixin):
    # image masking class

    def __init__(self, mask):

        if not isinstance(mask, sitk.Image):
            raise ValueError('mask should be a sitk image instance.')
        self.mask = mask

    def fit(self, X, y=None):
        # we need this step to fit the sklearn pipeline
        # dont do anything here, no fitting required
        return self

    def transform(self, X):
        # Returns a masked copy image with values from only non-zero
        # locations of the provided mask in initiation

        if not isinstance(X, sitk.Image):
            raise ValueError('Masker input should be a sitk image instance.')

        if X.GetSize() != self.mask.GetSize():
            print("*** Image masking Failed: Image and Mask sizes are\
                   different. Returning 'None'")
            print("*** Image size: {}".format(X.GetSize()))
            print("*** Mask size : {}".format(self.mask.GetSize()))
            return None

        X_np = sitk.GetArrayFromImage(X).T
        mask_np = sitk.GetArrayFromImage(self.mask).T
        X_np[mask_np == 0] = X_np.min()
        X_t = sitk.GetImageFromArray(X_np.T)
        X_t.CopyInformation(X)
        return X_t

