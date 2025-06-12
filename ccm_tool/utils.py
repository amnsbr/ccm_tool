import numpy as np

try:
    import cupy as cp
    has_cupy = True
except ImportError:
    cp = None
    has_cupy = False
else:
    import cupyx.scipy.ndimage

def calculate_tfce_gpu(
    arr4d,
    bin_struct,
    E=0.5,
    H=2,
    dh="auto",
    two_sided_test=True,
):
    # this function is adapted from
    # nilearn.mass_univariate._utils.calculate_tfce
    # at version 0.11.1
    """
    Calculate threshold-free cluster enhancement values for scores maps
    on GPU.

    The :term:`TFCE` calculation is mostly implemented as described in [1]_,
    with minor modifications to produce similar results to fslmaths, as well
    as to support two-sided testing.

    Parameters
    ----------
    arr4d : :obj:`numpy.ndarray` or :obj:`cupy.ndarray` of shape (X, Y, Z, R)
        Unthresholded 4D array of 3D t-statistic maps.
        R = regressor.
    bin_struct : :obj:`numpy.ndarray` or :obj:`cupy.ndarray` of shape (3, 3, 3)
        Connectivity matrix for defining clusters.
    E : :obj:`float`, default=0.5
        Extent weight.
    H : :obj:`float`, default=2
        Height weight.
    dh : 'auto' or :obj:`float`, default='auto'
        Step size for TFCE calculation.
        If set to 'auto', use 100 steps, as is done in fslmaths.
        A good alternative is 0.1 for z and t maps, as in [1]_.
    two_sided_test : :obj:`bool`, default=False
        Whether to assess both positive and negative clusters (True) or just
        positive ones (False).

    Returns
    -------
    tfce_arr : :obj:`cupy.ndarray`, shape=(n_descriptors, n_regressors)
        :term:`TFCE` values.

    Notes
    -----
    In [1]_, each threshold's partial TFCE score is multiplied by dh,
    which makes directly comparing TFCE values across different thresholds
    possible.
    However, in fslmaths, this is not done.
    In the interest of maximizing similarity between nilearn and established
    tools, we chose to follow fslmaths' approach.

    Additionally, we have modified the method to support two-sided testing.
    In fslmaths, only positive clusters are considered.

    References
    ----------
    .. [1] Smith, S. M., & Nichols, T. E. (2009).
       Threshold-free cluster enhancement: addressing problems of smoothing,
       threshold dependence and localisation in cluster inference.
       Neuroimage, 44(1), 83-98.
    """
    if not has_cupy:
        raise ImportError(
            "cupy is required for this function. "
            "Please install cupy to use it."
        )
    # convert input arrays to 
    if isinstance(arr4d, np.ndarray):
        arr4d = cp.asarray(arr4d)
    if isinstance(bin_struct, np.ndarray):
        bin_struct = cp.asarray(bin_struct)

    # initialize TFCE maps
    tfce_4d = cp.zeros_like(arr4d)

    # For each passed t map
    for i_regressor in range(arr4d.shape[3]):
        arr3d = arr4d[..., i_regressor]
        # Get signs / threshs
        if two_sided_test:
            signs = [-1, 1]
            max_score = cp.max(cp.abs(arr3d))
        else:
            signs = [1]
            max_score = cp.max(arr3d)
    
        step = max_score / 100 if dh == "auto" else dh
    
        # Set based on determined step size
        score_threshs = cp.arange(step.get(), max_score.get() + step.get(), step.get())
    
        # If we apply the sign first...
        for sign in signs:
            # Init a temp copy of arr3d with the current sign applied,
            # which can then be reused by incrementally setting more
            # voxel's to background, by taking advantage that each score_thresh
            # is incrementally larger
            temp_arr3d = arr3d * sign
    
            # Prep step
            for score_thresh in score_threshs:
                temp_arr3d[temp_arr3d < score_thresh] = 0
    
                # Label into clusters - importantly (for the next step)
                # this returns clusters labeled ordinally
                # from 1 to n_clusters+1,
                # which allows us to use bincount to count
                # frequencies directly.
                labeled_arr3d, _ = cupyx.scipy.ndimage.label(temp_arr3d, bin_struct)
    
                # Next, we want to replace each label with its cluster
                # extent, that is, the size of the cluster it is part of
                # To do this, we will first compute a flattened version of
                # only the non-zero cluster labels.
                labeled_arr3d_flat = labeled_arr3d.flatten()
                non_zero_inds = cp.where(labeled_arr3d_flat != 0)[0]
                labeled_non_zero = labeled_arr3d_flat[non_zero_inds]
    
                # Count the size of each unique cluster, via its label.
                # The reason why we pass only the non-zero labels to bincount
                # is because it includes a bin for zeros, and in our labels
                # zero represents the background,
                # which we want to have a TFCE value of 0.
                # Only doing the following steps if there are any clusters
                # NOTE: this differs from nilearn's implementation because with
                # cupy bincount raises an error if its input is empty, but
                # numpy does not raise an error
                if labeled_non_zero.size > 0:
                    cluster_counts = cp.bincount(labeled_non_zero)
                    
                    # Next, we convert each unique cluster count to its TFCE value.
                    # Where each cluster's tfce value is based
                    # on both its cluster extent and z-value
                    # (via the current score_thresh)
                    # NOTE: We do not multiply by dh, based on fslmaths'
                    # implementation. This differs from the original paper.
                    cluster_tfces = sign * (cluster_counts**E) * (score_thresh**H)
        
        
                    # Before we can add these values to tfce_4d, we need to
                    # map cluster-wise tfce values back to a voxel-wise array,
                    # including any zero / background voxels.
                    tfce_step_values = cp.zeros(labeled_arr3d_flat.shape)
                    tfce_step_values[non_zero_inds] = cluster_tfces[
                        labeled_non_zero
                    ]
                    
                    # Now, we just need to reshape these values back to 3D
                    # and they can be incremented to tfce_4d.
                    tfce_4d[..., i_regressor] += tfce_step_values.reshape(
                        temp_arr3d.shape
                    )
    return tfce_4d

def get_mask_idx_mapping(mask_arr):
    """
    Get the mapping from ijk coordinates to voxel
    index in mask (or -1 if voxel is outside mask).

    Parameters
    ----------
    mask_arr : :obj:`np.ndarray`
        3D numpy array representing the mask, where
        voxels inside the mask are 1 and outside are 0.

    Returns
    -------
    :obj:`np.ndarray`
        A 3D numpy array of the same shape as `mask_arr`,
        where each voxel contains its index in the mask
        or -1 if the voxel is outside the mask.
    """
    # get ijk of mask voxels
    in_mask_voxels = np.array(np.where(mask_arr)).T
    n_voxels_in_mask = in_mask_voxels.shape[0]
    if n_voxels_in_mask == 0:
        raise ValueError("No voxels found in mask. The voxel values must be 0 and 1.")
    # create a mapping from ijk coordinates to voxel index in mask
    # or -1 if voxel is outside mask
    # by setting the non-mask voxels to -1 ,
    # in permutation code, in 1D->3D mapping, it points to the last element
    # of the 1D array, which is a 0 appended to the end of 1D array on the fly, 
    # and so it makes the background all NaN
    # this is a faster way of 1D->3D mapping
    mask_idx_mapping = np.ones(mask_arr.shape, dtype=int)*-1
    for i in range(in_mask_voxels.shape[0]):
        mask_idx_mapping[in_mask_voxels[i,0], in_mask_voxels[i,1], in_mask_voxels[i,2]] = i
    return mask_idx_mapping