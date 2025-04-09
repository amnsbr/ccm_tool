import numpy as np
import nibabel
import nimare
import os
import sys
from scipy.spatial.distance import cdist
import copy
from tqdm import tqdm

if sys.version_info >= (3, 10):
    from importlib.resources import files
else:
    from importlib_resources import files


from ccm_tool import io
MASK_PATH = files("ccm_tool.data.maps").joinpath("Grey10.nii.gz").as_posix()
mask_img = nibabel.load(MASK_PATH)
mask_arr = np.isclose(mask_img.get_fdata(), 1)
n_voxels = mask_arr.sum()

def ccm(
    dset, contrast=None, dense_fc_path=None, 
    use_mmap=False, dense_fc_arr=None,
    distance_threshold=4, n_perm=1000,
    fixed_effects=False, weight_by_N=True, 
    z_from_p=True, seed=0, 
    out_path=None, save_null=True, override=False,
    verbose=False 
):
    """
    Convergent connectivity mapping of given coordinates based
    on HCP dense connectome and following an adaptation of the approach
    introduced by Cash et al. 2023 (https://doi.org/10.1038/s44220-023-00038-8).
    Here the combined RSFC is calculated first across the foci of each
    experiment separately and then combined across experiments (random-effects analysis)

    Parameters
    -------
    dset: (nimare.dataset.Dataset)
    contrast: (str | None)
    dense_fc_path: (str | None)
        Path to HCP dense connectome .bin file, which is recommended
        to be on a scratch (fast I/O) disk
        If None, the file is downloaded from the cloud and on the fly,
        which will be slower
    use_mmap: (bool)
        When dense connectome is available as local file setting this
        option will load the dense FC as a np.memmap
    dense_fc_arr: (np.ndarray)
        Dense FC loaded into memory. If this is provided `dense_fc_path`
        and `use_mmap` are ignored. It'll make the process faster
        but requires large amount of memory (80+ GB)
    distance_threshold: (float)
        in mm, if distance of a focus and its assigned grey matter mask voxel
        exceeds this threshold it will be excluded
    n_perm: (int)
    fixed_effects: (bool)
        use fixed-effect model in which convergent connectivity
        is calculated across individual foci rather than experiments.
        In random-effects model, convergent connectivity is averaged
        in two levels, first across foci of each experiment and then
        across experiments. But in fixed-effects model, it is averaged
        in a single step across all foci of all experiments.        
    weight_by_N: (bool)
        weight experiments by their sample sizes
    z_from_p: (bool)
        calculate z from non-parametric p-values
        rather than `(obs - mean(null)) / std(null)`
    seed: (int)
    out_path: (str | None)
        path to save the results
    save_null: (bool)
        saves null distribution to disk
    override: (bool)
        override files
    verbose: (bool)
    
    Returns
    -------
    zmap: (np.ndarray)
        z-score map. Shape: (n_voxels,)
    mean_mean_fcs: (np.ndarray)
        mean of observed convergent connectivity.
        Shape: (n_voxels,)
    mean_fc_null: (np.ndarray)
        null mean convergent connectivity maps.
        Shape: (n_perm, n_voxels)
    coordinates: (pd.DataFrame)
        filtered coordinates
    """
    # determine source of dense connectome
    if dense_fc_arr is not None:
        dense_fc_src = dense_fc_arr
        # TODO: in this case the loops summing up FCs can be vectorized
    else:
        if use_mmap and (dense_fc_path is not None):
            dense_fc_src = np.memmap(
                dense_fc_path, 
                dtype=io.DTYPE, 
                mode='r', 
                shape=(io.N_VOXELS, io.N_VOXELS)
            )
        else:
            dense_fc_src = dense_fc_path
    # make a copy of the dataset
    dset = dset.copy()
    # filter dataset to contrast
    if contrast:
        dset = dset.slice(
            dset.coordinates.loc[
                dset.coordinates['contrast_id']==contrast, 'id'
            ].unique()
        )
    # get sample sizes
    sample_sizes = dset.metadata.set_index('id')['sample_sizes'].apply(lambda c: c[0])
    # get closest in-mask voxel index to each coordinate
    coordinates = dset.coordinates.copy()
    ## first convert xyz to ijk coordinates
    dset_ijk = nimare.utils.mm2vox(coordinates[['x', 'y', 'z']].values, mask_img.affine)
    ## get ijk indices of in-mask voxels
    mask_ijk = np.vstack(np.where(mask_arr)).T
    ## find the closest in-mask voxel to each coordinate
    distances = cdist(dset_ijk, mask_ijk)
    vox_indices = np.argmin(distances, axis=1)
    shortest_dists = distances[np.arange(len(vox_indices)), vox_indices]
    coordinates['vox_idx'] = vox_indices
    coordinates['shortest_dist'] = shortest_dists
    exps_before = coordinates['id'].unique()
    n_coordinates_before = coordinates.shape[0]
    # filter out points exceeding the distance threshold
    coordinates = coordinates.loc[coordinates['shortest_dist'] <= distance_threshold]
    # exclude experiments with no coordinates left
    exps = coordinates['id'].unique()
    sample_sizes = sample_sizes.loc[exps]
    # print number of coordinates and experiments removed
    if verbose:
        print(f"Excluded {n_coordinates_before - coordinates.shape[0]} coordinates with > {distance_threshold}mm distance to the mask")
        print(f"{len(exps_before) - len(exps)} experiments excluded with no in-mask coordinates")
    # for fixed effects combine all foci into a single experiment
    if fixed_effects:
        coordinates['id'] = 'all'
        if weight_by_N:
            print("Fixed-effects model does not support weighting by sample sizes. Ignoring the option.")
            weight_by_N = False
    # step 1: calculate true (observed) convergent
    # connectivity map separately within each experiment
    # and take their (weighted) average
    n_points = {}
    mean_fcs = {}
    sum_mean_fcs = np.zeros(n_voxels, dtype=float)
    sum_mean_fcs_denom = 0
    for exp_id, exp_df in coordinates.groupby("id"):
        n_points[exp_id] = exp_df.shape[0]
        # calculate average dense RSFC of current
        # experiment foci
        sum_fc = np.zeros(n_voxels, dtype=float)
        for vox_idx in exp_df['vox_idx']:
            sum_fc += io.load_dense_fc(dense_fc_src, vox_idx)
        mean_fcs[exp_id] = sum_fc / n_points[exp_id]
        # add up this experiment's convergent connectivity
        if weight_by_N:
            sum_mean_fcs += sample_sizes.loc[exp_id] * mean_fcs[exp_id]
            sum_mean_fcs_denom += sample_sizes.loc[exp_id]
        else:
            sum_mean_fcs += mean_fcs[exp_id]
            sum_mean_fcs_denom += 1
    # calculate the (weighted) mean of convergent
    # connectivity maps across included experiments
    mean_mean_fcs = sum_mean_fcs / sum_mean_fcs_denom
    if out_path is not None:
        np.save(os.path.join(out_path, "true_fc.npy"), mean_mean_fcs)
    # step 2: similarly calculate null convergent connectivity
    # stratified by experiemtns
    np.random.seed(seed)
    mean_fc_null = np.full((n_perm, n_voxels), np.NaN)
    for perm_idx in tqdm(range(n_perm)):
        null_mean_fcs = {}
        null_sum_mean_fcs = np.zeros(n_voxels, dtype=float)
        # create null (simulated) set of coordinates with the same
        # number of grayordinates per experiment
        for exp_id in n_points.keys():
            rand_voxels = np.random.choice(
                n_voxels, size=n_points[exp_id]
            )
            sum_fc = np.zeros(n_voxels, dtype=float)
            for vox_idx in rand_voxels:
                sum_fc += io.load_dense_fc(dense_fc_src, vox_idx)
            null_mean_fcs[exp_id] = sum_fc / n_points[exp_id]
            if weight_by_N:
                null_sum_mean_fcs += sample_sizes.loc[exp_id] * null_mean_fcs[exp_id]
            else:
                null_sum_mean_fcs += null_mean_fcs[exp_id]
        # note that there is no need to recalculate denominator as it is
        # identical to observed
        mean_fc_null[perm_idx] = null_sum_mean_fcs / sum_mean_fcs_denom
    if save_null and (out_path is not None):
        np.save(os.path.join(out_path, "null_fcs.npy"), mean_fc_null)
    # step 3: calculate z-scores
    if z_from_p:
        zmap_name = "zmap"
        # calculate asymmetric one-tailed p-values
        p_right = ((mean_fc_null >= mean_mean_fcs).sum(axis=0) + 1) / (n_perm + 1)
        p_left = ((-mean_fc_null >= -mean_mean_fcs).sum(axis=0) + 1) / (n_perm + 1)
        # calculate two-tailed p-values
        p = np.minimum(p_right, p_left) * 2
        # convert z to p
        zmap = nimare.transforms.p_to_z(p, tail="two")
        # make z of voxels with more extreme values
        # towards left tail negative
        zmap[p_left < p_right] *= -1
    else:
        zmap_name = "zmap_from_std"
        diff_map = mean_mean_fcs - mean_fc_null.mean(axis=0)
        zmap = diff_map / mean_fc_null.std(axis=0)
    if out_path is not None:
        np.save(os.path.join(out_path, f"{zmap_name}.npy"), zmap)
    return zmap, mean_mean_fcs, mean_fc_null, coordinates