import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import scipy.ndimage
import statsmodels.stats.multitest
import matplotlib.pyplot as plt
import seaborn as sns
import nimare
import nilearn.mass_univariate
from tqdm import tqdm

try:
    import cupy as cp
    has_cupy = True
except ImportError:
    cp = None
    has_cupy = False

from ccm_tool import io, utils

def ccm(
    dset, contrast=None, dense_fc_path=None, 
    use_mmap=False, dense_fc_arr=None,
    distance_threshold=4, n_perm=1000,
    fixed_effects=False, weight_by_N=True, seed=0, 
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
    pmap: (np.ndarray)
        p-value map. Shape: (n_voxels,)
    mean_mean_fcs: (np.ndarray)
        mean of observed convergent connectivity.
        Shape: (n_voxels,)
    mean_fc_null: (np.ndarray)
        null mean convergent connectivity maps.
        Shape: (n_perm, n_voxels)
    coordinates: (pd.DataFrame)
        filtered coordinates
    """
    # +1 number of permutations to use for calculating
    # the null distribution of Z-scores when doing
    # cluster-based corrections
    # _n_perm will be used to calculate true z and p maps
    _n_perm = n_perm
    n_perm += 1
    # determine source of dense connectome
    if dense_fc_arr is not None:
        from_memory = True
        dense_fc_src = dense_fc_arr
        # TODO: in this case the loops summing up FCs can be vectorized
    else:
        from_memory = False
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
    dset_ijk = nimare.utils.mm2vox(coordinates[['x', 'y', 'z']].values, io.MASK_IMG.affine)
    ## get ijk indices of in-mask voxels
    mask_ijk = np.vstack(np.where(io.MASK_ARR)).T
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
    sum_mean_fcs = np.zeros(io.N_VOXELS, dtype=float)
    sum_mean_fcs_denom = 0
    for exp_id, exp_df in coordinates.groupby("id"):
        n_points[exp_id] = exp_df.shape[0]
        # calculate average dense RSFC of current
        # experiment foci
        sum_fc = np.zeros(io.N_VOXELS, dtype=float)
        if (from_memory or use_mmap):
            mean_fcs[exp_id] = dense_fc_src[exp_df['vox_idx'], :].mean(axis=0)
        else:
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
        # save as nii.gz
        io.MASKER.inverse_transform(mean_mean_fcs).to_filename(
            os.path.join(out_path, "true_fc.nii.gz")
        )
    # step 2: similarly calculate null convergent connectivity
    # stratified by experiemtns
    np.random.seed(seed)
    mean_fc_null = np.full((n_perm, io.N_VOXELS), np.nan)
    for perm_idx in tqdm(range(n_perm)):
        null_mean_fcs = {}
        null_sum_mean_fcs = np.zeros(io.N_VOXELS, dtype=float)
        # create null (simulated) set of coordinates with the same
        # number of grayordinates per experiment
        for exp_id in n_points.keys():
            rand_voxels = np.random.choice(
                io.N_VOXELS, size=n_points[exp_id]
            )
            sum_fc = np.zeros(io.N_VOXELS, dtype=float)
            if (from_memory or use_mmap):
                null_mean_fcs[exp_id] = dense_fc_src[rand_voxels, :].mean(axis=0)
            else:
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
    # step 3: calculate z-scores and p-values
    zmaps = {}
    pmaps = {}
    for zmap_name in ["zmap_from_p", "zmap_from_std"]:
        if zmap_name == "zmap_from_p":
            pmap_name = "pmap_exact"
            # calculate asymmetric one-tailed p-values
            p_upper = ((mean_fc_null[:_n_perm] >= mean_mean_fcs).sum(axis=0) + 1) / (_n_perm + 1)
            p_lower = ((-mean_fc_null[:_n_perm] >= -mean_mean_fcs).sum(axis=0) + 1) / (_n_perm + 1)
            # calculate two-tailed p-values
            # cap p-values at 1.0
            pmap = np.minimum(np.minimum(p_upper, p_lower) * 2, 1.0)
            # convert z to p
            zmap = nimare.transforms.p_to_z(pmap, tail="two")
            # specify negative z voxels (with more extreme values
            # towards left tail)
            z_neg = p_lower < p_upper
            zmap[z_neg] *= -1
            zmaps[zmap_name] = zmap
            pmaps[pmap_name] = pmap
        else:
            pmap_name = "pmap_from_std"
            diff_map = mean_mean_fcs - mean_fc_null[:_n_perm].mean(axis=0)
            zmap = diff_map / mean_fc_null[:_n_perm].std(axis=0)
            # convert z to p
            pmap = nimare.transforms.z_to_p(zmap, tail="two")
            zmaps[zmap_name] = zmap
            pmaps[pmap_name] = pmap
        # save zmap and pmap
        if out_path is not None:
            np.save(os.path.join(out_path, f"{zmap_name}.npy"), zmap)
            io.MASKER.inverse_transform(zmap).to_filename(
                os.path.join(out_path, f"{zmap_name}.nii.gz")
            )
            np.save(os.path.join(out_path, f"{pmap_name}.npy"), pmap)
            io.MASKER.inverse_transform(pmap).to_filename(
                os.path.join(out_path, f"{pmap_name}.nii.gz")
            )
    return zmaps, pmaps, mean_mean_fcs, mean_fc_null, coordinates

def corr_tfce(true_fc, null_fcs, gpu=False):
    """
    TFCE correction of convergent connectivity mapping

    Parameters
    ----------
    true_fc: (np.ndarray)
        true (observed) convergent FC in MNI masked voxels.
        Shape (n_voxels,)
    null_fcs: (np.ndarray)
        null FCs in MNI masked voxels.
        Shape (_n_perm, n_voxels)
    gpu: (bool)
        use GPU for the calculations

    Returns
    -------
    tfce_pvals: (np.ndarray)
        p-values of TFCE correction.
        Shape (x, y, z)
    true_z: (np.ndarray)
        true (observed) Z map.
        Shape (x, y, z)
    true_tfce: (np.ndarray)
        true (observed) TFCE map.
        Shape (x, y, z)
    null_max_tfce: (np.ndarray)
        null distribution of maximum absolute TFCE values.
        Shape (n_perm,)
    """
    print("Running TFCE correction...")
    # determine total number of permutations
    # calculated, and number of permutations used
    # for voxel-wise Z calculation
    _n_perm = null_fcs.shape[0]
    n_perm = _n_perm - 1
    # define connectivity for cluster detection
    conn = scipy.ndimage.generate_binary_structure(3, 1)
    if gpu:
        conn = cp.asarray(conn)
    # get mask index mapping for faster 1D->3D mapping
    mask_idx_mapping = utils.get_mask_idx_mapping(io.MASK_ARR)
    if gpu:
        mask_idx_mapping = cp.asarray(mask_idx_mapping)
    # calculate true (observed) Z map
    # excluding the last (extra) null FC (to have the same number of
    # permutations when calculating null Z maps)
    true_z = (true_fc - null_fcs[:n_perm].mean(axis=0)) / (null_fcs[:n_perm].std(axis=0)) 
    # calculate TFCE for the true Z map after mapping it to 4D
    if gpu:
        # TODO: avoid repeating
        # the code for CPU and GPU
        true_z_4d = cp.append(true_z, 0)[mask_idx_mapping][:, :, :, None]
        true_tfce = utils.calculate_tfce_gpu(true_z_4d, conn, two_sided_test=True).get().squeeze()
    else:
        true_z_4d = np.append(true_z, 0)[mask_idx_mapping][:, :, :, None]
        true_tfce = nilearn.mass_univariate._utils.calculate_tfce(
            true_z_4d, conn, two_sided_test=True
        ).squeeze()
    # calculate null Z and TFCEs to calculate 
    # the null distribution of max values
    # TODO: add options to keep null Zs and TFCEs
    null_max_tfce = []
    for perm_idx in tqdm(range(n_perm)):
        # create a mask of current nulls which
        # will be compared to the current permutation
        # null (as if it is the true FC)
        # it includes all nulls except the current one
        if gpu:
            mask = cp.ones(_n_perm, dtype=bool)
        else:
            mask = np.ones(_n_perm, dtype=bool)
        mask[perm_idx] = False
        # calculate null Z map
        null_z = (null_fcs[perm_idx] - null_fcs[mask].mean(axis=0)) / (null_fcs[mask].std(axis=0))
        # map null Z to 4D and calculate TFCE
        # then append the maximum absolute TFCE value
        # to the null distribution
        if gpu:
            null_z_4d = cp.append(null_z, 0)[mask_idx_mapping][:, :, :, None]
            null_tfce = utils.calculate_tfce_gpu(null_z_4d, conn, two_sided_test=True)
            null_max_tfce.append(cp.nanmax(cp.abs(null_tfce)).get())
        else:
            null_z_4d = np.append(null_z, 0)[mask_idx_mapping][:, :, :, None]
            null_tfce = nilearn.mass_univariate._utils.calculate_tfce(
                null_z_4d, conn, two_sided_test=True
            )
            null_max_tfce.append(np.nanmax(np.abs(null_tfce)))
    # convert null max TFCE to array
    null_max_tfce = np.array(null_max_tfce)
    # calculate p-TFCE (on flattened map of true TFCE, 
    # then reshape it to the original shape)
    tfce_pvals = (
        ((null_max_tfce >= np.abs(true_tfce).flatten()[:, None]).sum(axis=1) + 1)
        / (n_perm + 1)
    ).reshape(true_tfce.shape)
    return tfce_pvals, true_z_4d[..., 0], true_tfce, null_max_tfce

def ccm_yeo(true_fc, null_fcs, z_from_p=False, fdr=True, plot='bar', ax=None):
    """
    Convergent connectivity mapping resolved across seven
    resting state networks of Yeo et al. 2011
    based on true and null FCs calculated using `ccm`

    Parameters
    ----------
    true_fc: (np.ndarray)
        true (observed) convergent FC in MNI masked voxels.
        Shape (n_voxels,)
    null_fcs: (np.ndarray)
        null FCs in MNI masked voxels.
        Shape (n_perm, n_voxels)
    z_from_p: (bool)
        Calculate network-level Z-values from p-values.
        Otherwise calculates Z-values based on difference
        of means divided by standard deviation, and transforms
        them to p-values
    fdr: (bool)
        Apply FDR correction on p-values
    plot: {'all', 'bar', 'radar', 'violin', None}
        - 'all': all plots
        - 'bar': bar plot of network-level Z-values
        - 'radar': radar plot of network-level Z-values
        - 'violin': violin plot of network-level null distributions
        - None: do not plot
    ax: (None | AxesSubplot | list of AxesSubplot)

    Returns
    -------
    z_vals: (pd.Series)
    p_vals: (pd.Series)
    true_fc_yeo: (np.ndarray)
        true (observed) convergent FC averaged across Yeo networks.
        Shape (7,)
    null_fcs_yeo:
        null FCs averaged across Yeo networks.
        Shape (n_perm, 7)
    """
    if plot == 'all':
        plots = ['bar', 'radar', 'violin']
    elif plot is None:
        plots = []
    else:
        plots = [plot]
    if ax is not None:
        assert len(ax) == len(plots), "Number of axes must match number of plots"
    # load yeo map
    yeo_map = io.load_yeo_map()
    # average of true and null FCs across yeo networks
    true_fc_yeo = (
        pd.DataFrame(true_fc, index=yeo_map.categorical)
        .reset_index()
        .groupby("index")
        .mean()
    )
    null_fcs_yeo = (
        pd.DataFrame(null_fcs.T, index=yeo_map.categorical)
        .reset_index()
        .groupby("index")
        .mean()
    )
    # calculate z- and p-values per network
    if z_from_p:
        n_perm = null_fcs.shape[0]
        # calculate asymmetric one-tailed p-values
        p_upper = (
            (null_fcs_yeo.values >= true_fc_yeo.values).sum(axis=1) + 1
        ) / (n_perm + 1)
        p_lower = (
            (-null_fcs_yeo.values >= -true_fc_yeo.values).sum(axis=1) + 1
        ) / (n_perm + 1)
        # calculate two-tailed p-values
        # cap p-values at 1.0
        p_vals = np.minimum(np.minimum(p_upper, p_lower) * 2, 1.0)
        # convert p to z + assign z signs based on which one-tailed p is smaller
        z_vals = nimare.transforms.p_to_z(p_vals, tail="two")
        z_vals[p_lower < p_upper] *= -1
    else:
        z_vals = (true_fc_yeo.iloc[:,0] - null_fcs_yeo.mean(axis=1)) / null_fcs_yeo.std(axis=1)
        p_vals = nimare.transforms.z_to_p(z_vals, tail="two")
    # apply FDR correction
    if fdr:
        _, p_vals = statsmodels.stats.multitest.fdrcorrection(p_vals)
    # label networks
    p_vals = pd.Series(p_vals, index=yeo_map.names)
    z_vals = pd.Series(z_vals, index=yeo_map.names)
    # plotting
    for i_plot, plot in enumerate(plots):
        ax_i = ax[i_plot] if ax is not None else None
        vmin = np.min([np.min(z_vals), -np.max(z_vals)]) - 0.5
        vmax = -vmin
        if plot == 'violin':
            if ax_i is None:
                fig, ax_i = plt.subplots(figsize=(4, 3))
            sns.violinplot(
                data=null_fcs_yeo.unstack().reset_index(),
                x="index",
                y=0,
                split=True,
                gap=0,
                alpha=1.0,
                linewidth=0.5,
                inner="point",
                inner_kws=dict(s=0.5, alpha=0.5),
                ax=ax_i,
            )
            # Note: if colors are set within violinplot
            # the half-violins will face each other
            # this is a workaround: paint them after plotting
            for i_v, violin in enumerate(ax_i.collections[::2]):
                violin.set_facecolor(yeo_map.colors[i_v])
            plt.setp(ax_i.collections, zorder=0, label="")  # puts violins in the back
            # plot observed
            ax_i.scatter(
                x=np.arange(7) + 0.45, y=true_fc_yeo[0], marker=1, color="black", s=40
            )
            # aesthetics
            sns.despine(ax=ax_i, offset=10, trim=True)
            ax_i.set_ylabel("Mean RSFC", fontsize=12)
            ax_i.set_xlabel("")
            xticklabels = yeo_map.shortnames
            for i in range(7):
                if p_vals.values[i] < 0.05:
                    xticklabels[i] += "\n*"
            ax_i.set_xticklabels(xticklabels, fontsize=12)
        elif plot == 'radar':
            categories=yeo_map.names
            N = len(categories)
            # repeat last value to make it a closed loop
            values = z_vals.tolist()
            values += values[:1]
            # define angles of axes
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += angles[:1]
            # initialize the spider plot
            if ax_i is None:
                fig, ax_i = plt.subplots(1, figsize=(3, 3), subplot_kw=dict(polar=True))
            # draw one axe per variable + add labels
            ax_i.set_xticks(angles[:-1], [""]*N)
            for i in range(N):
                ha = 'left' if np.cos(angles[i]) >= 0 else 'right'
                ax_i.text(angles[i], vmax+0.5, categories[i], size=12, color='black',
                        horizontalalignment=ha, verticalalignment='center')
            # draw ylabels centered around 0
            # (but do not draw 0 itself)
            ax_i.set_rlabel_position(vmin)
            plt.ylim(vmin,vmax)
            yticks = np.unique(np.concatenate([np.linspace(vmin, 0, 3)[:-1], np.linspace(0, vmax, 3)[1:]]))
            yticklabels = np.array([f'{v:.1f}' for v in yticks])
            ax_i.set_yticks(yticks, yticklabels, color="grey", size=7)
            # 0-axis line
            ax_i.plot(np.linspace(0, 2 * np.pi, 100), [0]*100, color='black', linewidth=1, ls='--', label='Chance level')
            # plot data
            ax_i.plot(angles, values, 'o-', linewidth=1, markersize=3, color='saddlebrown', label='Z-value')
            # fill area
            ax_i.fill(angles, values, 'b', alpha=0.1, color='saddlebrown')
            # significance indicators
            for i in range(N):
                if p_vals.values[i] < 0.05:
                    ax_i.text(angles[i], values[i]+0.25, '*', size=12, color='black',
                            horizontalalignment='center', verticalalignment='center')
            ax_i.legend(loc='upper right', bbox_to_anchor=(2.15, 1.25))
        elif plot == 'bar':
            if ax_i is None:
                fig, ax_i = plt.subplots(figsize=(4, 3))
            sns.barplot(
                x=yeo_map.shortnames,
                y=z_vals, 
                hue=yeo_map.shortnames,
                palette=yeo_map.colors,
                legend=False,
                ax=ax_i
            )
            ax_i.set_ylim(vmin, vmax)
            ax_i.set_ylabel('Z-value')
            ax_i.axhline(0, color='black', ls='-')
            for i in range(7):
                if p_vals.values[i] < 0.05:
                    z_val = z_vals.values[i]
                    offset = 0.15 if z_val > 0 else -0.15
                    ax_i.text(i, z_vals.values[i]+offset, '*', size=12, color='black',
                            horizontalalignment='center', verticalalignment='center')
    return z_vals, p_vals, true_fc_yeo, null_fcs_yeo