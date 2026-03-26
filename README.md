# ccm_tool

> **Note:** This project is under active development. Interfaces and behavior may change at any point without prior notice or deprecation warnings.

`ccm_tool` is a Python package for **Convergent Connectivity Mapping (CCM)**, a coordinate-based approach that identifies brain regions whose resting-state functional connectivity (RSFC) converges across a set of reported peak coordinates from neuroimaging studies.

The method is an adaptation of the approach introduced by [Cash et al. 2023](https://doi.org/10.1038/s44220-023-00038-8) and modified by [Saberi et al. 2025](https://doi.org/10.1038/s41380-024-02780-6) and uses a pre-calculated dense RSFC matrix based on the Human Connectome Project (HCP) as its connectivity reference.

## Overview

Given a set of peak coordinates (e.g. from a CBMA), `ccm_tool`:

1. Reads an Excel file of peak coordinates ([JALE](https://github.com/LenFrahm/JALE)/[pyALE](https://github.com/LenFrahm/pyALE) format) and converts it to a [NiMARE](https://nimare.readthedocs.io) `Dataset`. It can also directly work with a NiMARE `Dataset` object.
2. Maps each peak coordinate to the nearest voxel within a grey matter mask (MNI152, 2 mm resolution, Gray10 threshold).
3. Extracts the RSFC profile (row) of each seed voxel from the HCP dense connectome.
4. Combines RSFC profiles across foci within each experiment, then across experiments using a random-effects model (default) or a fixed-effects model, with optional sample-size weighting.
5. Derives Z-maps and p-maps using permutation testing, with optional TFCE or cluster FWE corrections.
6. Optionally summarizes results at the level of the canonical resting state networks and produces bar, radar, or violin plots.

## Installation

```bash
pip install git+https://github.com/amnsbr/ccm_tool
```

**Optional GPU acceleration:** For GPU-based TFCE and cluster corrections, install [CuPy](https://cupy.dev) and `nimare_gpu`:

```bash
pip install cupy-cuda12x  # adjust to your CUDA version
pip install nimare_gpu
```

## Data

`ccm_tool` requires the HCP dense RSFC matrix, which is a symmetric 211,590 × 211,590 matrix of mean pairwise RSFC values stored as a giant binary file (~89 GB).

The file is publicly accessible on an S3-compatible object store.

### Recommended: download to disk

By default, `ccm()` fetches individual rows from S3 on the fly. This is convenient but slow, as each seed voxel requires a separate network request. It is strongly recommended to download the file to a local disk first:

```python
import ccm_tool.io as ccm_io

ccm_io.download_dense_fc("/path/to/data/dir")
```

Then pass the local path when running an analysis:

```python
results = ccm_tool.ccm.ccm(dset, dense_fc_path="/path/to/data/dir/<filename>.bin")
```

### Performance tips

- **Fast I/O storage matters.** Storing the file on a scratch partition, NVMe SSD, or any storage with high sequential read throughput substantially reduces analysis time.
- **Load into memory for maximum speed.** If you have sufficient RAM (~80+ GB), loading the full matrix as a NumPy array and passing it via `dense_fc_arr` eliminates all disk I/O during analysis:

  ```python
  import numpy as np
  dense_fc = np.fromfile("/path/to/data/<filename>.bin", dtype=np.float16).reshape(211590, 211590)
  results = ccm_tool.ccm.ccm(dset, dense_fc_arr=dense_fc)
  ```

## Code Structure

```
ccm_tool/
├── ccm.py           # CCM algorithms
├── io.py            # data I/O
└── utils.py         # utility functions
```

## Basic Usage

```python
import ccm_tool

# 1. Load peak coordinates from a JALE/pyALE-formatted Excel file
dset = ccm_tool.io.xlsx_to_nimare("coordinates.xlsx")

# 2. Run CCM (using a locally downloaded dense FC file)
results = ccm_tool.ccm.ccm(
    dset,
    dense_fc_path="/path/to/data/subs-100_space-MNI152_den-2mm_mask-Gray10_smooth-6_lp-0_08_hp-0_01_detrend_desc-FC_mean_sym.bin",
    n_perm=1000,
    out_path="./results"
)

# results is a dict containing Z-maps, p-maps, and null distributions
# Outputs are also saved as .nii.gz files under out_path
```

The input Excel file must contain the columns: `study` (or `experiment`/`article`), `subjects` (or `N`), `x`, `y`, `z`, and `space` (`MNI` or `TAL`). An optional `contrast` column allows grouping foci by contrast within a study.

## Citation

If you use `ccm_tool`, please cite the original CCM method papers:

> Cash et al. "Altered brain activity in unipolar depression unveiled using connectomics". Nature Mental Health (2023). https://doi.org/10.1038/s44220-023-00038-8

> Saberi et al. "Convergent functional effects of antidepressants in major depressive disorder: a neuroimaging meta-analysis". Molecular Psychiatry (2025). https://doi.org/10.1038/s41380-024-02780-6