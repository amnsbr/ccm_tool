import os
import numpy as np
import pandas as pd
import nimare
from tqdm import tqdm
import boto3
from botocore import UNSIGNED
from botocore.client import Config

# prepare the S3 client for accessing the HCP dense functional connectivity data
S3 = boto3.client(
    's3', 
    endpoint_url="https://s3.nexus.mpcdf.mpg.de",
    config=Config(signature_version=UNSIGNED)
)
BUCKET = "hcp-dense-fc"
FNAME = "subs-100_space-MNI152_den-2mm_mask-Gray10_smooth-4_desc-FC_mean_sym.bin"
# the datatype and number of voxels in the dense functional connectivity matrix
DTYPE=np.float16
N_VOXELS=211590


def jale_to_nimare(coordinates_path):
    """
    Converts a JALE-formatted coordinates excel file to a NiMARE dataset.
    The excel file must have a single sheet and contain the following columns
    (case-insensitive): 
    - "experiments": study name
    - "subjects": number of subjects (in the smallest group)
    - "x", "y", "z": peak coordinates
    - "space": coordinate space ("MNI" or "TAL")
    - (optional) "contrast": contrast name
    Note that other columns (e.g. tags) will be ignored.

    Parameters
    ----------
    coordinates_path: str
        Path to the JALE-formatted coordinates excel file

    Returns
    -------
    dset: nimare.dataset.Dataset
        NiMARE dataset object
    """
    # load the coordinates file
    coordinates = pd.read_excel(coordinates_path)
    # make all column names lower case
    coordinates.columns = coordinates.columns.str.lower()
    # TODO: add support for tags
    # drop empty rows
    coordinates = coordinates.dropna(subset=["experiments"])
    source = {}
    for study_id, study_df in coordinates.copy().groupby("experiments"):
        if study_id not in source:
            source[study_id] = {"contrasts":{}}
        if "contrast" in study_df.columns:
            exps = list(study_df.groupby("contrast"))
            # TODO: group by contrast and/or other tags
        else:
            exps = [('all', study_df)]
        for exp_id, exp_df in exps:
            # convert TAL to MNI
            # even though nimare automatically converts TAL to MNI, we do it here
            # to allow the individual coordinates of the contrast to be in different spaces
            # (as it may happen with coordinates reported in separate publications)
            if (exp_df["space"].str.lower().str.startswith('tal')).sum() > 0:
                exp_df.loc[exp_df["space"].str.lower().str.startswith('tal'), ["x", "y", "z"]] = (
                    nimare.utils.tal2mni(exp_df[["x", "y", "z"]].values)
                )
            # TODO: also handle unknown spaces
            source[study_id]["contrasts"][exp_id] = {
                "coords": {
                    "space": "MNI",
                    "x": exp_df['x'].tolist(),
                    "y": exp_df['y'].tolist(),
                    "z": exp_df['z'].tolist(),
                },
                "metadata": {
                    "sample_sizes": [int(exp_df['subjects'].values[0])],
                    # TODO: add additional tags as metadata
                }
            }
    # create the nimare dataset
    dset = nimare.dataset.Dataset(source)
    return dset

def load_dense_fc(path=None, row=None):
    """
    Loads HCP dense functional connectivity from local `path` or the cloud
    (if `path` is None) and returns the specified row or the entire matrix
    (if `row` is None). When loading from the cloud `row` cannot be None.

    Parameters
    ----------
    path: str or None
        Path to the local .bin file. If None, the file is downloaded from the cloud
    row: int or None
        Row to return. If None, the entire matrix is returned

    Returns
    -------
    fc: numpy.ndarray
        Dense functional connectivity row/matrix
    """
    if row is not None:
        offset = row * np.dtype(DTYPE).itemsize * N_VOXELS
    if path is None:
        if row is None:
            raise ValueError("Row must be specified when loading from the cloud")
        # specify byte range
        byte_size = np.dtype(DTYPE).itemsize * N_VOXELS
        byte_range = f"bytes={offset}-{offset + byte_size - 1}"
        # fetch object part
        response = S3.get_object(Bucket=BUCKET, Key=FNAME, Range=byte_range)
        # read binary data into an array
        return np.frombuffer(response["Body"].read(), dtype=DTYPE)
    else:
        if row is not None:
            return np.fromfile(path, dtype=DTYPE, offset=offset, count=N_VOXELS)
        else:
            return np.fromfile(path, dtype=DTYPE).reshape(N_VOXELS, N_VOXELS)

def download_dense_fc(out_dir):
    """
    Downloads the HCP dense functional connectivity matrix to local storage
    in the specified path. Note that this file is large (~84 GB).

    Parameters
    ----------
    out_dir: str
        Path to save the dense functional connectivity matrix
    """
    print("Warning: this file is large (~89 GB). Make sure you have enough space.")
    continue_input = input("Continue? [Y/n]")
    if continue_input.lower() == "n":
        print("Aborted.")
        return
    # create the output directory
    os.makedirs(out_dir, exist_ok=True)
    local_path = os.path.join(out_dir, FNAME)
    # determine total file size
    response = S3.head_object(Bucket=BUCKET, Key=FNAME)
    total_size = response["ContentLength"]
    # stream the download
    with S3.get_object(Bucket=BUCKET, Key=FNAME)["Body"] as body, open(local_path, "wb") as f, tqdm(
        total=total_size, unit="B", unit_scale=True, desc="Downloading"
    ) as pbar:
        # read in 1 MB chunks
        for chunk in iter(lambda: body.read(1024**2), b""):
            f.write(chunk)
            pbar.update(len(chunk))
    print(f"Download completed: {local_path}")