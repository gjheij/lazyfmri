import math
import csv
import numpy as np
import os
import operator
import pandas as pd
from nilearn import signal
from shapely import geometry
import matplotlib.colors as mcolors
import nibabel as nb
from PIL import ImageColor

opj = os.path.join

def disassemble_fmriprep_wf(wf_path, subj_ID, prefix="sub-"):
    """disassemble_fmriprep_wf

    Parses the workflow-folder from fMRIPrep into its constituents to recreate a filename. Searches for the following keys: `['ses', 'task', 'acq', 'run']`.

    Parameters
    ----------
    wf_path: str
        Path to workflow-folder
    subj_ID: str
        Subject ID to append to `prefix`
    prefix: str, optional
        Forms together with `subj_ID` the beginning of the new filename. By default "sub-"

    Returns
    ----------
    str
        filename based on constituent file parts

    Example
    ----------
    >>> from linescanning.utils import disassemble_fmriprep_wf
    >>> wf_dir = "func_preproc_ses_2_task_pRF_run_1_acq_3DEPI_wf"
    >>> fname = disassemble_fmriprep_wf(wf_dir, "001")
    >>> fname
    'sub-001_ses-2_task-pRF_acq-3DEPI_run-1'
    """
    wf_name = [ii for ii in wf_path.split(os.sep) if "func_preproc" in ii][0]
    wf_elem = wf_name.split("_")
    fname = [f"{prefix}{subj_ID}"]

    for tag in ['ses', 'task', 'acq', 'run']:

        if tag in wf_elem:
            idx = wf_elem.index(tag)+1
            fname.append(f"{tag}-{wf_elem[idx]}")

    fname = "_".join(fname)
    return fname

def assemble_fmriprep_wf(bold_path, wf_only=False):
    """assemble_fmriprep_wf

    Parses the bold file into a workflow name for fMRIPrep into its constituents to recreate a filename. Searches for the following keys: `['ses', 'task', 'acq', 'run']`.

    Parameters
    ----------
    bold_path: str
        Path to bold-file
    wf_only: bool, optional
        If `sub` tag is found in `bold_path`, we can reconstruct the full workflow folder including preceding `single_subject_<sub_id>_wf`. If you do not want this, set `wf_only` to **False**.

    Returns
    ----------
    str
        filename based on constituent file parts

    Example
    ----------
    >>> from linescanning.utils import disassemble_fmriprep_wf
    >>> bold_file = "sub-008_ses-2_task-SRFi_acq-3DEPI_run-1_desc-preproc_bold.nii.gz"
    >>> wf_name = assemble_fmriprep_wf(bold_file)
    >>> wf_name
    >>> 'single_subject_008_wf/func_preproc_ses_2_task_SRFi_run_1_acq_3DEPI_wf'

    >>> # workflow name only
    >>> wf_name = assemble_fmriprep_wf(bold_file, wf_only=True)
    >>> wf_name
    >>> 'func_preproc_ses_2_task_SRFi_run_1_acq_3DEPI_wf'
    """
    bids_comps = split_bids_components(os.path.basename(bold_path))
    fname = ["func_preproc"]

    for tag in ['ses', 'task', 'run', 'acq']:
        if tag in list(bids_comps.keys()):
            fname.append(f"{tag}_{bids_comps[tag]}")
    
    base_dir = ""
    fname = "_".join(fname)+"_wf"
    if 'sub' in list(bids_comps.keys()):
        base_dir = f"single_subject_{bids_comps['sub']}_wf"

        if wf_only:
            return fname
        else:
            return opj(base_dir, fname)
    else:
        return fname

class BIDSFile():

    def __init__(self, bids_file):
        self.bids_file = os.path.abspath(bids_file)

    def get_bids_basepath(self, *args):
        return self._get_bids_basepath(self.bids_file, *args)
    
    def get_bids_root(self, *args):
        return self._get_bids_root(self.bids_file, *args)

    def get_bids_workbase(self, *args):
        return self._get_bids_workbase(self.bids_file, *args) 

    def get_bids_workflow(self, **kwargs):
        return assemble_fmriprep_wf(self.bids_file, **kwargs)   

    # def get_bids_root(self):
    @staticmethod
    def _get_bids_basepath(file, pref="sub"):
        sp = file.split(os.sep)
        for i in sp:
            if i.startswith(pref) and not i.endswith('.nii.gz'):
                base_path = os.sep.join(sp[sp.index(i)+1:-1])
                break

        return base_path
    
    # def get_bids_root(self):
    @staticmethod
    def _get_bids_workbase(file, pref="sub"):
        sp = file.split(os.sep)
        for i in sp:
            if i.startswith(pref) and not i.endswith('.nii.gz'):
                base_path = os.sep.join(sp[sp.index(i):-2])
                break

        return base_path    
    
    @staticmethod
    def _get_bids_root(file, pref="sub"):
        sp = file.split(os.sep)
        for i in sp:
            if i.startswith(pref) and not i.endswith('.nii.gz'):
                bids_root = os.sep.join(sp[:sp.index(i)])
                break

        return bids_root
    
    def get_bids_ids(self, **kwargs):
        return split_bids_components(self.bids_file, **kwargs)

class color:
    # """color
    
    # Add some color to the terminal.

    # Example
    # ----------
    # >>> print("set orientation to " + utils.color.BOLD + utils.color.RED + "SOME TEXT THAT'LL BE IN TED" + utils.color.END)
    # """
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

def convert2unit(v, method="np"):
    """convert vector to unit vector"""
    import numpy as np

    if method.lower() == "np":
        v_hat = v / np.linalg.norm(v)
        return v_hat
    elif method.lower() == "mesh":
        # https://sites.google.com/site/dlampetest/python/calculating-normals-of-a-triangle-mesh-using-numpy
        lens = np.sqrt( v[:,0]**2 + v[:,1]**2 + v[:,2]**2 )
        v[:,0] /= lens
        v[:,1] /= lens
        v[:,2] /= lens
        return v

def string2list(string_array, make_float=False):
    """string2list

    This function converts a array in string representation to a list of string. This can happen, for instance, when you use bash to give a list of strings to python, where ast.literal_eval fails.

    Parameters
    ----------
    string_array: str
        string to be converted to a valid numpy array with float values

    Returns
    ----------
    numpy.ndarray
        array containing elements in float rather than in string representation

    Example
    ----------
    >>> string2list('[tc,bgfs]')
    ['tc', 'bgfs']
    """

    if type(string_array) == str:
        new = string_array.split(',')[0:]
        new = list(filter(None, new))

        if make_float:
            new = [float(ii) for ii in new]
            
        return new

    else:
        # array is already in non-string format
        return string_array

def string2float(string_array):
    """string2float

    This function converts a array in string representation to a regular float array. This can happen, for instance, when
    you've stored a numpy array in a pandas dataframe (such is the case with the 'normal' vector). It starts by splitting
    based on empty spaces, filter these, and convert any remaining elements to floats and returns these in an array.

    Parameters
    ----------
    string_array: str
        string to be converted to a valid numpy array with float values

    Returns
    ----------
    numpy.ndarray
        array containing elements in float rather than in string representation

    Example
    ----------
    >>> string2float('[ -7.42 -92.97 -15.28]')
    array([ -7.42, -92.97, -15.28])
    """

    if type(string_array) == str:
        new = string_array[1:-1].split(' ')[0:]
        new = list(filter(None, new))
        new = [float(i.strip(",")) for i in new]
        new = np.array(new)

        return new

    else:
        # array is already in non-string format
        return string_array

def reverse_sign(x):
    """reverse_sign

    Inverts the sign given set of values. Can be either one value or an array of values that need to be inverted

    Parameters
    ----------
    x: int,float,list,numpy.ndarray
        input that needs inverting, either one value or a list
    
    Returns
    ----------
    the inverse of whatever the input `x` was

    Example
    ----------
    >>> # input is integer
    >>> x = 5
    >>> reverse_sign(x)
    -5
    >>> # input is array
    >>> x = np.array([2, -2340, 2345,123342, 123])
    >>> In [6]: reverse_sign(x)
    array([-2.00000e+00,  2.34000e+03, -2.34500e+03, -1.23342e+05,-1.23000e+02])
    >>> # input is float
    >>> x = 5.0
    >>> reverse_sign(x)
    -5.0
    """

    import numpy as np

    inverted = ()

    if isinstance(x, int) or isinstance(x, float) or isinstance(x, np.float32):
        if x > 0:
            inverted = -x
        else:
            inverted = abs(x)
    elif isinstance(x, np.ndarray):
        for i in x:
            if float(i) > 0:
                val = -float(i)
            else:
                val = abs(float(i))

            inverted = np.append(inverted, val)

    return inverted

def convert_to_rgb(color, as_integer=False):
    if isinstance(color, tuple):
        (R, G, B) = color
    elif isinstance(color, str):
        if len(color) == 1:
            color = mcolors.to_rgb(color)
        else:
            color = ImageColor.getcolor(color, "RGB")

        (R, G, B) = color

    if not as_integer:
        rgb = []
        for v in [R, G, B]:
            if v > 1:
                v /= 255
            rgb.append(v)
        R, G, B = rgb
    else:
        rgb = []
        for v in [R, G, B]:
            if v <= 1:
                v = int(v*255)
            rgb.append(v)
        R, G, B = rgb

    return (R, G, B)

def decode(obj):
    """decode an object"""
    if isinstance(obj, bytes):
        obj = obj.decode()
    return obj

def copy_hdr(source_img,dest_img):
    """copy_hdr

    Similar functionality as fslcpgeom but than more rigorious using Nibabel. Copies the ENTIRE header, including affine, quaternion rotations, and dimensions.

    Parameters
    ----------
    source_img: str, nibabel.Nifti1Image
        source image from which to derive the header information
    dest_img: str, nibabel.Nifti1Image
        destination image to which to copy the header from <source image> to

    Returns
    ----------
    nibabel.Nifti1Image
        `source_img` with updated header information

    Example
    ----------
    >>> new_img = copy_hdr(img1,img2)
    """

    if isinstance(source_img, nb.Nifti1Image):
        src_img = source_img
    elif isinstance(source_img, str):
        src_img = nb.load(source_img)

    if isinstance(dest_img, nb.Nifti1Image):
        targ_img = dest_img
    elif isinstance(dest_img, str):
        targ_img = nb.load(dest_img)

    new = nb.Nifti1Image(targ_img.get_fdata(), affine=src_img.affine, header=src_img.header)
    return new

def ants_to_spm_moco(affine, deg=False, convention="SPM"):

    """SPM output = x [LR], y [AP], z [SI], rx, ry, rz. ANTs employs an LPS system, so y value should be switched"""
    dx, dy, dz = affine[9:]

    if convention == "SPM":
        dy = reverse_sign(dy)

    rot_x = np.arcsin(affine[6])
    cos_rot_x = np.cos(rot_x)
    rot_y = np.arctan2(affine[7] / cos_rot_x, affine[8] / cos_rot_x)
    rot_z = np.arctan2(affine[3] / cos_rot_x, affine[0] / cos_rot_x)

    if deg:
        rx,ry,rz = np.degrees(rot_x),np.degrees(rot_y),np.degrees(rot_z)
    else:
        rx,ry,rz = rot_x,rot_y,rot_z

    moco_pars = np.array([dx,dy,dz,rx,ry,rz])
    return moco_pars

def make_chicken_csv(
    coord,
    input="ras",
    output_file=None,
    vol=0.343
    ):

    """make_chicken_csv

    This function creates a .csv-file like the chicken.csv example from ANTs to warp a coordinate using a transformation file
    ANTs assumes the input coordinate to be LPS, but this function can deal with RAS-coordinates too.
    (see https://github.com/stnava/chicken for the reason of this function's name)

    Parameters
    ----------
    coord: np.ndarray
        numpy array containing the three coordinates in x,y,z direction
    input: str
        specify whether your coordinates uses RAS or LPS convention (default is RAS, and will be converted to LPS to create
        the file)
    output_file: str
        path-like string pointing to an output file (.csv!)
    vol: float
        volume of voxels (pixdim_x*pixdim_y*pixdim_z). If you're using the standard 0.7 MP2RAGE, the default vol will be ok

    Returns
    ----------
    str
        path pointing to the `csv`-file containing the coordinate

    Example
    ----------
    >>> make_chicken_csv(np.array([-16.239,-67.23,-2.81]), output_file="sub-001_space-fs_desc-lpi.csv")
    "sub-001_space-fs_desc-lpi.csv"
    """

    if len(coord) > 3:
        coord = coord[:3]

    if input.lower() == "ras":
        # ras2lps
        LPS = np.array([[-1,0,0],
                        [0,-1,0],
                        [0,0,1]])

        coord = LPS @ coord

    # rows = ["x,y,z,t,label,mass,volume,count", f"{coord[0]},{coord[1]},{coord[2]},0,1,1,{vol},1"]
    with open(output_file, "w") as target:
        writer = csv.writer(target, delimiter=",")
        writer.writerow(["x","y","z","t","label","mass","volume","count"])
        writer.writerow([coord[0],coord[1],coord[2],0,1,1,vol,1])

    return output_file

def read_chicken_csv(chicken_file, return_type="lps"):
    """read_chicken_csv

    Function to get at least the coordinates from a csv file used with antsApplyTransformsToPoints.
    (see https://github.com/stnava/chicken for the reason of this function's name)

    Parameters
    ----------
    chicken_file: str
        path-like string pointing to an input file (.csv!)
    return_type: str
        specify the coordinate system that the output should be in

    Returns
    ----------
    numpy.ndarray
        (3,) array containing the coordinate in `chicken_file`

    Example
    ----------
    >>> read_chicken_csv("sub-001_space-fs_desc-lpi.csv")
    array([-16.239,-67.23,-2.81])
    """
    
    contents = pd.read_csv(chicken_file)
    coord = np.squeeze(contents.iloc[:,0:3].values)

    if return_type.lower() == "lps":
        return coord
    elif return_type.lower() == "ras":
        # ras2lps
        LPS = np.array([[-1,0,0],
                        [0,-1,0],
                        [0,0,1]])

        return LPS@coord


def make_binary_cm(color):
    """make_binary_cm

    This function creates a custom binary colormap using matplotlib based on the RGB code specified. Especially useful if you
    want to overlay in imshow, for instance. These RGB values will be converted to range between 0-1 so make sure you're
    specifying the actual RGB-values. I like `https://htmlcolorcodes.com` to look up RGB-values of my desired color. The
    snippet of code used here comes from https://kbkb-wx-python.blogspot.com/2015/12/python-transparent-colormap.html

    Parameters
    ----------
    <color>: tuple, str
        either  hex-code with (!!) '#' or a tuple consisting of:

        * <R>     int | red-channel (0-255)
        * <G>     int | green-channel (0-255)
        * <B>     int | blue-channel (0-255)

    Returns
    ----------
    matplotlib.colors.LinearSegmentedColormap object
        colormap to be used with `plt.imshow`

    Example
    ----------
    >>> cm = make_binary_cm((232,255,0))
    >>> cm
    <matplotlib.colors.LinearSegmentedColormap at 0x7f35f7154a30>
    >>> cm = make_binary_cm("#D01B47")
    >>> cm
    >>> <matplotlib.colors.LinearSegmentedColormap at 0x7f35f7154a30>
    """

    # convert input to RGB
    R, G, B = convert_to_rgb(color)

    colors = [(R, G, B, c) for c in np.linspace(0, 1, 100)]
    cmap = mcolors.LinearSegmentedColormap.from_list('mycmap', colors, N=5)

    return cmap


def find_missing(lst):
    return [i for x, y in zip(lst, lst[1:])
            for i in range(x + 1, y) if y - x > 1]


def make_between_cm(
        col1,
        col2,
        as_list=False,
        **kwargs):

    input_list = [col1, col2]

    # scale to 0-1
    col_list = []
    for color in input_list:

        scaled_color = convert_to_rgb(color)
        col_list.append(scaled_color)

    cm = mcolors.LinearSegmentedColormap.from_list("", col_list, **kwargs)

    if as_list:
        return [mcolors.rgb2hex(cm(i)) for i in range(cm.N)]
    else:
        return cm


def make_stats_cm(
    direction,
    lower_neg=(51, 0, 248),
    upper_neg=(151, 253, 253),
    lower_pos=(217, 36, 36),
    upper_pos=(251, 255, 72),
    invert=False,
):

    if direction not in ["pos", "neg"]:
        raise ValueError(
            f"direction must be one of 'pos' or 'neg', not '{direction}'")

    if direction == "pos":
        input_list = [lower_pos, upper_pos]
    else:
        input_list = [lower_neg, upper_neg]

    if invert:
        input_list = input_list[::-1]

    # scale to 0-1
    col_list = []
    for color in input_list:
        scaled_color = convert_to_rgb(color)
        col_list.append(scaled_color)

    return mcolors.LinearSegmentedColormap.from_list("", col_list)


def remove_files(path, string, ext=False):
    """remove_files

    Remove files from a given path that containg a string as extension (`ext=True`), or at the
    start of the file (`ext=False`)

    Parameters
    ----------
    path: str
        path to the directory from which we need to remove files
    string: str
        tag for files we need to remove
    ext: str, optional
        only remove files containing `string` that end with `ext`
    """

    files_in_directory = os.listdir(path)

    if ext:
        filtered_files = [
            file for file in files_in_directory if file.endswith(string)]
    else:
        filtered_files = [
            file for file in files_in_directory if file.startswith(string)]

    for file in filtered_files:
        path_to_file = os.path.join(path, file)
        os.remove(path_to_file)
    
def filter_for_nans(array):
    """filter out NaNs from an array"""

    if np.isnan(array).any():
        return np.nan_to_num(array)
    else:
        return array
    
def calculate_tsnr(data, ax):
    mean_d = np.mean(data, axis=ax)
    std_d = np.std(data, axis=ax)
    tsnr = mean_d/std_d
    tsnr[np.where(np.isinf(tsnr))] = np.nan

    return tsnr


def percent_change(
    ts,
    ax,
    nilearn=False,
    baseline=20,
    prf=False,
    dm=None
):

    """percent_change

    Function to convert input data to percent signal change. Two options are current supported: the nilearn method
    (`nilearn=True`), where the mean of the entire timecourse if subtracted from the timecourse, and the baseline method
    (`nilearn=False`), where the median of `baseline` is subtracted from the timecourse.

    Parameters
    ----------
    ts: numpy.ndarray
        Array representing the data to be converted to percent signal change. Should be of shape (n_voxels, n_timepoints)
    ax: int
        Axis over which to perform the conversion. If shape (n_voxels, n_timepoints), then ax=1.
        If shape (n_timepoints, n_voxels), then ax=0.
    nilearn: bool, optional
        Use nilearn method, by default False
    baseline: int, list, np.ndarray optional
        Use custom method where only the median of the baseline (instead of the full timecourse) is subtracted, by default 20.
        Length should be in `volumes`, not `seconds`. Can also be a list or numpy array (1d) of indices which are to be
        considered as baseline. The list of indices should be corrected for any deleted volumes at the beginning.

    Returns
    ----------
    numpy.ndarray
        Array with the same size as `ts` (voxels,time), but with percent signal change.

    Raises
    ----------
    ValueError
        If `ax` > 2
    """

    if ts.ndim == 1:
        ts = ts[:, np.newaxis]
        ax = 0

    if prf:
        from fmriproc import prf

        # format data
        if ts.ndim == 1:
            ts = ts[..., np.newaxis]

        # read design matrix
        if isinstance(dm, str):
            dm = prf.read_par_file(dm)

        # calculate mean
        avg = np.mean(ts, axis=ax)
        ts *= (100/avg)

        # find points with no stimulus
        timepoints_no_stim = prf.baseline_from_dm(dm)

        # find timecourses with no stimulus
        if ax == 0:
            med_bsl = ts[timepoints_no_stim, :]
        else:
            med_bsl = ts[:, timepoints_no_stim]

        # calculat median over baseline
        median_baseline = np.median(med_bsl, axis=ax)

        # shift to zero
        ts -= median_baseline

        return ts
    else:
        if nilearn:
            if ax == 0:
                psc = signal._standardize(ts, standardize='psc')
            else:
                psc = signal._standardize(ts.T, standardize='psc').T
        else:

            # first step of PSC; set NaNs to zero if dividing by 0 (in case of crappy timecourses)
            ts_m = ts * \
                np.expand_dims(np.nan_to_num((100/np.mean(ts, axis=ax))), ax)

            # get median of baseline
            if isinstance(baseline, np.ndarray):
                baseline = list(baseline)

            if ax == 0:
                if isinstance(baseline, list):
                    median_baseline = np.median(ts_m[baseline, :], axis=0)
                else:
                    median_baseline = np.median(ts_m[:baseline, :], axis=0)
            elif ax == 1:
                if isinstance(baseline, list):
                    median_baseline = np.median(ts_m[:, baseline], axis=1)
                else:
                    median_baseline = np.median(ts_m[:, :baseline], axis=1)
            else:
                raise ValueError("ax must be 0 or 1")

            # subtract
            psc = ts_m-np.expand_dims(median_baseline, ax)

        return psc


def split_bids_components(fname, entities=False):

    comp_list = fname.split('_')
    comps = {}

    ids = ['sub', 'ses', 'task', 'acq', 'rec', 'run',
           'space', 'hemi', 'model', 'stage', 'desc', 'vox']

    full_entities = [
        "subject",
        "session",
        "task",
        "reconstruction",
        "acquisition",
        "run"
    ]
    for el in comp_list:
        for i in ids:
            if i in el:
                comp = el.split('-')[-1]

                if "." in comp:
                    ic = comp.index(".")
                    if ic > 0:
                        ex = 0
                    else:
                        ex = -1

                    comp = comp.split(".")[ex]

                # if i == "run":
                #     comp = int(comp)

                comps[i] = comp

    if len(comps) != 0:

        if entities:
            return comps, full_entities
        else:
            return comps
    else:
        raise ValueError(f"Could not find any element of {ids} in {fname}")


def get_ids(func_list, bids="task"):

    ids = []
    if isinstance(func_list, list):
        for ff in func_list:
            if isinstance(ff, str):
                bids_comps = split_bids_components(ff)
                if bids in list(bids_comps.keys()):
                    ids.append(bids_comps[bids])

    if len(ids) > 0:
        ids = list(np.unique(np.array(ids)))

        return ids
    else:
        return []


def str2operator(ops):

    if ops in ["and", "&", "&&"]:
        return operator.and_
    elif ops in ["or", "|", "||"]:
        return operator.or_
    elif ops in ["is not", "!="]:
        return operator.ne
    elif ops in ["is", "==", "="]:
        return operator.eq
    elif ops in ["gt", ">"]:
        return operator.gt
    elif ops in ["lt", "<"]:
        return operator.lt
    elif ops in ["ge", ">="]:
        return operator.ge
    elif ops in ["le", "<="]:
        return operator.le
    elif ops in ["x", "*"]:
        return operator.mul
    elif ops == "/":
        return operator.truediv
    else:
        raise NotImplementedError()


def select_from_df(
    df,
    expression=None,
    index=True,
    indices=None,
    match_exact=True
):
    """select_from_df

    Select a subset of a dataframe based on an expression. Dataframe should be indexed by the variable you want to select on
    or have the variable specified in the expression argument as column name. If index is True, the dataframe will be indexed
    by the selected variable. If indices is specified, the dataframe will be indexed by the indices specified through a list
    (only select the elements in the list) or a `range`-object (select within range).

    Parameters
    ----------
    df: pandas.DataFrame
        input dataframe
    expression: str, optional
        what subject of the dataframe to select, by default None. The expression must consist of a variable name and an
        operator. The operator can be any of the following: '=', '>', '<', '>=', '<=', '!=', separated by spaces. You can also
        change 2 operations by specifying the `&`-operator between the two expressions. If you want to use `indices`, specify
        `expression="ribbon"`.
    index: bool, optional
        return output dataframe with the same indexing as `df`, by default True
    indices: list, range, numpy.ndarray, optional
        List, range, or numpy array of indices to select from `df`, by default None
    match_exact: bool, optional:
        When you insert a list of strings with `indices` to be filtered from the dataframe, you can either request that the
        items of `indices` should **match** exactly (`match_exact=True`, default) the column names of `df`, or whether the
        columns of `df` should **contain** the items of `indices` (`match_exact=False`).

    Returns
    ----------
    pandas.DataFrame
        new dataframe where `expression` or `indices` were selected from `df`

    Raises
    ----------
    TypeError
        If `indices` is not a tuple, list, or array

    Notes
    ----------
    See https://linescanning.readthedocs.io/en/latest/examples/nideconv.html for an example of how to use this function (do
    ctrl+F and enter "select_from_df").
    """

    # not the biggest fan of a function within a function, but this allows
    # easier translation of expressions/operators
    def sort_expressions(expression):
        expr_ = expression.split(" ")
        if len(expr_) > 3:
            for ix, i in enumerate(expr_):
                try:
                    _ = str2operator(i)
                    break
                except BaseException:
                    pass

            col1 = " ".join(expr_[:ix])
            val1 = " ".join(expr_[(ix + 1):])
            operator1 = expr_[ix]
        else:
            col1, operator1, val1 = expr_

        return col1, operator1, val1

    if isinstance(indices, (list, tuple, np.ndarray)):

        if isinstance(indices, tuple):
            return df.iloc[:, indices[0]:indices[1]]
        elif isinstance(indices, list):
            if all(isinstance(item, str) for item in indices):
                if match_exact:
                    return df[df.columns[df.columns.isin(indices)]]
                else:
                    df_tmp = []
                    for item in indices:
                        df_tmp.append(
                            df[df.columns[df.columns.str.contains(item)]])

                    return pd.concat(df_tmp, axis=1)
            else:
                return df.iloc[:, indices]
        elif isinstance(indices, np.ndarray):
            return df.iloc[:, list(indices)]
        else:
            raise TypeError(
                f"""Unknown type '{type(indices)}' for indices; must be a tuple of 2 values representing a range, or a
                list/array of indices to select""")
    else:
        if not isinstance(expression, (str, tuple, list)):
            raise ValueError(
                f"""Please specify expressions to apply to the dataframe.
                Input is '{expression}' of type ({type(expression)})"""
            )

        # fetch existing indices
        idc = list(df.index.names)
        if idc[0] is not None:
            reindex = True
        else:
            reindex = False

        # sometimes throws an error if you're trying to reindex a non-indexed
        # dataframe
        try:
            df = df.reset_index()
        except BaseException:
            pass

        sub_df = df.copy()
        if isinstance(expression, str):
            expression = [expression]

        if isinstance(expression, (tuple, list)):

            expressions = expression[::2]
            operators = expression[1::2]

            if len(expressions) == 1:

                # find operator index
                col1, operator1, val1 = sort_expressions(expressions[0])

                # convert to operator function
                ops1 = str2operator(operator1)

                # use dtype of whatever dtype the colum is
                search_value = np.array(
                    [val1], dtype=type(
                        sub_df[col1].values[0]))
                sub_df = sub_df.loc[ops1(sub_df[col1], search_value[0])]

            if len(expressions) == 2:
                col1, operator1, val1 = sort_expressions(expressions[0])
                col2, operator2, val2 = sort_expressions(expressions[1])

                main_ops = str2operator(operators[0])
                ops1 = str2operator(operator1)
                ops2 = str2operator(operator2)

                # check if we should interpret values invididually as integers
                search_value1 = np.array(
                    [val1], dtype=type(
                        sub_df[col1].values[0]))[0]
                search_value2 = np.array(
                    [val2], dtype=type(
                        sub_df[col2].values[0]))[0]

                sub_df = sub_df.loc[main_ops(
                    ops1(sub_df[col1], search_value1), ops2(sub_df[col2], search_value2))]

        # first check if we should do indexing
        if index is not None:
            # then check if we actually have something to index
            if reindex:
                if idc[0] is not None:
                    sub_df = sub_df.set_index(idc)

        if sub_df.shape[0] == 0:
            raise ValueError(
                f"The following expression(s) resulted in an empty dataframe: {expression}")

        return sub_df


def multiselect_from_df(df, expression=[]):

    if not isinstance(expression, list):
        raise TypeError(
            f"expression must be list of tuples (see docs utils.select_from_df), not {type(expression)}")

    if len(expression) == 0:
        raise ValueError("List is empty")

    for expr in expression:
        df = select_from_df(df, expression=expr)

    return df


def round_decimals_down(number: float, decimals: int = 2):
    """
    Returns a value rounded down to a specific number of decimal places.
    see: https://kodify.net/python/math/round-decimals/#round-decimal-places-up-and-down-round
    """
    if not isinstance(decimals, int):
        raise TypeError("decimal places must be an integer")
    elif decimals < 0:
        raise ValueError("decimal places has to be 0 or more")
    elif decimals == 0:
        return math.floor(number)

    factor = 10 ** decimals
    return math.floor(number * factor) / factor


def round_decimals_up(number: float, decimals: int = 2):
    """
    Returns a value rounded up to a specific number of decimal places.
    see: https://kodify.net/python/math/round-decimals/#round-decimal-places-up-and-down-round
    """
    if not isinstance(decimals, int):
        raise TypeError("decimal places must be an integer")
    elif decimals < 0:
        raise ValueError("decimal places has to be 0 or more")
    elif decimals == 0:
        return math.ceil(number)

    factor = 10 ** decimals
    return math.ceil(number * factor) / factor


def verbose(msg, verbose, flush=True, **kwargs):
    if verbose:
        print(msg, flush=flush, **kwargs)


def get_unique_ids(df, id=None, sort=True, as_int=False, drop_na=True):
    try:
        df = df.reset_index()
    except BaseException:
        pass

    if not isinstance(id, str):
        raise ValueError("Please specify a identifier from the dataframe")

    try:
        a = df[id].values
        if not sort:
            indexes = np.unique(a, return_index=True)[1]
            ret_list = [a[index] for index in sorted(indexes)]
        else:
            ret_list = list(np.unique(a))

        # https://stackoverflow.com/a/50297200
        if drop_na:
            ret_list = [x for x in ret_list if x == x]

        if as_int:
            ret_list = [int(i) for i in ret_list]
        return ret_list

    except Exception:
        raise RuntimeError(f"Could not find '{id}' in {list(df.columns)}")

# Define a function 'pairwise' that iterates over all pairs of consecutive
# items in a list


def pairwise(l1):
    # Create an empty list 'temp' to store the pairs
    temp = []

    # Iterate through the list elements up to the second-to-last element
    for i in range(len(l1) - 1):
        # Get the current element and the next element in the list
        current_element, next_element = l1[i], l1[i + 1]

        # Create a tuple 'x' containing the current and next elements
        x = (current_element, next_element)

        # Append the tuple 'x' to the 'temp' list
        temp.append(x)

    # Return the list of pairs
    return temp


def get_file_from_substring(filt, path, return_msg='error', exclude=None):
    """get_file_from_substring

    This function returns the file given a path and a substring. Avoids annoying stuff with glob. Now also allows multiple
    filters to be applied to the list of files in the directory. The idea here is to construct a binary matrix of shape
    (files_in_directory, nr_of_filters), and test for each filter if it exists in the filename. If all filters are present in
    a file, then the entire row should be 1. This is what we'll be looking for. If multiple files are found in this manner, a
    list of paths is returned. If only 1 file was found, the string representing the filepath will be returned.

    Parameters
    ----------
    filt: str, list
        tag for files we need to select. Now also support a list of multiple filters.
    path: str
        path to the directory from which we need to remove files
    return_msg: str, optional
        whether to raise an error (*return_msg='error') or return None (*return_msg=None*). Default = 'error'.
    exclude: str, optional:
        Specify string to exclude from options. This criteria will be ensued after finding files that conform to `filt` as
        final filter.

    Returns
    ----------
    str
        path to the files containing `string`. If no files could be found, `None` is returned

    list
        list of paths if multiple files were found

    Raises
    ----------
    FileNotFoundError
        If no files usingn the specified filters could be found

    Example
    ----------
    >>> get_file_from_substring("R2", "/path/to/prf")
    '/path/to/prf/r2.npy'
    >>> get_file_from_substring(['gauss', 'best_vertices'], "path/to/pycortex/sub-xxx")
    '/path/to/pycortex/sub-xxx/sub-xxx_model-gauss_desc-best_vertices.csv'
    >>> get_file_from_substring(['best_vertices'], "path/to/pycortex/sub-xxx")
    ['/path/to/pycortex/sub-xxx/sub-xxx_model-gauss_desc-best_vertices.csv',
    '/path/to/pycortex/sub-xxx/sub-xxx_model-norm_desc-best_vertices.csv']
    """

    input_is_list = False
    if isinstance(filt, str):
        filt = [filt]

    if isinstance(exclude, str):
        exclude = [exclude]

    if isinstance(filt, list):
        # list and sort all files in the directory
        if isinstance(path, str):
            files_in_directory = sorted(os.listdir(path))
        elif isinstance(path, list):
            input_is_list = True
            files_in_directory = path.copy()
        else:
            raise ValueError(
                "Unknown input type; should be string to path or list of files")

        # the idea is to create a binary matrix for the files in 'path', loop
        # through the filters, and find the row where all values are 1
        filt_array = np.zeros((len(files_in_directory), len(filt)))
        for ix, f in enumerate(files_in_directory):
            for filt_ix, filt_opt in enumerate(filt):
                filt_array[ix, filt_ix] = filt_opt in f

        # now we have a binary <number of files x number of filters> array. If all filters were available in a file, the
        # entire row should be 1,
        # so we're going to look for those rows
        full_match = np.ones(len(filt))
        full_match_idc = np.where(np.all(filt_array == full_match, axis=1))[0]

        if len(full_match_idc) == 1:
            fname = files_in_directory[full_match_idc[0]]
            if input_is_list:
                return fname
            else:
                f = opj(path, fname)
                if isinstance(exclude, list):
                    if not any(x in f for x in exclude):
                        return opj(path, fname)
                    else:
                        if return_msg == "error":
                            raise FileNotFoundError(
                                f"Could not find file with filters: {filt} and exclusion of [{exclude}] in '{path}'")
                        else:
                            return None
                else:
                    return opj(path, fname)

        elif len(full_match_idc) > 1:
            match_list = []
            for match in full_match_idc:
                fname = files_in_directory[match]
                if input_is_list:
                    match_list.append(fname)
                else:
                    match_list.append(opj(path, fname))

            if isinstance(exclude, list):
                exl_list = []
                for f in match_list:
                    if not any(x in f for x in exclude):
                        exl_list.append(f)

                # return the string if there's only 1 element
                if len(exl_list) == 1:
                    return exl_list[0]
                else:
                    return exl_list
            else:
                return match_list
            # return match_list
        else:
            if return_msg == "error":
                raise FileNotFoundError(
                    f"Could not find file with filters: {filt} in {path}")
            else:
                return None


def update_kwargs(kwargs, el, val, force=False):
    if not force:
        if el not in list(kwargs.keys()):
            kwargs[el] = val
    else:
        kwargs[el] = val

    return kwargs


def find_nearest(array, value, return_nr=1):
    """find_nearest

    Find the index and value in an array given a value. You can either choose to have 1 item (the `closest`) returned, or the
    5 nearest items (`return_nr=5`), or everything you're interested in (`return_nr="all"`)

    Parameters
    ----------
    array: numpy.ndarray
        array to search in
    value: float
        value to search for in `array`
    return_nr: int, str, optional
        number of elements to return after searching for elements in `array` that are close to `value`. Can either be an
        integer or a string *all*

    Returns
    ----------
    int
        integer representing the index of the element in `array` closest to `value`.

    list
        if `return_nr` > 1, a list of indices will be returned

    numpy.ndarray
        value in `array` at the index closest to `value`
    """

    array = np.asarray(array)

    if return_nr == 1:
        idx = np.nanargmin((np.abs(array-value)))
        return idx, array[idx]
    else:

        # check nan indices
        nans = np.isnan(array)

        # initialize output
        idx = np.full_like(array, np.nan)

        # loop through values in array
        for qq, ii in enumerate(array):

            # don't do anything if value is nan
            if not nans[qq]:
                idx[qq] = np.abs(ii-value)

        # sort
        idx = np.argsort(idx)

        # return everything
        if return_nr == "all":
            idc_list = idx.copy()
        else:
            # return closest X values
            idc_list = idx[:return_nr]

        return idc_list, array[idc_list]


def find_intersection(xx, curve1, curve2):
    """find_intersection

    Find the intersection coordinates given two functions using `Shapely`.

    Parameters
    ----------
    xx: numpy.ndarray
        array describing the x-axis values
    curve1: numpy.ndarray
        array describing the first curve
    curve2: numpy.ndarray
        array describing the first curve

    Returns
    ----------
    tuple
        x,y coordinates where *curve1* and *curve2* intersect

    Raises
    ----------
    ValueError
        if no intersection coordinates could be found

    Example
    ----------
    See [refer to linescanning.prf.SizeResponse.find_stim_sizes]
    """

    first_line = geometry.LineString(np.column_stack((xx, curve1)))
    second_line = geometry.LineString(np.column_stack((xx, curve2)))
    geom = first_line.intersection(second_line)

    try:
        if isinstance(geom, geometry.multipoint.MultiPoint):
            # multiple coordinates
            coords = [i.coords._coords for i in list(geom.geoms)]
        elif isinstance(geom, geometry.point.Point):
            # single coordinate
            coords = [geom.coords._coords]
        elif isinstance(geom, geometry.collection.GeometryCollection):
            # coordinates + line segments
            mapper = geometry.mapping(geom)
            coords = []
            for el in mapper["geometries"]:
                coor = np.array(el["coordinates"])
                if coor.ndim > 1:
                    coor = coor[0]
                # to make indexing same as above
                coords.append(coor[np.newaxis, ...])
        else:
            raise ValueError(f"Can't deal with output of type {type(geom)}")
            # coords = geom
    except Exception:
        raise ValueError("Could not find intersection between curves..")

    return coords
