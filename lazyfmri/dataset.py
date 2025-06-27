import os
import json
import pickle
import numpy as np
import pandas as pd
import nibabel as nb
from scipy import io
import matplotlib.pyplot as plt
from . import glm, plotting, preproc, utils

try:
    import hedfpy
    HEDFPY_AVAILABLE = True
except ImportError:
    HEDFPY_AVAILABLE = False

opj = os.path.join
# disable warning thrown by string2float
pd.options.mode.chained_assignment = None

rb = f"{utils.color.RED}{utils.color.BOLD}"
gb = f"{utils.color.GREEN}{utils.color.BOLD}"
bm = f"{utils.color.MAGENTA}{utils.color.BOLD}"
end = utils.color.END

def filter_kwargs(ignore_kwargs, kwargs):

    if isinstance(ignore_kwargs, str):
        ignore_kwargs = [ignore_kwargs]

    tmp_kwargs = {}
    for ii in kwargs:
        if ii not in ignore_kwargs:
            tmp_kwargs[ii] = kwargs[ii]
    return tmp_kwargs


def check_input_is_list(obj, var=None, list_element=0, matcher="func_file"):

    if hasattr(obj, var):
        attr = getattr(obj, var)
    else:
        raise ValueError(f"Class does not have '{var}'-attribute")

    if isinstance(attr, (list, np.ndarray)):
        if len(attr) != len(getattr(obj, matcher)):
            raise ValueError(
                f"""
Length of '{var}' ({len(attr)}) does not match number of func files ({len(getattr(obj,matcher))}).
Either specify a list of equal lenghts or 1 integer value for all volumes""")

        return attr[list_element]
    else:
        return attr


class SetAttributes():

    def __init__(self):

        # store ParseEyetracker attributes
        self.eye_attributes = [
            "df_blinks",
            "df_space_func",
            "df_space_eye",
            "df_saccades"
        ]

        # store ParseExptoolsFile attributes
        self.exp_attributes = [
            "df_onsets",
            "df_rts",
            "df_accuracy",
            "df_responses",
        ]

        # store ParseFuncFile attributes
        self.func_attributes = [
            "df_func_psc",
            "df_func_raw",
            "df_func_zscore",
            "df_func_ica",
        ]

        # combine them all for Dataset-class
        self.all_attributes = self.eye_attributes + \
            self.exp_attributes+self.func_attributes


class ParseEyetrackerFile(SetAttributes):

    """ParseEyetrackerFile

    Class for parsing edf-files created during experiments with Exptools2. The class will read in the file, read when the
    experiment actually started, correct onset times for this start time and time deleted because of removing the first few
    volumes (to do this correctly, set the `TR` and `deleted_first_timepoints`). You can also provide a numpy array/file
    containing eye blinks that should be added to the onset times in real-world time (seconds). In principle, it will return a
    pandas DataFrame indexed by subject and run that can be easily concatenated over runs. This function relies on the naming
    used when programming the experiment. In the `session.py` file, you should have created ``phase_names=['iti', 'stim']``; the
    class will use these things to parse the file.


    Parameters
    ----------
    edf_file: str, list
        path pointing to the output file of the experiment; can be a list of multiple. Ideally, all these files belong to 1
        subject, otherwise it tries to write everything to 1 file, which is too much
    subject: int
        subject number in the returned pandas DataFrame (should start with 1, ..., n)
    run: int
        run number you'd like to have the onset times for
    low_pass_pupil_f: float, optional
        Low-pass cutoff frequency
    high_pass_pupil_f: float, optional
        High-pass cutoff frequency
    TR: float, optional (fMRI)
        Repetition time of experiment. Together with `nr_vols`, used to determine the period that needs to be extracted after
        onset of the first trial. Default = None
    nr_vols: int, optional (fMRI)
        Together with `TR`, used to determine the period that needs to be extracted after onset
        of the first trial. Default = None
    deleted_first_timepoints: int
        number of volumes to delete to correct onset times for deleted volumes
    h5_file: str, optional
        Custom path to h5-file in which to store the complete output from `edf_file`. If nothing's specified, it'll output an
        `eye.h5`-file in the directory of the first edf-file in the list.

    Examples
    ----------
    .. code-block:: python

        from lazyfmri.dataset import ParseExpToolsFile
        file = 'some/path/to/exptoolsfile.tsv'
        parsed_file = ParseExpToolsFile(
            file,
            subject=1,
            run=1,
            button=True
        )

        onsets = parsed_file.get_onset_df()

    Notes
    ----------
    If you have self-paced experiments or want to extract the full data from the eyetracker, keep `TR` and `nr_vols` **None**.
    """

    def __init__(
        self,
        edf_file,
        eye_h5=None,
        subject=1,
        run=1,
        task=None,
        low_pass_pupil_f=6.0,
        high_pass_pupil_f=0.01,
        func_file=None,
        TR=0.105,
        verbose=False,
        use_bids=True,
        nr_vols=None,
        save_as=None,
        invoked_from_func=False,
        overwrite=False,
        **kwargs
        ):

        super().__init__()

        if not HEDFPY_AVAILABLE:
            raise ModuleNotFoundError(
                "could not find 'hedfpy', so this functionality is disabled")

        self.edf_file = edf_file
        self.func_file = func_file
        self.sub = subject
        self.run = run
        self.task = task
        self.TR = TR
        self.low_pass_pupil_f = low_pass_pupil_f
        self.high_pass_pupil_f = high_pass_pupil_f
        self.verbose = verbose
        self.use_bids = use_bids
        self.nr_vols = nr_vols
        self.save_as = save_as
        self.invoked_from_func = invoked_from_func
        self.overwrite = overwrite
        self.eye_h5 = eye_h5
        self.__dict__.update(kwargs)

        # deal with edf-files
        self.edf_file = self.check_input(self.edf_file)
        self.func_file = self.check_input(self.func_file)

        # add all files to h5-file
        if isinstance(self.edf_file, (str, list)):

            # print message
            utils.verbose("\nEYETRACKER", self.verbose)

            # preprocess and fetch dataframes
            self.set_indices()
            self.define_hdf_file()
            self.write_edf_to_hdf()
            self.preprocess_edf_files()
            self.ho.close_hdf_file()

    def set_indices(self):
        # check if we should index task
        self.index_task = False
        self.blink_index = ['subject', 'run', 'event_type']
        self.eye_index = ['subject', 'run', 'eye', 't']
        self.sac_index = ['subject', 'run', 'event_type']
        if self.use_bids:
            self.task_ids = utils.get_ids(self.edf_file, bids="task")

            # insert task id in indexer
            if len(self.task_ids) > 1:
                self.index_task = True
                for idx in ["blink_index", "eye_index", "sac_index"]:
                    getattr(self, idx).insert(1, "task")

    def check_input(self, in_files):
        # deal with files
        if in_files is not None:
            if isinstance(in_files, str):
                return [os.path.abspath(in_files)]
            elif isinstance(in_files, list):
                return [os.path.abspath(i) for i in in_files]
            else:
                raise ValueError(
                    f"Input must be 'str' or 'list', not '{type(in_files)}'")

    def define_hdf_file(self):
        # get basename
        self.get_bids_info(self.edf_file[0], 0)
        self.base_name = self.get_base_name(
            subID=self.sub,
            sesID=self.ses
        )

        # check if there's an instance of h5_file
        if not isinstance(self.eye_h5, str):
            if self.save_as is not None:
                self.eye_h5 = opj(self.save_as, f"{self.base_name}_desc-preproc_eye.h5")
            else:
                self.eye_h5 = opj(os.path.dirname(self.edf_file[0]), "eye.h5")

        # check if we should overwrite
        if self.overwrite:
            if os.path.exists(self.eye_h5):
                store = pd.HDFStore(self.eye_h5)
                store.close()
                os.remove(self.eye_h5)

        self.ho = hedfpy.HDFEyeOperator(self.eye_h5)

    def write_edf_to_hdf(self):

        if not os.path.exists(self.eye_h5):
            for i, edf_file in enumerate(self.edf_file):

                if not os.path.exists(edf_file):
                    raise FileNotFoundError(
                        f"Could not read specified file: '{edf_file}'")

                self.run = i+1
                self.sub = 1
                if self.use_bids:
                    bids_comps = utils.split_bids_components(edf_file)
                    for el in ['sub', 'run', 'task']:
                        if el in list(bids_comps.keys()):
                            setattr(self, el, bids_comps[el])

                # set alias
                alias = f"run_{self.run}"
                if isinstance(self.task, str):
                    alias = f"task_{self.task}_run_{self.run}"

                self.ho.add_edf_file(edf_file)
                self.ho.edf_message_data_to_hdf(alias=alias)
                self.ho.edf_gaze_data_to_hdf(
                    alias=alias,
                    pupil_hp=self.high_pass_pupil_f,
                    pupil_lp=self.low_pass_pupil_f
                )

        else:
            self.ho.open_hdf_file()

        # clean up hedfpy-files
        for ext in [".pdf", ".gaz", ".msg", ".gaz.gz", ".asc"]:
            utils.remove_files(
                os.path.dirname(self.eye_h5),
                ext,
                ext=True
            )

    def get_base_name(
        self,
        subID=None,
        sesID=None,
        taskID=None
        ):

        base_name = ""

        # set base name based on presence of bids tags
        for key, val in zip(["sub", "ses", "task"], [subID, sesID, taskID]):
            if isinstance(val, (str, float, int)):
                if key == "sub":
                    base_name = f"{key}-{val}"
                else:
                    base_name += f"_{key}-{val}"

        return base_name

    def get_tr(self, run=None):
        if self.TR is not None:
            use_TR = check_input_is_list(
                self,
                "TR",
                list_element=run,
                matcher="edf_file"
            )
        else:
            use_TR = None

        return use_TR

    def get_vols(self, i):
        nr_vols = None
        if isinstance(self.func_file, list):
            nr_vols = self.vols(self.func_file[i])

        return nr_vols

    def get_bids_info(self, file, i):
        self.run = i+1
        self.ses = None
        self.task = None
        if self.use_bids:
            bids_comps = utils.split_bids_components(file)
            for el in ['sub', 'run', 'ses', 'task']:
                if el in list(bids_comps.keys()):
                    setattr(self, el, bids_comps[el])

    def fetch_extracted_data(
        self,
        run,
        task,
        TR,
        nr_vols
        ):

        alias = f"run_{run}"
        if isinstance(task, str):
            alias = f"task_{task}_run_{run}"

        fetch_data = True
        try:
            self.data = self.fetch_relevant_info(
                TR=TR,
                task=task,
                nr_vols=nr_vols,
                alias=alias
            )
        except Exception:
            fetch_data = False

        # collect outputs if all went well
        if fetch_data:
            self.df_blinks.append(self.fetch_eyeblinks())
            self.df_space_func.append(self.fetch_eye_func_time())
            self.df_space_eye.append(self.fetch_eye_tracker_time())
            self.df_saccades.append(self.fetch_saccades())

    def concat_dataframes(self):
        for ii, ll in zip(
            ["df_blinks", "df_space_eye", "df_saccades"],
                [self.blink_index, self.eye_index, self.sac_index]):
            if len(getattr(self, ii)) > 0:
                setattr(self, ii, pd.concat(getattr(self, ii)).set_index(ll))

        if len(self.df_space_func) > 0:
            # check if all elements are dataframes
            if all(isinstance(x, pd.DataFrame) for x in self.df_space_func):
                self.df_space_func = pd.concat(
                    self.df_space_func).set_index(self.eye_index)

    def preprocess_edf_files(self):

        # set them for internal reference
        for attr in self.eye_attributes:
            setattr(self, attr, [])

        for i, edf_file in enumerate(self.edf_file):

            utils.verbose(f"Preprocessing {gb}{edf_file}{end}", self.verbose)
            self.get_bids_info(edf_file, i)

            # get basename
            self.base_name = self.get_base_name(
                subID=self.sub,
                sesID=self.ses,
                taskID=self.task
            )

            # check if we got multiple TRs for different edf-files
            use_TR = self.get_tr(run=self.run)

            # check volumes
            nr_vols = self.get_vols(i)

            # fetch relevant data
            self.fetch_extracted_data(
                self.run,
                self.task,
                use_TR,
                nr_vols
            )

        # concatenate available dataframes
        self.concat_dataframes()

    def fetch_blinks_run(self, run=1, return_type='df'):
        blink_df = utils.select_from_df(
            self.df_blinks,
            expression=(f"run = {run}"),
            index=['subject', 'run', 'event_type'])

        if return_type == "df":
            return blink_df
        else:
            return blink_df.values

    def fetch_eyeblinks(self):
        return self.data['blink_events']

    def fetch_saccades(self):
        return self.data['saccades']

    def fetch_eye_func_time(self):
        return self.data['space_func']

    def fetch_eye_tracker_time(self):
        return self.data['space_eye']

    def fetch_relevant_info(
        self,
        task=None,
        nr_vols=None,
        alias=None,
        TR=None,
        save_as=None
        ):

        # load times per session:
        trial_times = self.ho.read_session_data(alias, 'trials')

        # fetch duration of scan or read until end of edf file
        use_end_point = True
        if TR is not None and nr_vols is not None:
            func_time = nr_vols*TR
            use_end_point = False

        # get block parameters
        self.session_start_EL_time = trial_times.iloc[0, 0]
        self.sample_rate = self.ho.sample_rate_during_period(alias)

        # add number of fMRI*samplerate as stop EL time or read until end of edf file
        if not use_end_point:
            self.session_stop_EL_time = self.session_start_EL_time + \
                (func_time*self.sample_rate)
        else:
            self.session_stop_EL_time = self.ho.block_properties(
                alias)["block_end_timestamp"].iloc[0]

        # define period
        self.time_period = [self.session_start_EL_time,
                            self.session_stop_EL_time]

        eye = self.ho.eye_during_period(self.time_period, alias)
        utils.verbose(f" Eye:         {rb}{eye}{end}", self.verbose)
        utils.verbose(f" Sample rate: {rb}{self.sample_rate}{end}", self.verbose)
        utils.verbose(f" Start time:  {rb}{self.time_period[0]}{end}", self.verbose)
        utils.verbose(f" Stop time:   {rb}{self.time_period[1]}{end}", self.verbose)

        # set some stuff required for successful plotting with seconds on the x-axis
        n_samples = int(self.time_period[1]-self.time_period[0])
        duration_sec = n_samples*(1/self.sample_rate)

        utils.verbose(
            f" Duration:    {rb}{duration_sec}{end}s [{rb}{n_samples}{end} samples]", self.verbose)

        # Fetch a bunch of data
        extract = [
            "pupil",
            "pupil_int",
            "gaze_x_int",
            "gaze_y_int",
            "gaze_x",
            "gaze_y"]

        utils.verbose(f" Fetching:    [{extract}]", self.verbose)

        tf = []
        tf_rs = []
        df_space_func = None
        for par_ix, extr in enumerate(extract):
            data = np.squeeze(
                self.ho.signal_during_period(
                    time_period=self.time_period,
                    alias=alias,
                    signal=extr,
                    requested_eye=eye
                ).values
            )

            if data.ndim < 2:
                data = data[..., np.newaxis]

            tmp = []
            tmp_rs = []
            for ix, ii in enumerate(list(eye)):

                if isinstance(nr_vols, int):
                    rs = glm.resample_stim_vector(data[:, ix], nr_vols)
                    hp = preproc.highpass_dct(
                        rs,
                        self.high_pass_pupil_f,
                        TR=TR
                    )[0]

                    psc = np.squeeze(
                        utils.percent_change(
                            rs,
                            0,
                            baseline=hp.shape[0]
                        )
                    )

                    psc_hp = np.squeeze(
                        utils.percent_change(
                            hp,
                            0,
                            baseline=hp.shape[0]
                        )
                    )

                    tmp_rs_df = pd.DataFrame({
                        f"{extr}": rs,
                        f"{extr}_psc": psc,
                        f"{extr}_hp": hp,
                        f"{extr}_hp_psc": psc_hp}
                    )

                    if par_ix == len(extr)-1:
                        tmp_rs_df["eye"] = ii
                        tmp_rs_df["t"] = list(
                            (1/self.sample_rate)*np.arange(rs.shape[0]))

                    tmp_rs.append(tmp_rs_df)

                psc = np.squeeze(
                    utils.percent_change(
                        data[:, ix],
                        0,
                        baseline=data[:, ix].shape[0]
                    )
                )

                tmp_df = pd.DataFrame({
                    f"{extr}": data[:, ix],
                    f"{extr}_psc": psc})

                if par_ix == len(extr)-1:
                    tmp_df["eye"] = ii
                    tmp_df["t"] = list((1/self.sample_rate) * np.arange(data.shape[0]))

                tmp.append(tmp_df)

            tmp = pd.concat(tmp)
            tf.append(tmp)

            if len(tmp_rs) > 0:
                tmp_df = pd.concat(tmp_rs)
                tf_rs.append(tmp_df)

        df_space_eye = pd.concat(tf, axis=1)

        # nothing to resample if nr_vols==None
        if len(tf_rs) > 0:
            df_space_func = pd.concat(tf_rs, axis=1)

        # add start time to it
        start_exp_time = trial_times.iloc[0, :][-1]

        utils.verbose(f" Start time:  {rb}{round(start_exp_time, 2)}{end}s", self.verbose)
        # get onset time of blinks, cluster blinks that occur within 350 ms
        bb = self.ho.blinks_during_period(self.time_period, alias=alias)

        onsets = bb["start_timestamp"].values

        # convert the onsets to seconds in experiment
        onsets = (onsets-self.time_period[0])*(1/self.sample_rate)

        # normal eye blink is 1 blink every 4 seconds, throw warning if we found more than a blink per second
        # ref: https://www.sciencedirect.com/science/article/abs/pii/S0014483599906607
        blink_rate = len(onsets)/duration_sec

        utils.verbose(
            f" Nr blinks:   {rb}{len(onsets)}{end} [{rb}{round(blink_rate, 2)}{end} blinks per second]", self.verbose)

        # extract saccades
        df_saccades = self.ho.detect_saccades_during_period(
            time_period=self.time_period, alias=alias)

        if len(df_saccades) > 0:
            utils.verbose(
                f" Nr saccades: {rb}{len(df_saccades)}{end} saccades", self.verbose)

            # convert saccade onset to seconds relative to start of run
            df_saccades["onset"] = df_saccades["expanded_start_time"] / \
                self.sample_rate

            # add some stuff for indexing
            for tt, mm in zip(["subject", "run", "event_type"], [self.sub, self.run, "saccade"]):
                df_saccades[tt] = mm

            # set task index
            df_saccades = self.set_task_index(df_saccades, task=task)

        # build dataframe with relevant information
        self.tmp_df = df_space_eye.copy()
        df_space_eye = pd.DataFrame(df_space_eye)

        # index
        df_space_eye['subject'], df_space_eye['run'] = self.sub, self.run

        # set task index
        df_space_eye = self.set_task_index(df_space_eye, task=task)

        # index
        if isinstance(df_space_func, pd.DataFrame):
            df_space_func['subject'], df_space_func['run'] = self.sub, self.run

            # set task index
            df_space_func = self.set_task_index(df_space_func, task=task)

        # index
        df_blink_events = pd.DataFrame(onsets, columns=['onset'])

        for tt, mm in zip(['subject', 'run', 'event_type'], [self.sub, self.run, "blink"]):
            df_blink_events[tt] = mm

        # set task index
        df_blink_events = self.set_task_index(df_blink_events, task=task)

        utils.verbose("Done", self.verbose)

        return_dict = {
            "space_eye": df_space_eye,
            "space_func": df_space_func,
            "blink_events": df_blink_events}

        if len(df_saccades) > 0:
            return_dict["saccades"] = df_saccades

        if self.report:
            # make some plots
            if save_as is None:
                save_as = opj(os.getcwd(), "figures")
                if not os.path.exists(save_as):
                    os.makedirs(save_as, exist_ok=True)

            fname = opj(save_as, f"{self.base_name}_run-{self.run}_desc")
            self.plot_trace_and_heatmap(
                df_space_eye,
                fname=fname
            )

        return return_dict


    def set_task_index(self, df, task=None):
        # index task if required
        if self.index_task:
            if isinstance(task, str):
                df["task"] = task

        return df

    def plot_trace_and_heatmap(
        self,
        df,
        fname=None,
        screen_size=(1920, 1080),
        scale="screen"
        ):

        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Input must be a pd.Dataframe, not {type(df)}")

        use_eyes = list(np.unique(df.reset_index()['eye']))

        fig = plt.figure(figsize=(24, len(use_eyes)*6),
                         constrained_layout=True)
        if len(use_eyes) > 1:
            y = 1.075
            hspace = 0.12
        else:
            y = 1.15
            hspace = 0

        sf = fig.subfigures(nrows=len(use_eyes), hspace=hspace)

        fig.suptitle('positional stability', fontsize=32, y=y)

        for ii, eye in enumerate(use_eyes):
            if len(use_eyes) > 1:
                sf_a = sf[ii]
            else:
                sf_a = sf

            axs = sf_a.subplots(ncols=2, gridspec_kw={
                "width_ratios": [0.35, 0.65],
                "hspace": hspace})

            eye_df = utils.select_from_df(df, expression=f"eye = {eye}")
            input_l = [eye_df[f"gaze_{i}_int"].values for i in ["x", "y"]]

            # x-coordinate of the fixation cross
            x_cor_fixcross = screen_size[0]/2
            # y-coordinate of the fixation cross
            y_cor_fixcross = screen_size[1]/2
            x_coor_relfix = (input_l[0] - x_cor_fixcross)
            y_coor_relfix = (input_l[1] - y_cor_fixcross)

            if isinstance(scale, str):
                if scale == "auto":
                    ext = [x_coor_relfix.min(), x_coor_relfix.max(),
                           y_coor_relfix.min(), y_coor_relfix.max()]
                elif scale == "screen":
                    xlim = (-screen_size[0]//2, screen_size[0]//2)
                    ylim = (-screen_size[1]//2, screen_size[1]//2)
                    ext = [xlim[0], xlim[1], ylim[0], ylim[1]]
                else:
                    raise ValueError(
                        f"Unknown scale '{scale}' specified. Must be 'auto' or 'screen', or specify a tuple")

            axs[0].hexbin(
                x_coor_relfix,
                y_coor_relfix,
                gridsize=100,
                cmap="magma",
                extent=ext
            )

            axs[0].axhline(y=0, color='white', linestyle='-', linewidth=0.5)
            axs[0].axvline(x=0, color='white', linestyle='-', linewidth=0.5)
            plotting.conform_ax_to_obj(
                ax=axs[0],
                y_label="y position",
                x_label="x position"
            )

            avg = [float(input_l[i].mean()) for i in range(len(input_l))]
            std = [float(input_l[i].std()) for i in range(len(input_l))]
            plotting.LazyLine(
                input_l,
                line_width=2,
                axs=axs[1],
                color=["#1B9E77", "#D95F02"],
                labels=[f"gaze {i} (M={round(avg[ix],2)}; SD={round(std[ix],2)}px)" for ix, i in enumerate(
                    ["x", "y"])],
                x_label="samples",
                y_label="position (pixels)",
                add_hline={"pos": avg},
            )

            y_pos = 1.05
            sf_a.suptitle(
                f"eye-{eye}",
                fontsize=24,
                y=y_pos, fontweight="bold"
            )

            # plt.tight_layout()

        if isinstance(fname, str):
            fname += f"-eye_qa.{self.save_ext}"
            utils.verbose(f" Writing {bm}{fname}{end}", self.verbose)
            fig.savefig(
                fname,
                bbox_inches='tight',
                dpi=300
            )

    def vols(self, func_file):
        if func_file.endswith("gz") or func_file.endswith('nii'):
            img = nb.load(func_file)
            nr_vols = img.get_fdata().shape[-1]
            self.TR2 = img.header['pixdim'][4]
        elif func_file.endswith("mat"):
            raw = io.loadmat(func_file)
            tag = list(raw.keys())[-1]
            raw = raw[tag]
            nr_vols = raw.shape[-1]
        else:
            raise ValueError(
                f"Could not derive number of volumes for file '{func_file}'")

        return nr_vols


class ParseExpToolsFile(ParseEyetrackerFile, SetAttributes):

    """ParseExpToolsFile()

    Class for parsing tsv-files created during experiments with Exptools2. The class will read in the file, read when the
    experiment actually started, correct onset times for this start time and time deleted because of removing the first few
    volumes (to do this correctly, set the `TR` and `deleted_first_timepoints`). You can also provide a numpy array/file
    containing eye blinks that should be added to the onset times in real-world time (seconds). In principle, it will return a
    pandas DataFrame indexed by subject and run that can be easily concatenated over runs. This function relies on the naming
    used when programming the experiment. In the `session.py` file, you should have created `phase_names=['iti', 'stim']`; the
    class will use these things to parse the file.

    Parameters
    ----------
    tsv_file: str, list
        path pointing to the output file of the experiment
    subject: int
        subject number in the returned pandas DataFrame (should start with 1, ..., n)
    run: int
        run number you'd like to have the onset times for
    button: bool
        boolean whether to include onset times of button responses (default is false). ['space'] will be ignored as response
    TR: float
        repetition time to correct onset times for deleted volumes
    deleted_first_timepoints: int
        number of volumes to delete to correct onset times for deleted volumes. Can be specified for each individual run if
        `tsv_file` is a list
    use_bids: bool, optional
        If true, we'll read BIDS-components such as 'sub', 'run', 'task', etc from the input file and use those as indexers,
        rather than sequential 1,2,3.
    funcs: str, list, optional
        List of functional files that is being passed down down to :class:`lazyfmri.dataset.ParseEyetrackerFile`. Required
        for correct resampling to functional space
    edfs: str, list, optional
        List of eyetracking output files that is being passed down down to :class:`lazyfmri.dataset.ParseEyetrackerFile`.
    verbose: bool, optional
        Print details to the terminal, default is False
    phase_onset: int, optional
        Which phase of exptools-trial should be considered the actual stimulus trial. Usually, `phase_onset=0` means the
        interstimulus interval. Therefore, default = 1
    stim_duration: str, int, optional
        If desired, add stimulus duration to onset dataframe. Can be one of 'None', 'stim' (to use duration from exptools' log
        file) or any given integer
    add_events: str, list, optional
        Add additional events to onset dataframe. Must be an existing column in the exptools log file. For intance,
        `responses` and `event_type = stim` are read in by default, but if we have a separate column containing the onset of
        some target (e.g., 'target_onset'), we can add these times to the dataframe with `add_events='target_onset'`.
    event_names: str, list, optional
        Custom names for manually added events through `add_events` if the column names are not the names you want to use in
        the dataframe. E.g., if I find `target_onset` too long of a name, I can specify `event_names='target'`. If
        `add_events` is a list, then `event_names` must be a list of equal length if custom names are desired. By default
        we'll take the names from `add_events`
    RTs: bool, optional
        If we have a design that required some response to a stimulus, we can request the reaction times. Default = False
    RT_relative_to: str, optional
        If `RTs=True`, we need to know relative to what time the button response should be offset. Only correct responses are
        considered, as there's a conditional statement that requires the present of the reference time (e.g., `target_onset`)
        and button response. If there's a response but no reference time, the reaction time cannot be calculated. If you do
        not have a separate reference time column, you can specify `RT_relative_to='start'` to calculate the reaction time
        relative to onset time. If `RT_relative_to != 'start'`, I'll assume you had a target in your experiment in X/n_trials.
        From this, we can calculate the accuracy and save that to `self.df_accuracy`, while reaction times are saved in
        `self.df_rts`
    button_duration: float, int, optional
        Set duration for button event
    response_window: float, int, optional
        Set window in which a response is counted
    merge: bool, optional
        Merge the dataframes containing responses and stimulus onsets. Default = True. Onset times will be sorted in an
        ascending order. Select parts of dataframes with :func:`lazyfmri.utils.select_from_df()`
    resp_as_cov: bool, optional
        If you have a design where you have button presses in half of the trials, you can add a covariate consisting of -1/1
        for the stimulus onsets where there was a response or not. This way, the response event will not steal variance from
        the stimulus event when fitting using :class:`lazyfmri.fitting.NideconvFitter()`. Default = False
    ev_onset: str, optional
        Sometimes experiments are coded such that the event name of interest is not called `stim`. Use this variable +
        `phase_onset` to extract the correct onset times
    key_press: list, optional
        Specify a custom list of button presses to extract. Default = `['b']`
    expr: str, tuple, optional
        Add additional conditions for your extracted onset times. This follows the formulation of
        :func:`lazyfmri.utils.select_from_df()`. E.g., `expr="movie_type != blank`. Acts as an extra filter to extract the
        relevant onsets
    filter_na: bool, optional
        Try to filter out NaN-events from the onset dataframe

    Examples
    ----------
    .. code-block:: python

        from lazyfmri.dataset import ParseExpToolsFile
        file = 'some/path/to/exptoolsfile.tsv'
        parsed_file = ParseExpToolsFile(file, subject=1, run=1, button=True)
        onsets = parsed_file.get_onset_df()

    """

    def __init__(
        self,
        tsv_file,
        subject=1,
        run=1,
        task=None,
        button=False,
        RTs=False,
        RT_relative_to=None,
        TR=0.105,
        deleted_first_timepoints=0,
        edfs=None,
        funcs=None,
        use_bids=True,
        verbose=False,
        phase_onset=1,
        stim_duration=None,
        add_events=None,
        add_cols=None,
        event_names=None,
        invoked_from_func=False,
        button_duration=1,
        response_window=3,
        merge=True,
        resp_as_cov=False,
        cov_amplitude=1,
        ev_onset="stim",
        duration_col="duration",
        key_press=["b"],
        expr=None,
        filter_na=False,
        **kwargs
        ):

        self.tsv_file = tsv_file
        self.sub = subject
        self.run = run
        self.task = task
        self.TR = TR
        self.deleted_first_timepoints = deleted_first_timepoints
        self.button = button
        self.funcs = funcs
        self.edfs = edfs
        self.use_bids = use_bids
        self.verbose = verbose
        self.phase_onset = phase_onset
        self.stim_duration = stim_duration
        self.RTs = RTs
        self.RT_relative_to = RT_relative_to
        self.add_events = add_events
        self.event_names = event_names
        self.invoked_from_func = invoked_from_func
        self.button_duration = button_duration
        self.response_window = response_window
        self.merge = merge
        self.resp_as_cov = resp_as_cov
        self.key_press = key_press
        self.ev_onset = ev_onset
        self.expr = expr
        self.duration_col = duration_col
        self.add_cols = add_cols
        self.cov_amplitude = cov_amplitude
        self.filter_na = filter_na

        # filter kwargs
        tmp_kwargs = filter_kwargs(
            [
                "ref_slice",
                "filter_strat",
            ],
            kwargs
        )

        # set attributes
        SetAttributes.__init__(self)

        # process edf
        self.process_edf_file(**tmp_kwargs)

        # deal with exptools files
        self.process_exptools_files()

    def process_edf_file(self, **kwargs):
        if self.edfs is not None or hasattr(self, "h5_file"):
            ParseEyetrackerFile.__init__(
                self,
                self.edfs,
                subject=self.sub,
                func_file=self.funcs,
                TR=self.TR,
                use_bids=self.use_bids,
                verbose=self.verbose,
                invoked_from_func=self.invoked_from_func,
                **kwargs
            )

    def process_exptools_files(self):

        utils.verbose("\nEXPTOOLS", self.verbose)

        if isinstance(self.tsv_file, str):
            self.tsv_file = [self.tsv_file]

        if isinstance(self.tsv_file, list):

            self.onset_index = ['subject', 'run', 'event_type']
            self._index = ['subject', 'run']
            self.index_task = False
            self.task_ids = []
            if self.use_bids:
                self.task_ids = utils.get_ids(self.tsv_file, bids="task")
            else:
                if isinstance(self.task, str):
                    self.task_ids = [self.task]

            # insert task id in indexer
            if len(self.task_ids) > 0:
                self.index_task = True
                for idx in ["onset_index", "_index"]:
                    getattr(self, idx).insert(1, "task")

            # set them for internal reference
            for attr in self.exp_attributes:
                setattr(self, attr, [])

            for run, onset_file in enumerate(self.tsv_file):

                self.run = run+1
                self.ses = None
                if self.use_bids:
                    bids_comps = utils.split_bids_components(onset_file)
                    for el in ['sub', 'run', 'ses', 'task']:
                        if el in list(bids_comps.keys()):
                            setattr(self, el, bids_comps[el])

                # check if we got different nr of vols to delete per run
                delete_vols = check_input_is_list(
                    self,
                    "deleted_first_timepoints",
                    list_element=run,
                    matcher="tsv_file")

                # check if we got different stimulus durations per run
                duration = check_input_is_list(
                    self,
                    var="stim_duration",
                    list_element=run,
                    matcher="tsv_file"
                )

                # read in the exptools-file
                self.preprocess_exptools_file(
                    onset_file,
                    run=self.run,
                    task=self.task,
                    delete_vols=delete_vols,
                    phase_onset=self.phase_onset,
                    duration=duration
                )

                # append to df
                self.df_onsets.append(self.get_onset_df(index=False))

                # check if we got RTs
                try:
                    self.df_rts.append(self.get_rts_df(index=False))
                except Exception:
                    pass

                # check if we got accuracy (only if RT_relative_to != 'start')
                try:
                    self.df_accuracy.append(self.get_accuracy(index=False))
                except Exception:
                    pass

                # check if we got responses (only if button == False)
                try:
                    self.df_responses.append(self.get_responses(index=False))
                except Exception:
                    pass

            # concatemate df
            self.df_onsets = pd.concat(
                self.df_onsets).set_index(self.onset_index)

            # filter for NaNs to be sure
            if self.filter_na:
                self.df_onsets = utils.select_from_df(
                    self.df_onsets, expression="event_type != nan")

            # rts
            try:
                self.df_rts = pd.concat(self.df_rts).set_index(self._index)
            except Exception:
                pass

            # accuracy
            try:
                self.df_accuracy = pd.concat(
                    self.df_accuracy).set_index(self._index)
            except Exception:
                pass

            # accuracy
            try:
                self.df_responses = pd.concat(
                    self.df_responses).set_index(self.onset_index)
            except Exception:
                pass

        # get events per run
        self.evs_per_run = utils.get_unique_ids(
            self.df_onsets, id="event_type")

        # check if we should merge responses with onsets
        if self.merge:
            # keep pure onsets too
            self.df_onsets_pure = self.df_onsets.copy()

            self.concat_list = [self.df_onsets]

            # look for responses
            if len(self.df_responses) > 0:
                self.concat_list += [self.df_responses.copy()]

            # look for blinks
            if hasattr(self, "df_blinks"):
                self.concat_list += [self.df_blinks.copy()]

            # concatenate and sort
            if len(self.concat_list) > 0:
                self.merged = pd.concat(self.concat_list).sort_values(
                    ["subject", "run", "onset"])

                # set merged to new df_onsets
                self.df_onsets = self.merged.copy()

    def events_per_run(self):
        n_runs = np.unique(self.df_onsets.reset_index()['run'].values)
        events = {}
        for run in n_runs:
            df = utils.select_from_df(
                self.df_onsets, expression=f"run = {run}", index=None)
            events[run] = np.unique(df['event_type'].values)

        return events

    def events_single_run(self, run=1):
        return self.events_per_run[run]

    def preprocess_exptools_file(
            self,
            tsv_file,
            task=None,
            run=1,
            delete_vols=0,
            phase_onset=1,
            duration=None):

        utils.verbose(f"Preprocessing {gb}{tsv_file}{end}", self.verbose)
        with open(tsv_file) as f:
            self.data = pd.read_csv(f, delimiter='\t')

        # trim onsets to first 't'
        delete_time = delete_vols*self.TR

        self.ev_types = utils.get_unique_ids(self.data, id="event_type")
        if "pulse" in self.ev_types:
            self.start_time = float(utils.select_from_df(
                self.data, expression="event_type = pulse").iloc[0].onset)
        else:
            self.start_time = 0

        utils.verbose(f" 1st 't' @{rb}{round(self.start_time,2)}{end}s", self.verbose)

        # select all events after start time with `ev_onset`
        self.trimmed = utils.select_from_df(
            self.data,
            expression=(
                f"event_type = {self.ev_onset}",
                "&",
                f"onset > {self.start_time}"
            )
        )

        # check for more conditions
        if isinstance(self.expr, (tuple, str)):
            utils.verbose(f" Adding filters: {rb}{self.expr}{end}", self.verbose)
            self.trimmed = utils.select_from_df(
                self.trimmed, expression=self.expr)

        # get the correct phase
        self.trimmed = utils.select_from_df(
            self.trimmed,
            expression=f"phase = {phase_onset}"
        )
        rm_cols = ["index", "level_0"]
        for col in rm_cols:
            if col in list(self.trimmed.columns):
                self.trimmed.drop([col], axis=1, inplace=True)

        # get onset times
        self.onset_times = self.trimmed['onset'].values[..., np.newaxis]
        self.n_trials = np.unique(self.trimmed["onset"].values).shape[0]

        skip_duration = False
        if isinstance(duration, (float, int)):
            self.durations = np.full_like(self.onset_times, float(duration))
        elif duration is None:
            skip_duration = True
        else:
            self.durations = self.trimmed[self.duration_col].values[..., np.newaxis]

        try:
            self.condition = self.trimmed['condition'].values[..., np.newaxis]
        except Exception:
            self.condition = np.full_like(
                self.onset_times, self.ev_onset, dtype=object)
            if self.condition.ndim < 2:
                self.condition = self.condition[..., np.newaxis]

        # get dataframe with responses
        if "response" in np.unique(self.data["event_type"]):
            self.response_df = self.data.loc[(self.data['event_type'] == "response") & (
                self.data['response'] != 'space')]

            if len(self.response_df) > 0:
                # filter out button responses before first trigger
                self.response_df = utils.select_from_df(
                    self.response_df, expression=f"onset > {self.start_time}")

                self.nr_resp = self.response_df.shape[0]
                utils.verbose(
                    f" Extracting {rb}{self.key_press}{end} button(s)", self.verbose)

                # loop through them
                self.button_df = []

                self.single_button = True
                if len(self.key_press) > 1:
                    self.single_button = False

                for button in self.key_press:

                    # get the onset times
                    self.response_times = self.response_df['onset'].values[..., np.newaxis]

                    # store responses in separate dataframe
                    self.response_times -= (self.start_time + delete_time)

                    # decide name
                    if self.single_button:
                        ev_name = "response"
                    else:
                        ev_name = button

                    # make into dataframe
                    tmp = pd.DataFrame(self.response_times, columns=["onset"])
                    tmp["event_type"] = ev_name

                    self.tmp_df = self.index_onset(
                        tmp,
                        task=task,
                        subject=self.sub,
                        run=run)

                    self.button_df.append(self.tmp_df)

                if len(self.button_df) > 0:
                    self.button_df = pd.concat(self.button_df)
            else:
                utils.verbose(" No button presses found", self.verbose)

        # check if we should include other events
        if isinstance(self.add_events, str):
            self.add_events = [self.add_events]

        if isinstance(self.event_names, str):
            self.event_names = [self.event_names]

        if isinstance(self.add_events, list):
            if isinstance(self.event_names, list):
                if len(self.event_names) != len(self.add_events):
                    raise ValueError(
                        f"""Length ({rb}{len(self.add_events)}{end}) of added events {rb}{self.add_events}{end} does not equal the length
                        ({rb}{len(self.event_names)}{end}) of requested event names {rb}{self.event_names}{end}""")
            else:
                self.event_names = self.add_events.copy()

            for ix, ev in enumerate(self.add_events):
                ev_times = np.array(
                    [ii for ii in np.unique(self.data[ev].values)])

                # filter for nan (https://stackoverflow.com/a/11620982)
                ev_times = ev_times[~np.isnan(ev_times)][..., np.newaxis]

                # create condition
                ev_names = np.full(ev_times.shape, self.event_names[ix])

                # add times and names to array
                self.onset_times = np.vstack((self.onset_times, ev_times))
                self.condition = np.vstack((self.condition, ev_names))

        # check if we should add duration (can't be used in combination with add_events)
        if not skip_duration:
            if isinstance(self.add_events, list):
                raise TypeError(
                    f"""Cannot do this operation because I don't know the durations for the added events. Please consider using
                    'stim_duration!={self.stim_duration}' or 'add_events=None'""")
            self.onset = np.hstack(
                (self.onset_times, self.condition, self.durations))
        else:
            self.onset = np.hstack((self.onset_times, self.condition))

        if isinstance(self.add_cols, (str, list)):
            if isinstance(self.add_cols, str):
                self.add_cols = [self.add_cols]

            utils.verbose(f" Adding {rb}{self.add_cols}{end} to onsets", self.verbose)
            add_cols = []
            for i in self.add_cols:
                add_cols.append(self.trimmed[i].values[..., np.newaxis])
            add_cols = np.hstack(add_cols)
            self.onset = np.hstack([self.onset, add_cols])

        # sort array based on onset times (https://stackoverflow.com/a/2828121)
        self.onset = self.onset[self.onset[:, 0].argsort()]

        # correct for start time of experiment and deleted time due to removal of inital volumes
        self.onset[:, 0] = self.onset[:, 0]-(self.start_time + delete_time)

        utils.verbose(
            f" Cutting {rb}{round(self.start_time + delete_time,2)}{end}s from onsets", self.verbose)
        if not skip_duration:
            utils.verbose(
                f" Avg duration = {rb}{round(self.durations.mean(),2)}{end}s", self.verbose)

        # make dataframe
        columns = ['onset', 'event_type']
        if not skip_duration:
            columns += ['duration']

        if isinstance(self.add_cols, list):
            columns += self.add_cols

        # check if we should do reaction times
        if self.RTs:
            if not isinstance(self.RT_relative_to, str):
                raise ValueError(
                    f"Need a reference column to calculate reaction times (RTs), not '{self.RT_relative_to}'")

            # get response dataframe
            self.response_df = self.data.loc[(self.data['event_type'] == "response") & (
                self.data['response'] != 'space')]

            # get target dataframe
            self.target_df = self.data.loc[~pd.isnull(self.data.target_onset)]

            # now we need to cross references them
            self.n_hits = 0
            self.n_miss = 0
            self.n_fa = 0
            self.n_cr = 0

            self.rts = []
            self.unique_targets = np.unique(self.target_df["trial_nr"].values)
            self.n_targets = self.unique_targets.shape[0]
            self.n_responses = len(
                np.unique(self.response_df["trial_nr"].values))

            for trial in self.unique_targets:

                # get target onset
                trial_targ = self.target_df.loc[(
                    self.target_df["trial_nr"] == trial)]
                targ_onset = trial_targ["target_onset"].values[0]

                # check if there's a response within window regardless of whether response occured in trial ID
                resp = self.response_df.query(
                    f"{targ_onset} < onset < {targ_onset+self.response_window}")

                if len(resp) > 0:
                    # found response
                    resp_time = resp["onset"].values[0]

                    if resp_time > targ_onset:
                        rt = resp_time-targ_onset
                        self.rts.append(rt)

                    self.n_hits += 1

            if len(self.rts) >= 1:
                self.rts = np.array(self.rts)
            else:
                self.rts = np.array([0])

            if self.n_hits > 0:
                self.hits = self.n_hits/self.n_targets
                self.n_miss = self.n_targets-self.n_hits
            else:
                self.n_miss = self.n_targets

            # track false alarms
            self.unique_responses = np.unique(
                self.response_df["trial_nr"].values)

            # track false alarms
            if self.n_responses > self.n_targets:
                self.n_fa = self.n_responses-self.n_targets
            else:
                self.n_fa = 0

            # track correct rejections
            if self.n_fa > 0:
                self.n_cr = self.n_targets-self.n_fa
            else:
                self.n_cr = int(self.n_trials-self.n_targets)

            # calculate d-prime
            #   d-prime=0 is considered as pure guessing.
            #   d-prime=1 is considered as good measure of signal sensitivity/detectability.
            #   d-prime=2 is considered as awesome.
            self.sdt_ = utils.SDT(
                self.n_hits,
                self.n_miss,
                self.n_fa,
                self.n_cr)

            if hasattr(self, 'sdt_'):
                utils.verbose(
                    f" Hits:\t{round(self.sdt_['hit'],2)}\t({self.n_hits}/{self.n_targets})", self.verbose)
                utils.verbose(
                    f" FA:\t{round(self.sdt_['fa'],2)}\t({self.n_fa}/{self.n_targets})", self.verbose)
                utils.verbose(
                    f" D':\t{round(self.sdt_['d'],2)}\t(0=guessing;1=good;2=awesome)", self.verbose)
                utils.verbose(
                    f" Average reaction time (RT) = {round(self.rts.mean(),2)}s (relative to '{self.RT_relative_to}').",
                    self.verbose)

            # parse into dataframe
            self.accuracy_df = self.index_accuracy(
                self.sdt_, subject=self.sub, run=run)
            self.rts_df = self.array_to_df(
                self.rts,
                columns=["RTs"],
                subject=self.sub,
                run=run,
                key="RTs")

            # keep track of response during trial
            self.response_during_trial = np.full(
                (self.n_trials), -self.cov_amplitude)
            self.cov_times = []
            for stim in range(self.n_trials):

                # get onset info
                self.trial_onset = self.trimmed.iloc[stim]

                # set default response time to stimulus onset
                append_time = self.trial_onset.onset

                # look if there's an actual response to stimulus
                trial = self.trial_onset.trial_nr
                if trial in self.unique_targets:
                    trial_targ = self.target_df.loc[(
                        self.target_df["trial_nr"] == trial)]
                    targ_onset = trial_targ[self.RT_relative_to].values[0]

                    # check if there's a response within window regardless of whether response occured in trial ID
                    resp = self.response_df.query(
                        f"{targ_onset} < onset < {targ_onset+self.response_window}")

                    # response during trial
                    if len(resp) > 0:
                        resp_time = resp["onset"].values[0]
                        self.response_during_trial[stim] = self.cov_amplitude
                        append_time = resp_time

                self.cov_times.append(append_time)

            # for stimuli without responses, we'll create a dummy onset time consisting of the response time of the closest
            # button press DURING stimulus relative to the onset of stimulus
            for stim in range(self.n_trials):

                # we have -1 for stimuli without response time
                if self.response_during_trial[stim] < 0:

                    # find closest trial with button press
                    closest_trial = utils.find_nearest(
                        self.response_during_trial[stim:], self.cov_amplitude)[0]
                    closest_trial += stim

                    # get reaction time of this stim
                    trial_onset = self.trimmed.iloc[closest_trial].onset
                    resp_onset = self.cov_times[closest_trial]
                    rt = resp_onset-trial_onset

                    # deal with button presses BEFORE stimulus onset
                    if rt == 0:
                        # invert array for last element
                        closest_trial = utils.find_nearest(
                            self.response_during_trial[::-1][-stim:], self.cov_amplitude)[0]
                        closest_trial = (stim-closest_trial)-1

                        # get reaction time of this stim
                        trial_onset = self.trimmed.iloc[closest_trial].onset
                        resp_onset = self.cov_times[closest_trial]
                        rt = resp_onset-trial_onset

                    if rt == 0:
                        raise ValueError("RT=0, something odd's going on here")

                    # shift mock response onset with this RT
                    # print(f"Adding {rt} to trial #{stim+1}")
                    self.cov_times[stim] += rt

            # convert list to array and correct for start time and deleted volumes
            self.cov_times = np.array(self.cov_times)
            self.cov_times -= (self.start_time + delete_time)

            # create dataframe
            self.cov_df = pd.DataFrame(
                {
                    "onset": self.cov_times,
                    "cov": self.response_during_trial
                }
            )

            for key, val in zip(
                ["event_type", "subject", "run"],
                    ["response", self.sub, run]):
                self.cov_df[key] = val

        # inset onsets
        self.onset_df = self.index_onset(
            self.onset,
            task=task,
            columns=columns,
            subject=self.sub,
            run=run)

        # add response covariate column  THIS IS A SHORTCUT FOR NOW!
        if self.resp_as_cov:
            # self.response_during_trial
            self.onset_df["cov"] = self.cov_amplitude

    def index_onset(
            self,
            array,
            columns=None,
            subject=1,
            run=1,
            task=None,
            set_index=False):

        if isinstance(array, dict):
            df = pd.DataFrame(array, index=[0])
        elif isinstance(array, pd.DataFrame):
            df = array.copy()
        else:
            df = pd.DataFrame(array, columns=columns)

        df['subject'] = subject
        df['run'] = run
        df['event_type'] = df['event_type'].astype(str)
        df['onset'] = df['onset'].astype(float)

        if self.index_task:
            if isinstance(task, str):
                df["task"] = task
            else:
                df["task"] = "task"

        # check if we got duration
        try:
            df['duration'] = df['duration'].astype(float)
        except Exception:
            pass

        if set_index:
            return df.set_index(self.onset_index)
        else:
            return df

    @staticmethod
    def array_to_df(
            array,
            columns=None,
            subject=1,
            key="RTs",
            run=1,
            set_index=False):

        if isinstance(array, dict):
            df = pd.DataFrame(array)
        else:
            df = pd.DataFrame(array, columns=columns)

        df['subject'] = subject
        df['run'] = run
        df[key] = df[key].astype(float)

        if set_index:
            return df.set_index(['subject', 'run'])
        else:
            return df

    @staticmethod
    def index_accuracy(array, columns=None, subject=1, run=1, set_index=False):

        if isinstance(array, dict):
            df = pd.DataFrame(array, index=[0])
        else:
            df = pd.DataFrame(array, columns=columns)

        df['subject'] = subject
        df['run'] = run

        if set_index:
            return df.set_index(['subject', 'run'])
        else:
            return df

    def get_onset_df(self, index=False):
        """Return the indexed DataFrame containing onset times"""

        if index:
            return self.onset_df.set_index(['subject', 'run', 'event_type'])
        else:
            return self.onset_df

    def get_rts_df(self, index=False):
        """Return the indexed DataFrame containing reaction times"""

        if index:
            return self.rts_df.set_index(['subject', 'run'])
        else:
            return self.rts_df

    def get_accuracy(self, index=False):
        """Return the indexed DataFrame containing reaction times"""

        if index:
            return self.accuracy_df.set_index(['subject', 'run'])
        else:
            return self.accuracy_df

    def get_responses(self, index=False):
        """Return the indexed DataFrame containing reaction times"""

        if self.resp_as_cov:
            ret_df = self.cov_df.copy()
        else:
            ret_df = self.button_df.copy()

        if index:
            return ret_df.set_index(['subject', 'run'])
        else:
            return ret_df

    def onsets_to_fsl(
            self,
            fmt='3-column',
            amplitude=1,
            duration=None,
            output_dir=None,
            output_base=None,
            from_event=True):
        """onsets_to_fsl

        This function creates a text file with a single column containing the onset times of a given condition. Such a file
        can be used for SPM or FSL modeling, but it should be noted that the onset times have been corrected for the deleted
        volumes at the beginning. So make sure your inputting the correct functional data in these cases.

        Parameters
        ----------
        fmt: str
            format for the onset file (default = 3-column format)
        amplitude: int, float
            amplitude for stimulus vector
        duration: int, float
            duration of the event; overwrite possible 'duration' column in onsets
        output_dir: str
            path to output name for text file(s)
        output_base: str
            basename for output file(s); should include full path. '<_task-{task}>_run-{run}_ev-{ev}.txt' will be appended
        from_event: bool
            take the event name as specified in the onset dataframe. By default, this is true. In some cases where your events
            consists of float numbers, it's sometimes easier to number them consecutively. In that case, specify
            from_event=False`

        Returns
        ----------
        str
            for each subject, task, and run, a text file for all events present in the onset dataframe (if only 1 task was
            present, this will be omitted)
        """

        onsets = self.df_onsets.copy()
        subj_list = self.get_subjects(onsets)

        for sub in subj_list:

            # fetch subject specific onsets
            df = utils.select_from_df(onsets, expression=f"subject = {sub}")

            # check if we got task in dataframe
            indices = list(onsets.index.names)
            tasks = []
            separate_task = False
            if self.use_bids:
                if "task" in indices:
                    tasks = utils.get_ids(self.tsv_file)
                    separate_task = True

            # we got task in dataframe
            if separate_task:
                for task in tasks:

                    # get task-specific onsets
                    task_df = utils.select_from_df(
                        df, expression=f"task = {task}")

                    # get runs within task
                    n_runs = self.get_runs(task_df)
                    for run in n_runs:
                        onsets_per_run = utils.select_from_df(
                            task_df, expression=f"run = {run}")
                        events_per_run = self.get_events(onsets_per_run)

                        for ix, ev in enumerate(events_per_run):

                            onsets_per_event = utils.select_from_df(
                                onsets_per_run, expression=f"event_type = {events_per_run[ix]}")

                            if from_event:
                                ev_tag = f"ev-{ev}"
                            else:
                                ev_tag = f"ev-{ix+1}"

                            if output_base is None:
                                if not isinstance(output_dir, str):
                                    if isinstance(self.tsv_file, list):
                                        outdir = os.path.dirname(
                                            self.tsv_file[0])
                                    elif isinstance(self.tsv_file, str):
                                        outdir = os.path.dirname(self.tsv_file)
                                    else:
                                        outdir = os.getcwd()
                                else:
                                    outdir = output_dir

                                # create output directory
                                if not os.path.exists(outdir):
                                    os.makedirs(outdir, exist_ok=True)

                                fname = opj(
                                    outdir, f"task-{task}_run-{run}_{ev_tag}.txt")
                            else:
                                fname = f"{output_base}_task-{task}_run-{run}_{ev_tag}.txt"

                            # fetch the onsets
                            event_onsets = onsets_per_event['onset'].values[..., np.newaxis]
                            if fmt == "3-column":

                                # check if we got duration
                                if 'duration' in list(onsets_per_event.columns):
                                    duration_arr = onsets_per_event['duration'].values[..., np.newaxis]
                                else:
                                    if not isinstance(duration, (int, float)):
                                        duration_arr = np.full_like(
                                            onsets_per_event, duration)
                                    else:
                                        duration_arr = np.ones_like(
                                            onsets_per_event)

                                amplitude_arr = np.full_like(
                                    event_onsets, amplitude)
                                three_col = np.hstack(
                                    (event_onsets, duration_arr, amplitude_arr))

                                print(f"Writing {fname}; {three_col.shape}")
                                np.savetxt(fname, three_col,
                                           delimiter='\t', fmt='%1.3f')
                            else:
                                np.savetxt(fname, event_onsets,
                                           delimiter='\t', fmt='%1.3f')

            else:
                n_runs = self.get_runs(df)
                for run in n_runs:
                    onsets_per_run = utils.select_from_df(
                        df, expression=f"run = {run}")
                    events_per_run = self.get_events(onsets_per_run)

                    for ix, ev in enumerate(events_per_run):

                        onsets_per_event = utils.select_from_df(
                            onsets_per_run, expression=f"event_type = {events_per_run[ix]}")

                        if from_event:
                            ev_tag = f"ev-{ev}"
                        else:
                            ev_tag = f"ev-{ix+1}"

                        if output_base is None:
                            if not isinstance(output_dir, str):
                                if isinstance(self.tsv_file, list):
                                    outdir = os.path.dirname(self.tsv_file[0])
                                elif isinstance(self.tsv_file, str):
                                    outdir = os.path.dirname(self.tsv_file)
                                else:
                                    outdir = os.getcwd()
                            else:
                                outdir = output_dir

                            fname = opj(outdir, f"{ev_tag}_run-{run}.txt")
                        else:
                            fname = f"{output_base}_run-{run}_{ev_tag}.txt"

                        # fetch the onsets
                        event_onsets = onsets_per_event['onset'].values[..., np.newaxis]
                        if fmt == "3-column":

                            # check if we got duration
                            if 'duration' in list(onsets_per_event.columns):
                                duration_arr = onsets_per_event['duration'].values[..., np.newaxis]
                            else:
                                if not isinstance(duration, (int, float)):
                                    duration_arr = np.full_like(
                                        onsets_per_event, duration)
                                else:
                                    duration_arr = np.ones_like(
                                        onsets_per_event)

                            amplitude_arr = np.full_like(
                                event_onsets, amplitude)
                            three_col = np.hstack(
                                (event_onsets, duration_arr, amplitude_arr))

                            print(f"Writing {fname}; {three_col.shape}")
                            np.savetxt(fname, three_col,
                                       delimiter='\t', fmt='%1.3f')
                        else:
                            np.savetxt(fname, event_onsets,
                                       delimiter='\t', fmt='%1.3f')

    @staticmethod
    def get_subjects(df):
        try:
            df = df.reset_index()
        except Exception:
            pass

        return np.unique(df['subject'].values)

    @staticmethod
    def get_runs(df):
        try:
            df = df.reset_index()
        except Exception:
            pass

        return np.unique(df['run'].values)

    @staticmethod
    def get_events(df):
        try:
            df = df.reset_index()
        except Exception:
            pass

        return np.unique(df['event_type'].values)


class ParsePhysioFile():

    """ParsePhysioFile

    In similar style to :class:`lazyfmri.dataset.ParseExpToolsFile` and :class:`lazyfmri.dataset.ParseFuncFile`, we use
    this class to read in physiology-files created with the PhysIO-toolbox (https://www.tnu.ethz.ch/en/software/tapas/
    documentations/physio-toolbox) (via `call_spmphysio` for instance). Using the *.mat*-file created with `PhysIO`, we can
    also attempt to extract `heart rate variability` measures. If this file cannot be found, this operation will be skipped

    Parameters
    ----------
    physio_file: str
        path pointing to the regressor file created with PhysIO (e.g., `call_spmphysio`)
    physio_mat: str
        path pointing to the *.mat*-file created with PhysIO (e.g., `call_spmphysio`)
    subject: int
        subject number in the returned pandas DataFrame (should start with 1, ..., n)
    run: int
        run number you'd like to have the onset times for
    TR: float
        repetition time to correct onset times for deleted volumes
    orders: list
        list of orders used to create the regressor files (see `call_spmphysio`, but default = [2,2,2,]). This one is
        necessary to create the correct column names for the dataframe
    deleted_first_timepoints: int, optional
        number of volumes deleted at the beginning of the timeseries
    deleted_last_timepoints: int, optional
        number of volumes deleted at the end of the timeseries

    Example
    ----------
    .. code-block:: python
    
        physio_file = opj(os.path.dirname(func_file), "sub-001_ses-1_task-SR_run-1_physio.txt")
        physio_mat  = opj(os.path.dirname(func_file), "sub-001_ses-1_task-SR_run-1_physio.mat")
        physio = utils.ParsePhysioFile(
            physio_file,
            physio_mat=physio_mat,
            subject=func.subject,
            run=func.run,
            TR=func.TR,
            deleted_first_timepoints=func.deleted_first_timepoints,
            deleted_last_timepoints=func.deleted_last_timepoints)

        physio_df = physio.get_physio(index=False)

    """

    def __init__(
            self,
            physio_file,
            physio_mat=None,
            subject=1,
            run=1,
            TR=0.105,
            orders=[3, 4, 1],
            deleted_first_timepoints=0,
            deleted_last_timepoints=0,
            use_bids=False,
            verbose=True,
            **kwargs):

        self.physio_file = physio_file
        self.physio_mat = physio_mat
        self.sub = subject
        self.run = run
        self.TR = TR
        self.orders = orders
        self.deleted_first_timepoints = deleted_first_timepoints
        self.deleted_last_timepoints = deleted_last_timepoints
        self.physio_mat = physio_mat
        self.use_bids = use_bids
        self.verbose = verbose
        self.__dict__.update(kwargs)

        utils.verbose("\nPHYSIO", self.verbose)

        self.physio_cols = [f'c_{i}' for i in range(self.orders[0])] + [f'r_{i}' for i in range(
            self.orders[1])] + [f'cr_{i}' for i in range(self.orders[2])]

        if isinstance(self.physio_file, str):
            self.physio_file = [self.physio_file]

        if isinstance(self.physio_mat, str):
            self.physio_mat = [self.physio_mat]

        if isinstance(self.physio_file, list):

            df_physio = []
            for run, func in enumerate(self.physio_file):

                utils.verbose(f"Preprocessing {gb}{func}{end}", self.verbose)

                self.run = run+1
                self.ses = None
                self.task = None
                if self.use_bids:
                    bids_comps = utils.split_bids_components(func)
                    for el in ['sub', 'run', 'ses', 'task']:
                        if el in list(bids_comps.keys()):
                            setattr(self, el, bids_comps[el])

                # check if deleted_first_timepoints is list or not
                delete_first = check_input_is_list(
                    self,
                    var="deleted_first_timepoints",
                    list_element=run,
                    matcher="func_file")

                # check if deleted_last_timepoints is list or not
                delete_last = check_input_is_list(
                    self,
                    var="deleted_last_timepoints",
                    list_element=run,
                    matcher="func_file")

                if self.physio_mat is not None:
                    if isinstance(self.physio_mat, list):
                        if len(self.physio_mat) == len(self.physio_file):
                            mat_file = self.physio_mat[run]
                        else:
                            raise ValueError(
                                f"""Length of mat-files ({len(self.physio_mat)}) does not match length of physio-files
                                ({len(self.physio_mat)})""")
                    else:
                        raise ValueError(
                            "Please specify a list of mat-files of equal lengths to that of the list of physio files")
                else:
                    mat_file = None

                self.preprocess_physio_file(
                    func,
                    physio_mat=mat_file,
                    deleted_first_timepoints=delete_first,
                    deleted_last_timepoints=delete_last)

                df_physio.append(self.get_physio(index=False))

            self.df_physio = pd.concat(df_physio).set_index(
                ['subject', 'run', 't'])

    def preprocess_physio_file(
            self,
            physio_tsv,
            physio_mat=None,
            deleted_first_timepoints=0,
            deleted_last_timepoints=0):

        self.physio_data = pd.read_csv(
            physio_tsv,
            header=None,
            sep="\t",
            engine='python',
            skiprows=deleted_first_timepoints,
            usecols=list(range(0, len(self.physio_cols))))

        self.physio_df = pd.DataFrame(self.physio_data)
        self.physio_df.drop(self.physio_df.tail(
            deleted_last_timepoints).index, inplace=True)
        self.physio_df.columns = self.physio_cols

        # Try to get the heart rate
        if physio_mat is not None:

            self.mat = io.loadmat(physio_mat)
            try:
                self.hr = self.mat['physio']['ons_secs'][0][0][0][0][12]
            except Exception:
                print(" WARNING: no heart rate trace found..")

            try:
                self.rvt = self.mat['physio']['ons_secs'][0][0][0][0][13]
            except Exception:
                print(" WARNING: no respiration trace found..")

            # trim beginning and end
            for trace in ['hr', 'rvt']:
                if hasattr(self, trace):
                    if deleted_last_timepoints != 0:
                        self.physio_df[trace] = getattr(
                            self, trace)[deleted_first_timepoints:-deleted_last_timepoints, :]
                    else:
                        self.physio_df[trace] = getattr(
                            self, trace)[deleted_first_timepoints:, :]

        self.physio_df['subject'], self.physio_df['run'], self.physio_df['t'] = self.sub, self.run, list(
            self.TR*np.arange(self.physio_df.shape[0]))

    def get_physio(self, index=True):
        if index:
            return self.physio_df.set_index(['subject', 'run', 't'])
        else:
            return self.physio_df


class ParseFuncFile(ParseExpToolsFile, ParsePhysioFile):

    """ParseFuncFile

    Class for parsing func-files created with Luisa's reconstruction. It can do filtering, conversion to percent signal
    change, and create power spectra. It is supposed to look similar to :class:`lazyfmri.dataset.ParseExpToolsFile` to make
    it easy to translate between the functional data and experimental data.

    Parameters
    ----------
    func_file: str, list
        path or list of paths pointing to the output file of the experiment
    subject: int, optional
        subject number in the returned pandas DataFrame (should start with 1, ..., n)
    run: int, optional
        run number you'd like to have the onset times for
    baseline: float, int, optional
        Duration of the baseline used to calculate the percent-signal change. This method is the default over `psc_nilearn`
    baseline_units: str, optional
        Units of the baseline. Use `seconds`, `sec`, or `s` to imply that `baseline` is in seconds. We'll convert it to
        volumes internally. If `deleted_first_timepoints` is specified, `baseline` will be corrected for that as well.
    psc_nilearn: bool, optional
        Use nilearn method of calculating percent signal change. This method uses the mean of the entire timecourse, rather
        than the baseline period. Overwrites `baseline` and `baseline_units`. Default is False.
    standardize: str, optional
        method of standardization (e.g., "zscore" or "psc")
    low_pass: bool, optional
        Temporally smooth the data. It's a bit of a shame if this is needed
    lb: float, optional
        lower bound for signal filtering
    TR: float, optional
        repetition time to correct onset times for deleted volumes
    deleted_first_timepoints: int, list, optional
        number of volumes deleted at the beginning of the timeseries. Can be specified for each individual run if `func_file`
        is a list
    deleted_last_timepoints: int, list, optional
        number of volumes deleted at the end of the timeseries. Can be specified for each individual run if `func_file` is a
        list
    window_size: int, optional
        size of window for rolling median and Savitsky-Golay filter
    poly_order: int, optional
        The order of the polynomial used to fit the samples. polyorder must be less than window_length.
    use_bids: bool, optional
        If true, we'll read BIDS-components such as 'sub', 'run', 'task', etc from the input file and use those as indexers,
        rather than sequential 1,2,3.
    verbose: bool, optional
        Print details to the terminal, default is False
    retroicor: bool, optional
        WIP: implementation of retroicor, requires the specification of `phys_file` and `phys_mat` containing the output from
        the PhysIO-toolbox
     n_components: int, optional
        Number of components to use for WM/CSF PCA during ICA
    select_component: int, optional
        If `verbose=True` and `ICA=True`, we'll create a scree-plot of the PCA components. With this flag, you can re-run this
        call but regress out only this particular component. [Deprecated: `filter_confs` is much more effective]
    filter_confs: float, optional
        High-pass filter the components from the components during ICA. This seems to be pretty effective. Default is
        0.2Hz.
    save_as: str, optional
        Directory + basename for several figures that can be created during the process
    transpose: bool, optional
        The data needs to be in the format of <time,voxels>. We'll be trying to force the input data into this format, but
        sometimes this breaks. This flag serves as an opportunity to flip whatever the default is for a particular input file
        (e.g., `gii`, `npy`, or `np.ndarray`), so that your final dataframe has the format it needs to have. For gifti-input,
        we transpose by default. `transpose=True` turns this transposing *off*. For `npy`-inputs, we do **NOT** transpose (we
        assume the numpy arrays are already in <time,voxels> format). `transpose=True` will transpose this input.
    report: bool:
        Save a bunch of figures along the process, including:
            - Eyetracking fidelity
            - ICA components
            - tSNR before/after ICA

        Directory used is <save_as>/figures + basename that is derived from BIDS components.

    Example
    ----------
    .. code-block:: python

        from lazyfmri import utils, dataset
        func_file = utils.get_file_from_substring(
            f"run-1_bold.mat",
            opj('sub-001', 'ses-1', 'func')
        )
        func = utils.ParseFuncFile(
            func_file,
            subject=1,
            run=1,
            deleted_first_timepoints=100,
            deleted_last_timepoints=300
        )

        raw = func.get_raw(index=True)
        psc = func.get_psc(index=True)

    """

    def __init__(
        self,
        func_file,
        subject=1,
        run=1,
        filter_strategy="hp",
        TR=0.105,
        lb=0.01,
        deleted_first_timepoints=0,
        deleted_last_timepoints=0,
        window_size=11,
        poly_order=3,
        attribute_tag=None,
        hdf_key="df",
        tsv_file=None,
        edf_file=None,
        phys_file=None,
        phys_mat=None,
        use_bids=True,
        button=False,
        verbose=True,
        retroicor=False,
        ica=False,
        n_components=5,
        func_tag=None,
        select_component=None,
        standardization="psc",
        filter_confs=0.2,
        keep_comps=None,
        ses1_2_ls=None,
        run_2_run=None,
        save_as=None,
        gm_range=[355, 375],
        tissue_thresholds=[0.7, 0.7, 0.7],
        save_ext="svg",
        transpose=False,
        baseline=20,
        baseline_units="seconds",
        psc_nilearn=False,
        foldover="FH",
        report=False,
        **kwargs
        ):

        self.sub = subject
        self.run = run
        self.TR = TR
        self.lb = lb
        self.deleted_first_timepoints = deleted_first_timepoints
        self.deleted_last_timepoints = deleted_last_timepoints
        self.window_size = window_size
        self.poly_order = poly_order
        self.attribute_tag = attribute_tag
        self.hdf_key = hdf_key
        self.button = button
        self.func_file = func_file
        self.tsv_file = tsv_file
        self.edf_file = edf_file
        self.phys_file = phys_file
        self.phys_mat = phys_mat
        self.use_bids = use_bids
        self.verbose = verbose
        self.retroicor = retroicor
        self.foldover = foldover
        self.func_tag = func_tag
        self.n_components = n_components
        self.select_component = select_component
        self.filter_confs = filter_confs
        self.standardization = standardization
        self.ses1_2_ls = ses1_2_ls
        self.run_2_run = run_2_run
        self.save_as = save_as
        self.gm_range = gm_range
        self.tissue_thresholds = tissue_thresholds
        self.save_ext = save_ext
        self.filter_strategy = filter_strategy
        self.transpose = transpose
        self.baseline = baseline
        self.baseline_units = baseline_units
        self.psc_nilearn = psc_nilearn
        self.ica = ica
        self.keep_comps = keep_comps
        self.report = report
        self.__dict__.update(kwargs)

        # sampling rate and nyquist freq
        self.fs = 1/self.TR
        self.fn = self.fs/2

        if self.phys_file is not None:

            ParsePhysioFile.__init__(
                self,
                self.phys_file,
                physio_mat=self.phys_mat,
                use_bids=self.use_bids,
                TR=self.TR,
                deleted_first_timepoints=self.deleted_first_timepoints,
                deleted_last_timepoints=self.deleted_last_timepoints,
                **kwargs
            )

        utils.verbose("\nFUNCTIONAL", self.verbose)
        if isinstance(self.func_file, (str, np.ndarray)):
            self.func_file = [self.func_file]

        # store settings
        self.func_settings = {}
        self.incl_settings = [
            "TR",
            "lb",
            "deleted_first_timepoints",
            "deleted_last_timepoints",
            "window_size",
            "poly_order",
            "button",
            "func_file",
            "tsv_file",
            "edf_file",
            "verbose",
            "retroicor",
            "foldover",
            "func_tag",
            "n_components",
            "select_component",
            "filter_confs",
            "standardization",
            "ses1_2_ls",
            "run_2_run",
            "save_as",
            "gm_range",
            "tissue_thresholds",
            "filter_strategy",
            "baseline",
            "baseline_units",
            "psc_nilearn",
            "ica",
            "keep_comps"
        ]

        for key in self.incl_settings:
            self.func_settings[key] = getattr(self, key)

        # check if we should index task
        self.index_task = False
        self.index_list = ["subject", "run", "t"]
        if self.use_bids:
            self.task_ids = utils.get_ids(self.func_file, bids="task")

            if len(self.task_ids) > 1:
                self.index_task = True
                self.index_list = ["subject", "task", "run", "t"]

        if isinstance(self.func_file, list):

            # initiate some dataframes
            self.df_psc = []    # psc-data (filtered or not)
            self.df_raw = []    # raw-data (filtered or not)
            # z-score data (retroicor'ed, `if retroicor=True`)
            self.df_retro = []
            # r2 for portions of retroicor-regressors (e.g., 'all', 'cardiac', etc)
            self.df_r2 = []
            self.df_zscore = []    # zscore-d data
            self.df_ica = []    # ica'ed data
            self.ica_objs = []    # keep track of all ICA objects

            # reports
            for run_id, func in enumerate(self.func_file):

                if isinstance(func, str):
                    utils.verbose(f"Preprocessing {gb}{func}{end}", self.verbose)
                elif isinstance(func, np.ndarray):
                    utils.verbose(
                        f"Preprocessing {gb}array {run_id+1}{end} in list", self.verbose)

                    # override use_bids. Can't be use with numpy arrays
                    self.use_bids = False
                else:
                    raise ValueError(
                        f"Unknown input type '{type(func)}'. Must be string or numpy-array")

                self.run = run_id+1
                self.ses = None
                self.task = None
                if self.use_bids:
                    if isinstance(func, str):
                        bids_comps = utils.split_bids_components(func)
                        for el in ['sub', 'run', 'ses', 'task']:
                            if el in list(bids_comps.keys()):
                                setattr(self, el, bids_comps[el])

                # set base name based on presence of bids tags
                self.base_name = f"sub-{self.sub}"
                if isinstance(self.ses, (str, float, int)):
                    self.base_name += f"_ses-{self.ses}"

                if isinstance(self.task, str):
                    self.base_name += f"_task-{self.task}"

                # check if deleted_first_timepoints is list or not
                delete_first = check_input_is_list(
                    self,
                    var="deleted_first_timepoints",
                    list_element=run_id,
                    matcher="func_file")

                # check if deleted_last_timepoints is list or not
                delete_last = check_input_is_list(
                    self,
                    var="deleted_last_timepoints",
                    list_element=run_id,
                    matcher="func_file")

                # check if baseline is list or not
                baseline = check_input_is_list(
                    self,
                    var="baseline",
                    list_element=run_id,
                    matcher="func_file")

                utils.verbose(
                    f" Filtering strategy: '{rb}{self.filter_strategy}{end}'", self.verbose)
                utils.verbose(
                    f" Standardization strategy: '{rb}{self.standardization}{end}'", self.verbose)

                self.preprocess_func_file(
                    func,
                    run=self.run,
                    task=self.task,
                    deleted_first_timepoints=delete_first,
                    deleted_last_timepoints=delete_last,
                    baseline=baseline,
                    **kwargs
                )

                if self.standardization == "psc":
                    self.df_psc.append(
                        self.get_data(
                            index=False,
                            filter_strategy=self.filter_strategy,
                            dtype='psc',
                            ica=self.ica
                        )
                    )

                elif self.standardization == "zscore":
                    self.df_zscore.append(
                        self.get_data(
                            index=False,
                            filter_strategy=self.filter_strategy,
                            dtype='zscore',
                            ica=False
                        )
                    )

                self.df_raw.append(
                    self.get_data(
                        index=False,
                        filter_strategy=None,
                        dtype='raw'))

                if self.retroicor:
                    self.df_retro.append(self.get_retroicor(index=False))
                    self.df_r2.append(self.r2_physio_df)

                if self.ica:
                    self.df_ica.append(
                        self.get_data(
                            index=False,
                            filter_strategy=self.filter_strategy,
                            dtype=self.standardization,
                            ica=True
                        )
                    )

                    self.ica_objs.append(self.ica_obj)

            # check for standardization method
            if self.standardization == "psc":
                self.df_func_psc = pd.concat(self.df_psc)
            elif self.standardization == "zscore":
                self.df_func_zscore = pd.concat(self.df_zscore)

            # we'll always have raw data
            self.df_func_raw = pd.concat(self.df_raw)

            if self.retroicor:
                try:
                    self.df_func_retroicor = pd.concat(
                        self.df_retro).set_index(self.index_list)
                    self.df_physio_r2 = pd.concat(self.df_r2)
                except Exception:
                    raise ValueError(
                        "RETROICOR did not complete successfully..")

            if self.ica:

                # check if elements of list contain dataframes
                if all(elem is None for elem in self.df_ica):
                    utils.verbose(
                        "WARNING: ICA did not execute properly. All runs have 'None'", True)
                else:
                    try:
                        self.df_func_ica = pd.concat(
                            self.df_ica).set_index(self.index_list)
                    except Exception:
                        self.df_func_ica = pd.concat(self.df_ica)

        # now that we have nicely formatted functional data, initialize the ParseExpToolsFile-class
        if self.tsv_file is not None:
            ParseExpToolsFile.__init__(
                self,
                self.tsv_file,
                subject=self.sub,
                deleted_first_timepoints=self.deleted_first_timepoints,
                TR=self.TR,
                edfs=self.edf_file,
                funcs=self.func_file,
                use_bids=self.use_bids,
                button=self.button,
                verbose=self.verbose,
                invoked_from_func=True,
                save_as=self.save_as,
                **kwargs
            )

    def preprocess_func_file(
        self,
        func_file,
        run=1,
        task=None,
        deleted_first_timepoints=0,
        deleted_last_timepoints=0,
        baseline=None,
        **kwargs
    ):

        # ----------------------------------------------------------------------------------------------------------------------------------------------
        # BASIC DATA LOADING

        # Load in datasets with tag "wcsmtSNR"
        self.stop_process = False
        if isinstance(func_file, str):
            if func_file.endswith("mat"):

                # load matlab file
                self.ts_wcsmtSNR = io.loadmat(func_file)

                # decide which key to read from the .mat file
                if self.func_tag is None:
                    self.tag = list(self.ts_wcsmtSNR.keys())[-1]
                else:
                    self.tag = self.func_tag

                # select data
                self.ts_wcsmtSNR = self.ts_wcsmtSNR[self.tag]
                self.ts_complex = self.ts_wcsmtSNR
                self.ts_magnitude = np.abs(self.ts_wcsmtSNR)

            elif func_file.endswith('gii'):
                self.gif_obj = ParseGiftiFile(func_file)

                self.ts_magnitude = self.gif_obj.data
                if not self.transpose:
                    self.ts_magnitude = self.gif_obj.data.T

            elif func_file.endswith("npy"):
                self.ts_magnitude = np.load(func_file)

                if self.transpose:
                    self.ts_magnitude = np.load(func_file).T

            elif func_file.endswith("nii") or func_file.endswith("gz"):

                # read niimg
                nimg = nb.load(func_file)
                fdata = nimg.get_fdata()

                # read TR from header
                if not isinstance(nimg, nb.cifti2.cifti2.Cifti2Image):
                    self.hdr = nimg.header
                    self.affine = nimg.affine
                    self.TR = self.hdr["pixdim"][4]

                # cifti nii's are already 2D
                if fdata.ndim > 2:
                    self.orig_dim = fdata.shape
                    xdim, ydim, zdim, time_points = fdata.shape
                    self.ts_magnitude = fdata.reshape(
                        xdim*ydim*zdim, time_points)
                else:
                    self.ts_magnitude = fdata.copy().T

            elif func_file.endswith("pkl"):
                with open(func_file, 'rb') as handle:
                    df = pickle.load(handle)
                    setattr(self, f"data_{self.standardization}_df", df)
                    setattr(self, "data_raw_df", df)

                self.stop_process = True
        elif isinstance(func_file, np.ndarray):
            self.ts_magnitude = func_file.copy()

        else:
            raise NotImplementedError(
                f"Input type {type(func_file)} not supported")

        if not self.stop_process:
            # check baseline
            if not self.psc_nilearn:
                if self.baseline_units in ["seconds", "s", "sec"]:
                    baseline_vols_old = int(np.round(baseline*self.fs, 0))
                    utils.verbose(
                        f" Baseline is {rb}{baseline}{end} seconds, or {rb}{baseline_vols_old}{end} TRs", self.verbose)
                else:
                    baseline_vols_old = baseline
                    utils.verbose(f" Baseline is {rb}{baseline}{end} TRs", self.verbose)

                # correct for deleted samples
                baseline_vols = baseline_vols_old-self.deleted_first_timepoints
                txt = f" (also cut from baseline (was {baseline_vols_old}, now {baseline_vols} TRs)"
            else:
                txt = ""
                baseline_vols = 0

            # trim beginning and end
            if deleted_last_timepoints != 0:
                self.ts_corrected = self.ts_magnitude[:,
                                                      deleted_first_timepoints:-deleted_last_timepoints]
            else:
                self.ts_corrected = self.ts_magnitude[:,
                                                      deleted_first_timepoints:]

            utils.verbose(
                f" Cutting {rb}{deleted_first_timepoints}{end} vols from beginning{txt} | {rb}{deleted_last_timepoints}{end} vols from end",
                self.verbose)
            self.vox_cols = [f'vox {x}' for x in range(
                self.ts_corrected.shape[0])]

            # ----------------------------------------------------------------------------------------------------------------------------------------------
            # STANDARDIZATION OF UNFILTERED DATA & CREATE DATAFRAMES

            # dataframe of raw, unfiltered data
            self.data_raw = self.ts_corrected.copy()
            self.data_raw_df = self.index_func(
                self.data_raw,
                columns=self.vox_cols,
                subject=self.sub,
                task=task,
                run=run,
                TR=self.TR,
                set_index=True
            )

            # dataframe of unfiltered PSC-data
            self.data_psc = utils.percent_change(
                self.data_raw,
                1,
                baseline=baseline_vols,
                nilearn=self.psc_nilearn
            )

            self.data_psc_df = self.index_func(
                self.data_psc,
                columns=self.vox_cols,
                subject=self.sub,
                task=task,
                run=run,
                TR=self.TR,
                set_index=True
            )

            # dataframe of unfiltered z-scored data
            self.data_zscore = (self.data_raw-self.data_raw.mean(axis=1,
                                keepdims=True))/self.data_raw.std(axis=1, keepdims=True)
            self.data_zscore_df = self.index_func(
                self.data_zscore,
                columns=self.vox_cols,
                subject=self.sub,
                task=task,
                run=run,
                TR=self.TR,
                set_index=True)

            # ----------------------------------------------------------------------------------------------------------------------------------------------
            # HIGH PASS FILTER
            self.clean_tag = None
            if self.filter_strategy != "raw":

                utils.verbose(
                    f" DCT-high pass filter [removes low frequencies <{rb}{self.lb}{end} Hz] to correct low-frequency drifts.",
                    self.verbose)

                # highpass data
                self.hp_raw, self._cosine_drift = preproc.highpass_dct(
                    self.data_raw,
                    lb=self.lb,
                    TR=self.TR
                )

                # index df
                self.hp_raw_df = self.index_func(
                    self.hp_raw,
                    columns=self.vox_cols,
                    subject=self.sub,
                    task=task,
                    run=run,
                    TR=self.TR,
                    set_index=True
                )

                # dataframe of high-passed PSC-data (set NaN to 0)
                self.hp_psc = np.nan_to_num(
                    utils.percent_change(
                        self.hp_raw,
                        1,
                        baseline=baseline_vols,
                        nilearn=self.psc_nilearn
                    )
                )

                # index df
                self.hp_psc_df = self.index_func(
                    self.hp_psc,
                    columns=self.vox_cols,
                    subject=self.sub,
                    run=run,
                    task=task,
                    TR=self.TR,
                    set_index=True
                )

                # dataframe of high-passed z-scored data
                self.hp_zscore = (self.hp_raw-self.hp_raw.mean(axis=1,
                                  keepdims=True))/self.hp_raw.std(axis=1, keepdims=True)
                
                # index df
                self.hp_zscore_df = self.index_func(
                    self.hp_zscore,
                    columns=self.vox_cols,
                    subject=self.sub,
                    run=run,
                    TR=self.TR,
                    task=task,
                    set_index=True
                )

                # save SD and Mean so we can go from zscore back to original
                self.zscore_SD = self.hp_raw.std(axis=-1, keepdims=True)
                self.zscore_M = self.hp_raw.mean(axis=-1, keepdims=True)

                if self.ica:
                    
                    # run ICA
                    self.run_ica(
                        task=task,
                        save_as=self.save_as
                    )
                    self.clean_tag = "ica"
                    self.clean_data = self.ica_obj.ica_data

                if hasattr(self, "clean_data"):
                    self.tmp_df = self.index_func(
                        self.clean_data,
                        columns=self.vox_cols,
                        subject=self.sub,
                        task=task,
                        run=run,
                        TR=self.TR,
                        set_index=True
                    )

                    setattr(self, f"hp_{self.clean_tag}_df", self.tmp_df)

                    # multiply by SD and add mean
                    self.tmp_raw = (self.clean_data * self.zscore_SD) + self.zscore_M
                    setattr(self, f"hp_{self.clean_tag}_raw", self.tmp_raw)

                    self.tmp_raw_df = self.index_func(
                        self.tmp_raw,
                        columns=self.vox_cols,
                        subject=self.sub,
                        task=task,
                        run=run,
                        TR=self.TR,
                        set_index=True
                    )

                    setattr(self, f"hp_{self.clean_tag}_raw_df", self.tmp_raw)

                    # make percent signal
                    self.hp_tmp_psc = np.nan_to_num(
                        utils.percent_change(
                            self.tmp_raw,
                            1,
                            baseline=baseline_vols,
                            nilearn=self.psc_nilearn
                        )
                    )

                    setattr(self, f"hp_{self.clean_tag}_psc", self.hp_tmp_psc)
                    self.hp_tmp_psc_df = self.index_func(
                        self.hp_tmp_psc,
                        columns=self.vox_cols,
                        subject=self.sub,
                        task=task,
                        run=run,
                        TR=self.TR,
                        set_index=True
                    )

                    setattr(self, f"hp_{self.clean_tag}_psc_df", self.hp_tmp_psc_df)

                # --------------------------------------------------------------------------------
                # LOW PASS FILTER
                if "lp" in self.filter_strategy:

                    if self.ica:
                        info = f" Using {self.clean_tag}-data for low-pass filtering"
                        data_for_filtering = self.get_data(
                            index=True,
                            filter_strategy="hp",
                            dtype=self.standardization,
                            ica=self.ica
                        ).T.values
                        out_attr = f"lp_{self.clean_tag}_{self.standardization}"
                    elif hasattr(self, f"hp_{self.standardization}"):
                        info = " Using high-pass filtered data for low-pass filtering"
                        data_for_filtering = getattr(
                            self, f"hp_{self.standardization}")
                        out_attr = f"lp_{self.standardization}"
                    else:
                        info = " Using unfiltered data for low-pass filtering"
                        data_for_filtering = getattr(
                            self, f"data_{self.standardization}")
                        out_attr = f"lp_data_{self.standardization}"

                    utils.verbose(info, self.verbose)
                    utils.verbose(
                        f" Savitsky-Golay low-pass filter [removes high freqs] (w={rb}{self.window_size}{end}, o={rb}{self.poly_order}{end})",
                        self.verbose)

                    tmp_filtered = preproc.lowpass_savgol(
                        data_for_filtering,
                        window_length=self.window_size,
                        polyorder=self.poly_order
                    )

                    tmp_filtered_df = self.index_func(
                        tmp_filtered,
                        columns=self.vox_cols,
                        subject=self.sub,
                        task=task,
                        run=run,
                        TR=self.TR,
                        set_index=True
                    )

                    setattr(self, out_attr, tmp_filtered.copy())
                    setattr(self, f'{out_attr}_df', tmp_filtered_df.copy())

            else:
                self.clean_tag = None

            # get basic qualities
            self.basic_qa(
                self.ts_corrected,
                run=run,
                save_as=self.save_as,
                make_figure=self.report
            )

    def to_nifti(self, func, fname=None):

        func_res = func.reshape(self.orig_dim)
        print(func_res.shape)
        niimg = nb.Nifti1Image(func_res, affine=self.affine, header=self.hdr)

        if isinstance(fname, str):
            niimg.to_filename(fname)
            return fname
        else:
            return niimg

    def run_ica(
        self,
        task=None,
        save_as=None
        ):
        
        utils.verbose(
            f" Running FastICA with {rb}{self.n_components}{end} components", self.verbose)

        self.ica_obj = preproc.ICA(
            self.hp_zscore_df,
            subject=f"sub-{self.sub}",
            ses=self.ses,
            run=self.run,
            task=task,
            n_components=self.n_components,
            TR=self.TR,
            filter_confs=self.filter_confs,
            keep_comps=self.keep_comps,
            verbose=self.verbose,
            zoom_freq=True,
            ribbon=tuple(self.gm_range),
            save_as=save_as,
            summary_plot=self.report
        )

        # regress
        self.ica_obj.regress()

    def basic_qa(
        self,
        data,
        run=1,
        make_figure=False,
        save_as=None
        ):

        # tsnr
        tsnr_pre = utils.calculate_tsnr(data, -1)
        mean_tsnr_pre = float(np.nanmean(np.ravel(tsnr_pre)))

        # variance
        var_pre = np.var(data, axis=-1)
        mean_var_pre = float(var_pre.mean())

        tsnr_inputs = [tsnr_pre]
        var_inputs = [var_pre]
        colors = "#1B9E77"
        tsnr_lbl = None
        var_lbl = None
        tsnr_lines = {
            "pos": mean_tsnr_pre,
            "color": colors
        }

        var_lines = {
            "pos": mean_var_pre,
            "color": colors
        }
        if self.verbose:
            if self.clean_tag is None:
                info = "no cleaning"
            else:
                info = f"before '{self.clean_tag}'"

            utils.verbose(
                f" tSNR [{info}]: {rb}{round(mean_tsnr_pre,2)}{end}\t| variance: {rb}{round(mean_var_pre,2)}{end}", self.verbose)

        if self.clean_tag in ["ica"]:

            # get aCompCor/ICA'ed tSNR
            tsnr_post = utils.calculate_tsnr(
                getattr(self, f"hp_{self.clean_tag}_raw"), -1)
            mean_tsnr_post = float(np.nanmean(np.ravel(tsnr_post)))

            # variance
            var_post = np.var(
                getattr(self, f"hp_{self.clean_tag}_raw"), axis=-1)
            mean_var_post = float(var_post.mean())

            # sort out plotting stuff
            tsnr_inputs += [tsnr_post]
            var_inputs += [var_post]
            colors = [colors, "#D95F02"]
            tsnr_lbl = [
                f'no {self.clean_tag} [{round(mean_tsnr_pre,2)}]', f'{self.clean_tag} [{round(mean_tsnr_post,2)}]']
            var_lbl = [
                f'no {self.clean_tag} [{round(mean_var_pre,2)}]', f'{self.clean_tag} [{round(mean_var_post,2)}]']
            tsnr_lines = {
                "pos": [mean_tsnr_pre, mean_tsnr_post],
                "color": colors
            }

            var_lines = {
                "pos": [mean_var_pre, mean_var_post],
                "color": colors
            }

            utils.verbose(
                f" tSNR [after '{self.clean_tag}']:  {rb}{round(mean_tsnr_post,2)}{end}\t| variance: {rb}{round(mean_var_post,2)}{end}",
                self.verbose)

        if make_figure:
            # initiate figure
            fig = plt.figure(figsize=(24, 7))
            gs = fig.add_gridspec(1, 3, width_ratios=[10, 10, 10])

            # imshow for apparent motion
            ax1 = fig.add_subplot(gs[0])
            ax1.imshow(np.rot90(data.T), aspect=8/1)

            vox_ticks = [0, data.shape[0]//2, data.shape[0]]
            ax1 = plotting.conform_ax_to_obj(
                ax1,
                x_label="volumes",
                y_label="voxels",
                title="Stability of position",
                font_size=16,
                x_ticks=[0, data.shape[-1]],
                y_ticks=vox_ticks
            )

            # line plot for tSNR over voxels
            ax2 = fig.add_subplot(gs[1])
            plotting.LazyLine(
                tsnr_inputs,
                axs=ax2,
                title="tSNR across the line",
                font_size=16,
                linewidth=2,
                color=colors,
                x_label="voxels",
                y_label="tSNR (a.u.)",
                labels=tsnr_lbl,
                add_hline=tsnr_lines,
                x_ticks=vox_ticks,
                line_width=2)

            ax3 = fig.add_subplot(gs[2])
            plotting.LazyLine(
                var_inputs,
                axs=ax3,
                title="Variance across the line",
                font_size=16,
                linewidth=2,
                color=colors,
                x_label="voxels",
                y_label="Variance",
                labels=var_lbl,
                add_hline=var_lines,
                x_ticks=vox_ticks,
                line_width=2
            )

            plt.close(fig)
            if self.report:
                if save_as is None:
                    save_as = opj(os.getcwd(), "figures")
                    if not os.path.exists(save_as):
                        os.makedirs(save_as, exist_ok=True)
                
                # save
                fname = opj(save_as, f"{self.base_name}_run-{run}_desc-qa.{self.save_ext}")
                utils.verbose(f" Writing {bm}{fname}{end}", self.verbose)
                fig.savefig(fname, bbox_inches='tight', dpi=300)

    def get_data(
        self,
        filter_strategy=None,
        index=False,
        dtype="psc",
        ica=False
    ):

        if dtype not in ["psc", "zscore", "raw"]:
            raise ValueError(
                f"Requested data type '{dtype}' is not supported. Use 'psc', 'zscore', or 'raw'")

        return_data = None
        allowed = [None, "raw", "hp", "lp"]

        if ica:
            tag = f"{self.clean_tag}_{dtype}"
        else:
            tag = dtype

        if filter_strategy is None or filter_strategy == "raw":
            attr = f"data_{tag}_df"
        elif filter_strategy == "lp":
            attr = f"lp_{tag}_df"
        elif filter_strategy == "hp":
            attr = f"hp_{tag}_df"
        else:
            raise ValueError(
                f"Unknown attribute '{filter_strategy}'. Must be one of: {allowed}")

        if hasattr(self, attr):
            # print(f" Fetching attribute: {attr}")
            return_data = getattr(self, attr)
        else:
            raise ValueError(
                f"{self} does not have an attribute called '{attr}'")

        if isinstance(return_data, pd.DataFrame):
            if index:
                try:
                    return return_data.set_index(['subject', 'run', 't'])
                except Exception:
                    return return_data
            else:
                return return_data

    def index_func(
            self,
            array,
            columns=None,
            subject=1,
            run=1,
            task=None,
            TR=0.105,
            set_index=False):

        if columns is None:
            df = pd.DataFrame(array.T)
        else:
            df = pd.DataFrame(array.T, columns=columns)

        df['subject'] = subject
        df['run'] = run
        df['t'] = list(TR*np.arange(df.shape[0]))

        if self.index_task:
            if isinstance(task, str):
                df["task"] = task
            else:
                df["task"] = "task"

        if set_index:
            return df.set_index(self.index_list)
        else:
            return df


class Dataset(ParseFuncFile, SetAttributes):
    """Dataset

    Main class for retrieving, formatting, and preprocessing of all datatypes including fMRI (2D), eyetracker (.edf),
    physiology (.log [WIP]), and experiment files derived from `Exptools2` (.tsv). If you leave `subject` and `run` empty,
    these elements will be derived from the file names. So if you have BIDS-like files, leave them empty and the dataframe
    will be created for you with the correct subject/run IDs.

    Inherits from :class:`lazyfmri.dataset.ParseFuncFile`, so all arguments from that class are available and are passed
    on via `kwargs`. Only `func_file` and `verbose` are required. The first one is necessary because if the input is an
    *h5*-file, we'll set the attributes accordingly. Otherwise :class:`lazyfmri.dataset.ParseFuncFile` is invoked.
    `verbose` is required for aesthetic reasons. Given that :class:`lazyfmri.dataset.ParseFuncFile` inherits in turn from
    :class:`lazyfmri.dataset.ParseExpToolsFile`, you can pass the arguments for that class here as well.

    Parameters
    ----------
    func_file: str, list
        path or list of paths pointing to the output file of the experiment
    verbose: bool, optional
        Print details to the terminal, default is False

    Example
    ----------
    .. code-block:: python

        from fmriproc import dataset
        from lazyfmri import utils
        func_dir = "/some/dir"
        exp = utils.get_file_from_substring("tsv", func_dir)
        funcs = utils.get_file_from_substring("bold.mat", func_dir)

        # only cut from SR-runs
        delete_first = 100
        delete_last = 0
        window = 19
        order = 3

        data = dataset.Dataset(
            funcs,
            deleted_first_timepoints=delete_first,
            deleted_last_timepoints=delete_last,
            tsv_file=exp,
            verbose=True)
        
        # retrieve data
        fmri = data.fetch_fmri()
        onsets = data.fetch_onsets()
    """

    def __init__(
        self,
        func_file,
        verbose=False,
        **kwargs
        ):

        self.verbose = verbose
        utils.verbose("DATASET", self.verbose)

        # set attributes
        SetAttributes.__init__(self)

        if isinstance(func_file, str) and func_file.endswith(".h5"):
            self.from_hdf(func_file)

            # set all kwargs
            # print(kwargs)
            self.__dict__.update(kwargs)
        else:
            ParseFuncFile.__init__(
                self,
                func_file,
                verbose=self.verbose,
                **kwargs
            )

        utils.verbose("\nDATASET: created", self.verbose)

    def fetch_fmri(self, strip_index=False, dtype=None):

        if dtype is None:
            if hasattr(self, "ica"):
                if self.ica:
                    dtype = "ica"
            elif hasattr(self, "standardization"):
                dtype = self.standardization
            else:
                dtype = "psc"

        if dtype == "psc":
            attr = 'df_func_psc'
        elif dtype == "retroicor":
            attr = 'df_func_retroicor'
        elif dtype == "raw" or dtype is None:
            attr = 'df_func_raw'
        elif dtype == "zscore":
            attr = 'df_func_zscore'
        elif dtype == "ica":
            attr = 'df_func_ica'
        else:
            raise ValueError(
                f"Unknown option '{dtype}'. Must be 'psc', 'retroicor', or 'zscore'")

        if hasattr(self, attr):

            utils.verbose(
                f"Fetching dataframe from attribute '{rb}{attr}{end}'", self.verbose)

            df = getattr(self, attr)
            if strip_index:
                return df.reset_index().drop(labels=['subject', 'run', 't'], axis=1)
            else:
                return df
        else:
            utils.verbose(f"Could not find '{attr}' attribute", True)

    def fetch_onsets(self, strip_index=False, button=True):
        if hasattr(self, 'df_onsets'):
            if strip_index:
                df = self.df_onsets.reset_index().drop(
                    labels=list(self.df_onsets.index.names), axis=1)
            else:
                df = self.df_onsets

            # filter out button
            if not button:
                df = utils.select_from_df(
                    df, expression="event_type != response")

            return df

        else:
            utils.verbose("No event-data was provided", True)

    def fetch_rts(self, strip_index=False):
        if hasattr(self, 'df_rts'):
            if strip_index:
                return self.df_rts.reset_index().drop(labels=list(self.df_rts.index.names), axis=1)
            else:
                return self.df_rts
        else:
            utils.verbose("No reaction times were provided", True)

    def fetch_accuracy(self, strip_index=False):
        if hasattr(self, 'df_accuracy'):
            if strip_index:
                return self.df_accuracy.reset_index().drop(labels=list(self.df_accuracy.index.names), axis=1)
            else:
                return self.df_accuracy
        else:
            utils.verbose("No accuracy measurements found", True)

    def fetch_physio(self, strip_index=False):
        if hasattr(self, 'df_physio'):
            if strip_index:
                return self.df_physio.reset_index().drop(labels=list(self.df_physio.index.names), axis=1)
            else:
                return self.df_physio
        else:
            utils.verbose("No physio-data was provided", True)

    def fetch_trace(self, strip_index=False):
        if hasattr(self, 'df_space_func'):
            if strip_index:
                return self.df_space_func.reset_index().drop(labels=list(self.df_space_func.index.names), axis=1)
            else:
                return self.df_space_func
        else:
            utils.verbose("No eyetracking-data was provided", True)

    def fetch_blinks(self, strip_index=False):
        if hasattr(self, 'df_blinks'):
            return self.df_blinks
        else:
            utils.verbose("No eyetracking-data was provided", True)

    def from_hdf(self, input_file=None):

        if not isinstance(input_file, str):
            raise ValueError("No output file specified")
        else:
            self.h5_file = input_file

        if not os.path.exists(self.h5_file):
            raise FileNotFoundError(f"Could not find file: '{self.h5_file}'")

        utils.verbose(f"Reading from {gb}{self.h5_file}{end}", self.verbose)
        hdf_store = pd.HDFStore(self.h5_file)
        hdf_keys = hdf_store.keys()
        for key in hdf_keys:
            key = key.strip("/")

            try:
                setattr(self, key, hdf_store.get(key))
                utils.verbose(f" Set attribute: {rb}{key}{end}", self.verbose)
            except Exception:
                utils.verbose(
                    f" Could not set attribute '{key}'", self.verbose)

        hdf_store.close()

    def to_hdf(
        self,
        h5_file=None,
        overwrite=False,
    ):

        # make the directory
        if not os.path.exists(os.path.dirname(h5_file)):
            os.makedirs(os.path.dirname(h5_file))

        if overwrite:
            if os.path.exists(h5_file):
                store = pd.HDFStore(h5_file)
                store.close()
                os.remove(h5_file)

        utils.verbose(f"Saving to {bm}{h5_file}{end}", self.verbose)
        for attr in self.all_attributes:
            if hasattr(self, attr):

                add_df = getattr(self, attr)
                # try regular storing
                if isinstance(add_df, pd.DataFrame):
                    try:
                        add_df.to_hdf(
                            h5_file,
                            key=attr,
                            append=True,
                            mode='a',
                            format='t'
                        )

                        utils.verbose(f" Stored attribute: {gb}{attr}{end}", self.verbose)
                    except Exception:
                        # send error message
                        utils.verbose(f" Could not store attribute '{attr}'", self.verbose)

        # define json file
        self.json_file = h5_file.split(".")[0]+".json"
        if os.path.exists(self.json_file):
            os.remove(self.json_file)

        # Serializing json
        json_object = json.dumps(self.func_settings, indent=4)

        # Writing to sample.json
        with open(self.json_file, "w") as outfile:
            outfile.write(json_object)

        utils.verbose("Done", self.verbose)

        store = pd.HDFStore(h5_file)
        store.close()

    def to4D(self, fname=None, desc=None, dtype=None, mask=None):

        # get dataset
        df = self.fetch_fmri(dtype=dtype)

        subj_list = self.get_subjects(df)
        file_counter = 0
        for sub in subj_list:

            # get subject-specific data
            data_per_subj = utils.select_from_df(
                df, expression=f"subject = {sub}")

            # get run IDs
            n_runs = self.get_runs(df)
            for run in n_runs:

                # get run-specific data
                data_per_run = utils.select_from_df(
                    data_per_subj, expression=f"run = {run}")

                # get corresponding reference image from self.func_file either based on index (if use_bids=False), or based on
                # BIDS-elements (use_bids=True)
                if self.use_bids:
                    ref_img = utils.get_file_from_substring(
                        [f'sub-{sub}', f'run-{run}'], self.func_file)
                else:
                    ref_img = self.func_file[file_counter]

                utils.verbose(f"Ref img = {gb}{ref_img}{end}", self.verbose)
                if isinstance(ref_img, nb.Nifti1Image):
                    ref_img = ref_img
                elif isinstance(ref_img, str):
                    if ref_img.endswith("gz") or ref_img.endswith("nii"):
                        ref_img = nb.load(ref_img)
                    else:
                        raise TypeError(
                            f"Unknown reference type '{ref_img}'. Must be a string pointing to 'nii' or 'nii.gz' file")
                else:
                    raise ValueError(
                        "'ref_img' must either be string pointing to nifti image or a nb.Nifti1Image object")

                # get information of reference image
                dims = ref_img.get_fdata().shape
                aff = ref_img.affine
                hdr = ref_img.header

                utils.verbose(f"Ref shape = {rb}{dims}{end}", self.verbose)
                data_per_run = data_per_run.values
                # time is initially first axis, so transpose
                if data_per_run.shape[-1] != dims[-1]:
                    utils.verbose(
                        f"Data shape = {rb}{data_per_run.shape}{end}; transposing..", self.verbose)
                    data_per_run = data_per_run.T
                else:
                    utils.verbose(
                        f"Data shape = {rb}{data_per_run.shape}{end}; all good..", self.verbose)

                utils.verbose(
                    f"Final shape = {rb}{data_per_run.shape}{end}", self.verbose)

                # check if we have mask
                if isinstance(mask, nb.Nifti1Image) or isinstance(mask, str) or isinstance(mask, list):

                    utils.verbose(
                        "Masking with given mask-object", self.verbose)
                    if isinstance(mask, nb.Nifti1Image):
                        mask = mask
                    elif isinstance(mask, str):
                        if mask.endswith("gz") or mask.endswith("nii"):
                            mask = nb.load(mask)
                        else:
                            raise TypeError(
                                f"Unknown reference type '{mask}'. Must be a string pointing to 'nii' or 'nii.gz' file")
                    elif isinstance(mask, list):
                        # select mask based on BIDS-components or index
                        if self.use_bids:
                            mask = utils.get_file_from_substring(
                                [f'sub-{sub}', f'run-{run}'], mask)
                        else:
                            mask = mask[file_counter]
                    else:
                        raise TypeError(
                            f"""Unknown input '{type(mask)}', must be nibabel.Nifti1Image-object or string pointing to
                            nifti-image""")

                    # mask array
                    mask_data = mask.get_fdata()
                    mask_data = mask_data.reshape(np.prod(mask_data.shape))
                    brain_idc = np.where(mask_data > 0)[0]
                    data_masked = np.zeros_like(data_per_run)

                    # fill zeroed array with brandata
                    data_masked[brain_idc, :] = data_per_run[brain_idc, :]

                    # overwrite
                    data_per_run = data_masked.copy()

                # reshape
                data_per_run = data_per_run.reshape(*dims)

                # save
                if not isinstance(fname, str):
                    if isinstance(desc, str):
                        fname = f"sub-{sub}_run-{run}_desc-{desc}.nii.gz"
                    else:
                        fname = f"sub-{sub}_run-{run}.nii.gz"
                else:
                    if isinstance(desc, str):
                        fname = f"{fname}_run-{run}_desc-{desc}.nii.gz"
                    else:
                        fname = f"{fname}_run-{run}.nii.gz"

                utils.verbose(f"Writing {gb}{fname}{end}", self.verbose)
                nb.Nifti1Image(data_per_run, affine=aff,
                               header=hdr).to_filename(fname)

                file_counter += 1


class DatasetCollector():
    def __init__(self, dataset_objects):

        self.datasets = dataset_objects
        if len(self.datasets) is not None:
            self.data = []
            self.onsets = []
            for dataset in self.datasets:
                self.data.append(dataset.fetch_fmri())

                # check if we got onsets
                if hasattr(dataset, 'df_onsets'):
                    onsets = True
                    self.onsets.append(dataset.fetch_onsets())
                else:
                    onsets = False

            self.data = pd.concat(self.data)
            if onsets:
                self.onsets = pd.concat(self.onsets)

# this is basically a wrapper around pybest.utils.load_gifti


class ParseGiftiFile():

    """ParseGiftiFile

    Read a gifti-file into a dataframe similar to :class:`lazyfmri.dataset.ParseFuncFile`. Also allows you to set the RepetitionTime in the metadata and rewrite the gifti-file.
    The final data is set as ``self.data``, representing the `numpy.ndarray`-form of the gifti-file.

    Parameters
    ----------
    gifti_file: str
        Path pointing to gifti-file
    set_tr: int, float, optional
        Set the TR in milliseconds in the metadata, by default None
    \*gii_args, \*gii_kwargs: dict
        Arguments passed to `nb.gifti.GiftiDataArray`
    Raises
    ----------
    ValueError
        If ``gifti-file`` does not end with ".gii".

    Example
    ----------
    .. code-block:: python

        from lazyfmri import dataset
        gii_file = "sub-01_ses-1_task-rest_space-fsnative_bold.gii"
        obj = dataset.ParseGiftiFile(gii_file)
        data = obj.data

    """

    def __init__(self, gifti_file, set_tr=None, *gii_args, **gii_kwargs):

        if isinstance(gifti_file, str):
            self.gifti_file = gifti_file
            self.f_gif = nb.load(self.gifti_file)
            self.data = np.vstack([arr.data for arr in self.f_gif.darrays])
        elif isinstance(gifti_file, np.ndarray):
            self.gifti_file = None
            self.data = gifti_file
        else:
            raise ValueError("Input must be a string ending with '.gii' or a numpy array")

        # get TR
        self.set_tr = set_tr

        if isinstance(self.set_tr, (int, float)):
            self.meta_obj = self.set_metadata(tr=self.set_tr)
            self.TR_ms = float(self.meta_dict['TimeStep'])
            self.TR_sec = float(self.meta_dict['TimeStep']) / 1000

            # overwrite original file
            if isinstance(self.gifti_file, str):
                self.write_file(self.gifti_file, *gii_args, **gii_kwargs)
        else:
            if len(self.f_gif.darrays[0].metadata) > 0:
                self.TR_ms = float(self.f_gif.darrays[0].metadata['TimeStep'])
                self.TR_sec = float(
                    self.f_gif.darrays[0].metadata['TimeStep']) / 1000

    def set_metadata(self, tr=None):
        self.meta_dict = {'TimeStep': str(float(tr))}
        return nb.gifti.GiftiMetaData().from_dict(self.meta_dict)

    def get_tr(self, units="sec"):
        if units not in ["sec", "ms"]:
            raise ValueError(
                f"units must be one of 'sec' or 'ms', not '{units}'")

        if hasattr(self, f"TR_{units}"):
            return getattr(self, f"TR_{units}")
        else:
            return None

    def write_file(
        self,
        filename,
        tr=None,
        **kwargs
        ):

        metadata = None
        if not isinstance(tr, (int, float)):
            if hasattr(self, "meta_obj"):
                metadata = self.meta_obj
        else:
            metadata = self.set_metadata(tr=tr)
            self.TR_ms = tr
            self.TR_sec = tr/1000

        # copy old data and combine it with metadata
        darray = nb.gifti.GiftiDataArray(
            self.data,
            meta=metadata,
            **kwargs
        )

        # store in new gifti image object
        gifti_image = nb.GiftiImage()

        # add data to this object
        gifti_image.add_gifti_data_array(darray)

        # save in same file name
        nb.save(gifti_image, filename)
