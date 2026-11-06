from . import (
    glm,
    utils,
    plotting,
)
import lmfit
import warnings
import fnmatch
import operator
from contextlib import nullcontext
from itertools import combinations
import numpy as np
import pandas as pd
import nideconv as nd
import seaborn as sns
import matplotlib as mpl
from typing import Union
from scipy import signal
from sklearn import metrics, linear_model
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from joblib import Parallel, delayed
from alive_progress import alive_bar


def _sample_time_grid(interval, step):
    start, stop = map(float, interval)
    if not np.isfinite([start, stop, step]).all() or step <= 0 or stop <= start:
        raise ValueError('Specify a positive step and increasing finite interval')
    count = int(np.ceil((stop - start) / step - 1e-10))
    return start + np.arange(count) * step

class CurveFitter():
    """CurveFitter

    Simple class to perform a quick curve fitting procedure on ``y_data``.
    You can either specify your own function with ``func``, or select a polynomial of order ``order`` (currently up until
    3rd-order is included).
    Internally uses ``lmfit.Model`` to perform the fitting, allowing for access to confidence intervals.

    Parameters
    ----------
    y_data: np.ndarray
        Data-points to perform fitting on
    x: np.ndarray, optional
        Array describing the x-axis, by default None.
    func: function-object, optional
        Use custom function describing the behavior of the fit, by default None.
        If ``None``, we'll assume either a linear or polynomial fit (up to 3rd order)
    order: str, int, optional
        Order of polynomial fit, by default 1 (linear).
    verbose: bool, optional
        Print summary of fit, by default True
    interpolate: str, optional
        Method of interpolation for an upsampled version of the predicted data (default = 1000 samples)

    Raises
    ----------
    NotImplementedError
        If ``func=None`` and no valid polynomial order (see above) was specified

    Example
    ----------
    .. code-block:: python

        # imports
        from lazyfmri import fitting
        import matplotlib.pyplot as plt
        import numpy as np

        # define data points
        data = np.array([5.436, 5.467, 5.293, 0.99 , 2.603, 1.902, 2.317])

        # create instantiation of CurveFitter
        cf = fitting.CurveFitter(data, order=3, verbose=False)

        # initiate figure with axis to be fed into LazyLine
        fig, axs = plt.subplots(figsize=(8,8))

        # plot original data points
        axs.plot(cf.x, data, 'o', color="#DE3163", alpha=0.6)

        # plot upsampled fit with 95% confidence intervals as shaded error
        plotting.LazyLine(
            cf.y_pred_upsampled,
            xx=cf.x_pred_upsampled,
            error=cf.ci_upsampled,
            axs=axs,
            color="#cccccc",
            x_label="x-axis",
            y_label="y-axis",
            title="Curve-fitting with polynomial (3rd-order)"
        )
    """

    def __init__(
            self,
            y_data,
            x=None,
            func=None,
            order=1,
            verbose=True,
            interpolate='linear',
            fix_intercept=False,
            sigma=1):

        self.y_data = y_data
        self.func = func
        self.order = order
        self.x = x
        self.verbose = verbose
        self.interpolate = interpolate
        self.fix_intercept = fix_intercept
        self.sigma = sigma

        self.init_params()
        self.run()

    def init_params(self):

        """init_params

        Initializes parameters for the selected fitting model. If no custom function (`func`) is provided,
        it selects a polynomial or predefined model (`"gaussian"`, `"exponential"`).

        Raises
        ------
        NotImplementedError
            If an unsupported model type is provided.

        Returns
        -------
        None
            Updates `self.pmodel` with the selected model and initializes `self.params`.

        Example
        ----------
        .. code-block:: python

            cf = CurveFitter(y_data, order=2)
            cf.init_params()
        """

        if self.func is None:
            self.guess = True
            if isinstance(self.order, int):
                if self.order == 1:
                    self.pmodel = lmfit.models.LinearModel()
                elif self.order == 2:
                    self.pmodel = lmfit.models.QuadraticModel()
                else:
                    self.pmodel = lmfit.models.PolynomialModel(
                        order=self.order)
            elif isinstance(self.order, str):
                if self.order == 'gauss' or self.order == 'gaussian':
                    self.pmodel = lmfit.models.GaussianModel()
                elif self.order == 'exp' or self.order == 'exponential':
                    self.pmodel = lmfit.models.ExponentialModel()
                else:
                    raise NotImplementedError(
                        f"Model {self.order} is not implemented because I'm lazy..")
        else:
            self.guess = False
            self.pmodel = lmfit.Model(self.func)

        if not isinstance(self.x, list) and not isinstance(self.x, np.ndarray):
            self.x = np.arange(self.y_data.shape[0])

        # self.params = self.pmodel.make_params(a=1, b=1, c=1, d=1)
        if self.guess:
            self.params = self.pmodel.guess(self.y_data, self.x)
        else:
            self.params = self.pmodel.make_params(a=1, b=1, c=1, d=1)

        if self.fix_intercept:
            self.params['intercept'].value = 0
            self.params['intercept'].vary = False

    def run(self):

        """run

        Executes the fitting procedure using the selected model and updates predictions.

        Returns
        -------
        None
            Updates `self.result` with the fitting results and generates predicted values (`self.y_pred`),
            confidence intervals (`self.ci`), and upsampled versions (`self.y_pred_upsampled`).

        Example
        ----------
        .. code-block:: python

            cf = CurveFitter(y_data, order=3)
            cf.run()
        """

        self.result = self.pmodel.fit(self.y_data, self.params, x=self.x)

        if self.verbose:
            print(self.result.fit_report())

        # create predictions & confidence intervals that are compatible with
        # LazyLine
        self.y_pred = self.result.best_fit
        self.x_pred_upsampled = np.linspace(self.x[0], self.x[-1], 1000)
        self.y_pred_upsampled = self.result.eval(x=self.x_pred_upsampled)
        self.ci = self.result.eval_uncertainty(sigma=self.sigma)
        self.ci_upsampled = glm.resample_stim_vector(self.ci, len(
            self.x_pred_upsampled), interpolate=self.interpolate)
    
    @staticmethod
    def first_order(x, a, b):

        """first_order

        First-order polynomial function (linear model).

        Parameters
        ----------
        x : float or np.ndarray
            Input x-values.
        a : float
            Slope of the linear function.
        b : float
            Intercept of the linear function.

        Returns
        -------
        float or np.ndarray
            Computed y-values based on the linear function.

        Example
        ----------
        .. code-block:: python

            y = CurveFitter.first_order(x, a=2, b=1)
        """

        return a * x + b

    @staticmethod
    def second_order(x, a, b, c):

        """second_order

        Second-order polynomial function (quadratic model).

        Parameters
        ----------
        x : float or np.ndarray
            Input x-values.
        a : float
            Coefficient for the linear term.
        b : float
            Coefficient for the quadratic term.
        c : float
            Intercept.

        Returns
        -------
        float or np.ndarray
            Computed y-values based on the quadratic function.

        Example
        ----------
        .. code-block:: python

            y = CurveFitter.second_order(x, a=1, b=-0.5, c=2)
        """

        return a * x + b * x**2 + c

    @staticmethod
    def third_order(x, a, b, c, d):

        """third_order

        Third-order polynomial function (cubic model).

        Parameters
        ----------
        x : float or np.ndarray
            Input x-values.
        a : float
            Coefficient for the linear term.
        b : float
            Coefficient for the quadratic term.
        c : float
            Coefficient for the cubic term.
        d : float
            Intercept.

        Returns
        -------
        float or np.ndarray
            Computed y-values based on the cubic function.

        Example
        ----------
        .. code-block:: python

            y = CurveFitter.third_order(x, a=0.5, b=-0.3, c=2, d=1)
        """

        return (a * x) + (b * x**2) + (c * x**3) + d


class InitFitter():

    """InitFitter

    Initializes the fitter with functional data and onset times, formatting them for analysis.

    Parameters
    ----------
    func : pd.DataFrame
        Functional fMRI data as a pandas DataFrame.
    onsets : pd.DataFrame
        Onset timings corresponding to the events in the functional data.
    TR : float
        Repetition time (TR) in seconds.
    merge : bool, optional
        Whether to concatenate runs before analysis, by default `False`.

    Example
    ----------
    .. code-block:: python

        fitter = InitFitter(func, onsets, TR=1.32, merge=True)
    """

    def __init__(
        self,
        func,
        onsets,
        TR,
        merge=False
        ):

        self.func = func
        self.onsets = onsets
        self.TR = TR
        self.merge = merge

        # format inputs into Fitter compatbile input
        self.prepare_data()
        self.prepare_onsets()

        # concatenate runs
        if self.merge:
            self.concatenate()

    def concatenate(self):

        """concatenate

        Concatenates functional data and onset times across runs.

        Returns
        -------
        None
            Updates `self.func` and `self.onsets` with concatenated data.

        Example
        ----------
        .. code-block:: python

            fitter.concatenate()
        """

        # store originals
        self.og_func = self.func.copy()
        self.og_onsets = self.onsets.copy()

        # create concatenator dictionary
        self.concat_obj = self.concat_runs(self.func, self.onsets, TR=self.TR)

        # store new dataframes
        self.func = self.concat_obj["func"]
        self.onsets = self.concat_obj["onsets"]
        if getattr(self, 'confounds', None) is not None:
            self.confounds = self.concat_runs(self.confounds, self.og_onsets, TR=self.TR)['func']

    def _get_timepoints(self):

        """concatenate

        Concatenates functional data and onset times across runs.

        Returns
        -------
        None
            Updates `self.func` and `self.onsets` with concatenated data.

        Example
        ----------
        .. code-block:: python

            fitter.concatenate()
        """

        return list(np.arange(0, self.func.shape[0]) * self.TR)

    def prepare_onsets(self):
        """Keep session/task labels and event covariates when preparing onsets."""
        self.orig_onsets = self.onsets.copy()
        if isinstance(self.onsets, np.ndarray):
            frame = pd.DataFrame(self.onsets)
            if frame.shape[1] == 1:
                frame.columns = ['onset']
        else:
            frame = self.onsets.copy()
            if any(name is not None for name in frame.index.names):
                frame = frame.reset_index()
            else:
                frame = frame.reset_index(drop=True)
        frame = frame.rename(columns={k: v for k, v in
            [('ses', 'session'), ('trial_type', 'event_type'), ('event type', 'event_type')]
            if k in frame and v not in frame})
        for field, default in [('subject', 1), ('run', 1), ('event_type', 'stim')]:
            if field not in frame:
                frame[field] = default
        if 'onset' not in frame:
            raise ValueError('Onset data must contain an onset column')
        frame['onset'] = pd.to_numeric(frame['onset'], errors='raise')
        if not np.isfinite(frame['onset'].to_numpy()).all():
            raise ValueError('Event onsets must be finite')
        acquisition = [k for k in ('subject', 'session', 'task', 'run') if k in frame]
        self.final_index = acquisition + ['event_type']
        self.onset_index = self.final_index.copy()
        self.onsets = frame.set_index(self.final_index)

    def prepare_data(self):
        """Prepare signal data without dropping acquisition identifiers."""
        self.orig_data = self.func.copy()
        if isinstance(self.func, np.ndarray):
            frame = pd.DataFrame(self.func)
        else:
            frame = self.func.copy()
            if any(name is not None for name in frame.index.names):
                frame = frame.reset_index()
            else:
                frame = frame.reset_index(drop=True)
        if 'ses' in frame and 'session' not in frame:
            frame = frame.rename(columns={'ses': 'session'})
        if 'time' in frame and 't' not in frame:
            frame = frame.rename(columns={'time': 't'})
        for field in ('subject', 'run'):
            if field not in frame:
                frame[field] = 1
        acquisition = [k for k in ('subject', 'session', 'task', 'run') if k in frame]
        if 't' not in frame:
            frame['t'] = frame.groupby(acquisition, sort=False).cumcount() * self.TR
        self.final_index = acquisition + ['t']
        self.task_ids = frame['task'].unique().tolist() if 'task' in frame else None
        self.time = frame['t'].to_numpy()
        self.func = frame.set_index(self.final_index)
        if not self.func.index.is_unique:
            raise ValueError('Duplicate acquisition/time rows in functional data')
        if any(not pd.api.types.is_numeric_dtype(dtype) for dtype in self.func.dtypes):
            raise ValueError('Functional signal columns must be numeric; put acquisition labels in the index and remove source_file metadata')

    @classmethod
    def concat_func(self, df, TR=None):
        """Concatenate one subject/session/task's runs on a continuous time axis."""
        contexts = [key for key in df.index.names if key not in ('run', 't')]
        for key in contexts:
            if df.index.get_level_values(key).nunique() != 1:
                raise ValueError('concat_func expects one subject/session/task')
        pieces, offset = [], 0.
        for _, group in df.groupby(level='run', sort=False):
            group = group.sort_index(level='t')
            times = group.index.get_level_values('t').to_numpy(dtype=float)
            step = float(TR) if TR is not None else (float(np.diff(times)[0]) if len(times) > 1 else None)
            if step is None or step <= 0 or (len(times) > 1 and not np.allclose(np.diff(times), step)):
                raise ValueError('Provide TR and regularly sampled data for concatenation')
            frame = group.reset_index()
            frame['t'] = offset + np.arange(len(frame)) * step
            frame['run'] = 1
            offset += len(frame) * step
            pieces.append(frame)
        return pd.concat(pieces, ignore_index=True).set_index(contexts + ['run', 't'])

    @classmethod
    def concat_onsets(self, onsets, func, TR=None):
        """Shift onsets by cumulative run durations in seconds, not volume counts."""
        pieces, offset = [], 0.
        for run, group in func.groupby(level='run', sort=False):
            times = np.sort(group.index.get_level_values('t').to_numpy(dtype=float))
            step = float(TR) if TR is not None else (float(np.diff(times)[0]) if len(times) > 1 else None)
            if step is None or step <= 0:
                raise ValueError('Provide TR for concatenation of single-volume runs')
            frame = onsets.loc[onsets.index.get_level_values('run') == run].reset_index()
            frame['onset'] = frame['onset'] + offset
            frame['run'] = 1
            pieces.append(frame)
            offset += len(group) * step
        return pd.concat(pieces, ignore_index=True).set_index(onsets.index.names)

    @classmethod
    def concat_runs(self, func, onsets, TR=None):
        """Concatenate runs separately for each subject/session/task."""
        contexts = [key for key in func.index.names if key not in ('run', 't')]
        output = {'func': [], 'onsets': []}
        for key, data in func.groupby(level=contexts, sort=False):
            key = key if isinstance(key, tuple) else (key,)
            events = onsets
            for field, value in zip(contexts, key):
                if field not in events.index.names:
                    raise ValueError(f'Onsets lack acquisition field {field!r}')
                events = events.loc[events.index.get_level_values(field) == value]
            runs = data.index.get_level_values('run').unique()
            if not events.index.get_level_values('run').isin(runs).all():
                raise ValueError('Events reference runs absent from functional data')
            output['func'].append(self.concat_func(data, TR=TR))
            output['onsets'].append(self.concat_onsets(events, data, TR=TR))
        return {key: pd.concat(frames) for key, frames in output.items()}

    @staticmethod
    def merge_dfs(
        src,
        trg,
        name="full HRF",
        first="basissets"
        ):

        """merge_dfs

        Merges two dataframes by aligning their indices. Typically used for combining basis set estimates with 
        full HRF predictions.

        Parameters
        ----------
        src : pd.DataFrame
            Source dataframe.
        trg : pd.DataFrame
            Target dataframe to merge with.
        name : str, optional
            Label assigned to the merged dataframe, by default `"full HRF"`.
        first : str, optional
            Determines merge order, either `"basissets"` or `"full"`, by default `"basissets"`.

        Returns
        -------
        pd.DataFrame
            Merged dataframe.

        Example
        ----------
        .. code-block:: python

            merged_df = NideconvFitter.merge_dfs(df1, df2, name="HRF Comparison", first="full")
        """

        tm = trg.copy().reset_index()
        tm["covariate"] = name
        tm.set_index(list(src.index.names), inplace=True)

        if "basis" in first:
            order = [src, tm]
        else:
            order = [tm, src]

        return pd.concat(order)

    def merge_basissets_with_full_hrf(self, **kwargs):

        """merge_basissets_with_full_hrf

        Merges basis set regressors with full hemodynamic response function (HRF) predictions for further analysis.

        Parameters
        ----------
        **kwargs : dict
            Additional parameters for merging.

        Example
        ----------
        .. code-block:: python

            fitter = NideconvFitter(func, onsets, TR=1.32)
            merged_data = fitter.merge_basissets_with_full_hrf()
        """

        for pars in ["avg_pars", "pars"]:

            if hasattr(self, f"{pars}_basissets") and hasattr(
                    self, f"{pars}_subjects"):

                df1 = getattr(self, f"{pars}_basissets")
                df2 = getattr(self, f"{pars}_subjects")
                df3 = self.merge_dfs(
                    df1,
                    df2,
                    **kwargs
                )
                setattr(self, f"{pars}_merged", df3)

        if hasattr(self, "sub_basis") and hasattr(self, "tc_subjects"):
            df1 = self.sub_basis.copy()
            df2 = self.tc_subjects.copy().reset_index()
            df2.rename(columns={"time": "t"}, inplace=True)
            df2.set_index(["subject", "run", "event_type",
                          "covariate", "t"], inplace=True)

            df3 = self.merge_dfs(
                df1,
                df2,
                **kwargs
            )

            setattr(self, "sub_tc_merged", df3)

        if hasattr(self, "sub_basis") and hasattr(self, "tc_subjects"):
            df1 = self.sub_basis.copy()
            df2 = self.tc_subjects.copy().reset_index()
            df2.rename(columns={"time": "t"}, inplace=True)
            df2.set_index(["subject", "run", "event_type",
                          "covariate", "t"], inplace=True)

            df3 = self.merge_dfs(
                df1,
                df2,
                **kwargs
            )

            setattr(self, "sub_tc_merged", df3)

    def parameters_for_basis_sets(
        self,
        *args,
        **kwargs
        ):

        """parameters_for_basis_sets

        Extracts HRF parameters from basis set regressors.

        Parameters
        ----------
        *args : tuple, optional
            Additional arguments for :class:`lazyfmri.fitting.HRFMetrics`.
        **kwargs : dict, optional
            Additional keyword arguments for :class:`lazyfmri.fitting.HRFMetrics`.

        Returns
        -------
        None
            Updates `self.pars_basissets` with extracted parameters.

        Example
        ----------
        .. code-block:: python

            fitter.parameters_for_basis_sets()
        """

        if not hasattr(self, "sub_basis"):
            self.get_basisset_timecourses()

        utils.verbose(
            "Deriving parameters from basis sets with 'HRFMetrics'",
            self.verbose)
        subj_ids = utils.get_unique_ids(self.sub_basis, id="subject")

        self.avg_tcs_subjects = self.tc_subjects.groupby(
            ["subject", "event_type", "time"]).mean()

        subjs = []
        subjs_avg_run = []
        for sub in subj_ids:

            # get subject specific dataframe
            sub_df = utils.select_from_df(
                self.sub_basis, expression=f"subject = {sub}")

            # get basis sets
            basis_ids = utils.get_unique_ids(sub_df, id="covariate")
            basis = []
            basis_avg = []
            for bs in basis_ids:

                basis_df = utils.select_from_df(
                    sub_df, expression=f"covariate = {bs}")
                ev_ids = utils.get_unique_ids(basis_df, id="event_type")

                # loop through evs
                evs = []
                evs_avg = []
                for ev in ev_ids:

                    # get event-specific dataframe
                    ev_df = utils.select_from_df(
                        basis_df, expression=f"event_type = {ev}")
                    run_ids = utils.get_unique_ids(ev_df, id="run")

                    # loop through runs
                    runs = []
                    for run in run_ids:

                        # get run-specific dataframe
                        run_df = utils.select_from_df(
                            ev_df, expression=f"run = {run}")
                        pars = HRFMetrics(
                            run_df,
                            TR=self.TR,
                            *args,
                            **kwargs
                        ).return_metrics()

                        pars["event_type"], pars["run"] = ev, run
                        runs.append(pars)

                    # also get parameters of average across runs
                    avg_pars = HRFMetrics(
                        utils.multiselect_from_df(
                            self.avg_tcs_subjects,
                            expression=[
                                f"subject = {sub}",
                                f"event_type = {ev}"
                            ]
                        ),
                        TR=self.TR,
                        *args,
                        **kwargs
                    ).return_metrics()

                    # save average
                    avg_pars["event_type"] = ev
                    evs_avg.append(avg_pars)

                    # save single runs
                    runs = pd.concat(runs)
                    evs.append(runs)

                # avg
                evs_avg = pd.concat(evs_avg)
                evs_avg["covariate"] = bs
                basis_avg.append(evs_avg)

                # single runs
                evs = pd.concat(evs)
                evs["covariate"] = bs
                basis.append(evs)

            # avg
            basis_avg = pd.concat(basis_avg)
            basis_avg["subject"] = sub
            subjs_avg_run.append(basis_avg)

            # single runs
            basis = pd.concat(basis)
            basis["subject"] = sub
            subjs.append(basis)

        df_conc = pd.concat(subjs)
        df_avg = pd.concat(subjs_avg_run)
        avg_idx = ["subject", "event_type", "covariate"]
        idx = avg_idx + ["run"]

        add_idx = self.find_pars_index(df_conc)
        if isinstance(add_idx, str):
            idx += [add_idx]
            avg_idx += [add_idx]

        self.pars_basissets = df_conc.set_index(idx)
        self.avg_pars_basissets = df_avg.set_index(avg_idx)

    @staticmethod
    def find_pars_index(df):

        """find_pars_index

        Identifies any additional index columns present in a DataFrame that are not part of the expected indices.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to inspect.

        Returns
        -------
        str or list
            The name of the additional index column(s), if any.

        Example
        ----------
        .. code-block:: python

            index_col = InitFitter.find_pars_index(df)
        """

        default_indices = ["subject", "event_type", "covariate", "run"]
        parameters = [
            "magnitude",
            "magnitude_ix",
            "fwhm",
            "fwhm_obj",
            "time_to_peak",
            "half_rise_time",
            "half_max",
            "rise_slope",
            "onset_time",
            "positive_area",
            "undershoot",
            "auc_simple_pos",
            "auc_simple_neg",
            "auc_simple_total",
            "auc_simple_pos_norm",
            "auc_simple_neg_norm", 
            "auc_simple_total_norm",
            '1st_deriv_magnitude',
            '1st_deriv_time_to_peak',
            '2nd_deriv_magnitude',
            '2nd_deriv_time_to_peak',
        ]

        combined = default_indices + parameters
        df.reset_index(inplace=True)
        if "index" in list(df.columns):
            df.drop(["index"], axis=1, inplace=True)

        columns = list(df.columns)
        not_shared = []
        for i in columns:
            if i not in combined:
                not_shared.append(i)

        if len(not_shared) > 0:
            if len(not_shared) > 1:
                raise ValueError(
                    f"""Found multiple ({len(not_shared)}) column names in the dataframe, but I'm expecting only 1 to not
                    match.. These column names are extra: {not_shared}""")
            else:
                return not_shared[0]
        else:
            return []

    def parameters_for_tc_subjects(self, **kwargs):
        """Extract metrics independently per acquisition/event/covariate."""
        if not hasattr(self, 'tc_subjects'):
            raise ValueError('Run timecourses_condition() before extracting parameters')
        frame = self.tc_subjects
        names = list(frame.index.names)
        time = 'time' if 'time' in names else 't'
        groups = [key for key in names if key != time]
        self.pars_subjects = self._metrics_for_groups(frame, groups, **kwargs)
        average_groups = [key for key in groups if key != 'run']
        averaged = frame.groupby(level=average_groups + [time]).mean()
        self.avg_pars_subjects = self._metrics_for_groups(averaged, average_groups, **kwargs)

    def _metrics_for_groups(self, frame, groups, **kwargs):
        """Keep metric ROI identifiers and all profile grouping levels."""
        results = []
        for key, profile in frame.groupby(level=groups, sort=False, dropna=False):
            key = key if isinstance(key, tuple) else (key,)
            profile = profile.droplevel(groups).sort_index()
            if not profile.index.is_unique:
                raise ValueError('Metrics require one profile per acquisition/event/covariate')
            metrics = HRFMetrics(profile, TR=self.TR, **kwargs).return_metrics().copy()
            roi_field = kwargs.get('col_name', 'vox')
            metrics = metrics.reset_index(drop=True)
            if roi_field not in metrics:
                if len(metrics) != len(profile.columns):
                    raise ValueError('Metric rows do not match input signal columns')
                metrics[roi_field] = (np.arange(len(profile.columns)) if kwargs.get('vox_as_index', False)
                                      else profile.columns.tolist())
            for field, value in zip(groups, key):
                metrics[field] = value
            # Keep the historic leading subject/event/run order when present.
            order = [field for field in ('subject', 'event_type', 'run') if field in groups]
            order += [field for field in groups if field not in order]
            order += [roi_field] if roi_field not in order else []
            results.append(metrics.set_index(order))
        return pd.concat(results) if results else pd.DataFrame()

    def parameters_for_epochs(
        self,
        df=None,
        **kwargs
        ):

        if not isinstance(df, pd.DataFrame):
            raise ValueError(f"Please specify an input dataframe")
        
        epoch_ids = utils.get_unique_ids(df, id="epoch")
        self.tmp_pars_subjects = []
        self.tmp_avg_pars_subjects = []
        for i in epoch_ids:
            epoch_df = utils.select_from_df(
                df,
                expression=f"epoch = {i}"
            ).groupby(
                ["subject", "run", "event_type", "t"]
            ).mean()

            self.tc_subjects = epoch_df.copy()
            self.parameters_for_tc_subjects(**kwargs)
            
            # format (add epoch ID)
            self.tmp_pars_subjects.append(self.format_epoch_pars(self.pars_subjects.copy(), ix=i))
            self.tmp_avg_pars_subjects.append(self.format_epoch_pars(self.avg_pars_subjects.copy(), ix=i))

        self.pars_subjects = pd.concat(self.tmp_pars_subjects)
        self.avg_pars_subjects = pd.concat(self.tmp_avg_pars_subjects)
    
    def format_epoch_pars(
        self,
        pars_df,
        ix=1,
        idx=["subject", "event_type", "run", "vox"]
        ):
        
        idx = list(pars_df.index.names)
        try:
            pars_df.reset_index(inplace=True)
        except:
            pass

        if "index" in list(pars_df.columns):
            pars_df.drop(["index"], axis=1, inplace=True)

        pars_df["epoch"] = ix
        idx.insert(-1, "epoch")
        pars_df.set_index(idx, inplace=True)

        return pars_df

    def parameters_for_tc_condition(self, *args, **kwargs):
        """Extract group-profile metrics without mixing conditions or covariates."""
        if args:
            raise TypeError('Pass HRFMetrics options as keyword arguments')
        if not hasattr(self, 'tc_condition'):
            raise ValueError('Run timecourses_condition() before extracting parameters')
        frame = self.tc_condition.copy()
        frame['subject'] = 'avg'
        frame = frame.set_index('subject', append=True)
        groups = [key for key in frame.index.names if key not in ('time', 't')]
        self.pars_condition = self._metrics_for_groups(frame, groups, **kwargs)
    
    @classmethod
    def format_evs_for_plotting(self, df, indexer="event_type", time_col="time", **kwargs):

        """
        Formats a dataframe of event-related responses for plotting by averaging across
        time and computing error margins (SEM or STD) per condition to be compatible with.
        This output is compatible with plotting utilities such as `LazyLine <https://lazyfmri.readthedocs.io/en/latest/classes/plotting.html#lazyfmri.plotting.LazyLine>`_ plotting class from the LazyfMRI package.


        Parameters
        ----------
        df : pd.DataFrame
            Multi-indexed dataframe containing timecourses with event labels and time.
            Expected indices include at least `event_type` and `time`.
        indexer : str, optional
            Index level name corresponding to conditions (e.g., "event_type").
        time_col : str, optional
            Index level name corresponding to the time axis.
        **kwargs : dict
            Additional arguments passed to `format_for_plotting`, such as error type (`se='sem'` or `'std'`).

        Returns
        -------
        dict
            A dictionary with keys:
                - "tc" : list of mean timecourses per condition
                - "err": list of SEM or STD values per condition
                - "labels": list of condition names (e.g., event types)
                - "time": common time axis (shared across events)

        Example
        -------
        .. code-block:: python

            # initialize the fitter
            dec = fitting.NideconvFitter(
                df_func,
                df_onset,
                basis_sets="canonical_hrf_with_time_derivative",
                TR=0.105,
                interval=[-2,16]
            )

            # fetch the profiles
            dec.timecourses_condition()

            # format for plotting
            fmt = dec.format_evs_for_plotting(dec.tc_condition)

            # plot
            pl = plotting.LazyLine(
                fmt["tc"],
                xx=fmt["time"],
                figsize=(5,5),
                labels=["center","medium","large"],
                x_label="time (s)",
                y_label="amplitude",
                error=fmt["err"],
                line_width=3,
                color=["#1B9E77","#D95F02","#4c75ff"],
                add_hline=0
            )

            plotting.add_axvspan(
                pl.axs, 
                ymax=0.1
            )
        """ 
        evs = utils.get_unique_ids(df, id=indexer)
        ev_ddict = {}
        
        for i in ["tc", "err"]:
            ev_ddict[i] = []

        for ev in evs:
            ev_df = utils.select_from_df(df, expression=f"{indexer} = {ev}")
            tc_dict = self.format_for_plotting(ev_df, **kwargs)
            ev_ddict["tc"].append(tc_dict["tc"])
            ev_ddict["err"].append(tc_dict["err"])

        ev_ddict["labels"] = evs
        ev_ddict["time"] = utils.get_unique_ids(df, id=time_col)
        return ev_ddict
    
    @staticmethod
    def format_for_plotting(df, se="sem"):
        """
        Computes the mean and error values (SEM or STD) for a timecourse dataframe.
        
        Parameters
        ----------
        df : pd.DataFrame
            A dataframe where rows represent timepoints and columns are different samples or subjects.
        se : str, optional
            Error metric to compute:
                - "sem" or "se": standard error of the mean
                - "std" or "sd": standard deviation

        Returns
        -------
        dict
            A dictionary with:
                - "tc" : mean values (1D array)
                - "err": error margins (SEM or STD, 1D array)

        Raises
        ------
        NotImplementedError
            If `se` is not one of the supported values.

        Example
        -------
        >>> result = format_for_plotting(df, se="std")
        >>> plt.plot(time, result["tc"])
        >>> plt.fill_between(time, result["tc"] - result["err"], result["tc"] + result["err"])
        """


        ddict = {}
        ddict["tc"] = df.mean(axis=1).values

        if se in ["sem", "se"]:
            ddict["err"] = df.sem(axis=1).values
        elif se in ["std", "sd"]:
            ddict["err"] = df.std(axis=1).values
        else:
            raise NotImplementedError(f"Error type '{se}' is not supported. Must be one of 'sem', 'se', 'std' or 'sd'")

        return ddict
    
class ParameterFitter(InitFitter):

    """ParameterFitter

    Initializes the `ParameterFitter` class for estimating hemodynamic response function (HRF) parameters
    using an optimizer procedure.

    Parameters
    ----------
    func : pd.DataFrame
        Dataframe containing fMRI data indexed by subject, run, and time.
    onsets : pd.DataFrame
        Dataframe containing onset timings indexed by subject, run, and event type.
    TR : float, optional
        Repetition time (TR) of the fMRI acquisition, by default 0.105 seconds.
    merge : bool, optional
        Whether to concatenate runs before parameter estimation, by default `False`.
    verbose : bool, optional
        Whether to print additional information during processing, by default `False`.
    *args : tuple
        Additional positional arguments.
    **kwargs : dict
        Additional keyword arguments for parameter fitting.

    Returns
    -------
    None
        Initializes the object and stores parameters internally.

    Example
    ----------
    .. code-block:: python

        from lazyplot import fitting

        # Initialize the fitter
        nd_fit = fitting.ParameterFitter(
            func,
            onsets,
            merge=True,
            verbose=True
        )
    """

    def __init__(
        self,
        func,
        onsets,
        TR=0.105,
        merge=False,
        verbose=False,
        *args,
        **kwargs
        ):

        self.func = func
        self.onsets = onsets
        self.TR = TR
        self.merge = merge
        self.verbose = verbose

        # prepare data
        super().__init__(
            self.func,
            self.onsets,
            self.TR,
            merge=self.merge
        )

        # self.func is overwritting by FitHRFParams class; so store original
        # inputs otherwise looping goes awry
        self.orig_func = self.func.copy()
        self.orig_onsets = self.onsets.copy()

        # get info about dataframe
        self.sub_ids = utils.get_unique_ids(self.orig_func, id="subject")
        self.run_ids = utils.get_unique_ids(self.orig_func, id="run")
        self.ev_ids = utils.get_unique_ids(self.orig_onsets, id="event_type")

    @staticmethod
    def single_response_fitter(
            data,
            onsets,
            TR=0.105,
            **kwargs):

        """single_response_fitter

        Fits a single HRF response using the `FitHRFparams` optimizer.

        Parameters
        ----------
        data : np.ndarray or pd.DataFrame
            Functional data, either a NumPy array or a Pandas DataFrame with time-series data.
        onsets : pd.DataFrame
            Dataframe containing onset timings indexed by subject, run, and event type.
        TR : float, optional
            Repetition time (TR) of the fMRI acquisition, by default 0.105 seconds.
        **kwargs : dict
            Additional keyword arguments for HRF fitting.

        Returns
        -------
        FitHRFparams
            A fitted HRF parameters object.

        Example
        ----------
        .. code-block:: python

            from lazyplot.fitting import FitHRFparams

            # Fit single HRF response
            fitted_hrf = ParameterFitter.single_response_fitter(func, onsets, TR=0.105)
        """

        cl_fit = FitHRFparams(
            data,
            onsets,
            TR=TR,
            **kwargs
        )

        cl_fit.iterative_fit()

        return cl_fit

    @staticmethod
    def _set_dict():

        """_set_dict

        Initializes an empty dictionary to store HRF fitting results.

        Returns
        -------
        dict
            Dictionary with keys: `predictions`, `profiles`, and `pars`, each initialized as an empty list.

        Example
        ----------
        .. code-block:: python

            fit_dict = ParameterFitter._set_dict()
            print(fit_dict)  # {'predictions': [], 'profiles': [], 'pars': []}
        """

        ddict = {}
        for el in ["predictions", "profiles", "pars"]:
            ddict[el] = []

        return ddict

    @staticmethod
    def _concat_dict(ddict):

        """_concat_dict

        Concatenates lists of Pandas DataFrames stored in a dictionary.

        Parameters
        ----------
        ddict : dict
            Dictionary containing lists of DataFrames under the keys: `predictions`, `profiles`, and `pars`.

        Returns
        -------
        dict
            Dictionary where lists of DataFrames are concatenated.

        Example
        ----------
        .. code-block:: python

            combined_dict = ParameterFitter._concat_dict(fit_dict)
        """

        new_dict = {}
        for key, val in ddict.items():
            if len(val) > 0:
                new_dict[key] = pd.concat(val)

        return new_dict

    def fit(self, debug=False, **kwargs):
        """Fit only existing acquisition/event combinations and retain ROI labels."""
        self.ddict_ev = self._set_dict()
        identities = [key for key in ('subject', 'session', 'task', 'run') if key in self.orig_func.index.names]
        for key, data in self.orig_func.groupby(level=identities, sort=False):
            key = key if isinstance(key, tuple) else (key,)
            selection = dict(zip(identities, key))
            events = NideconvFitter._select_acquisition(self.orig_onsets, selection)
            if events.empty:
                raise ValueError(f'No onsets for functional acquisition {selection}')
            for event, onsets in events.groupby(level='event_type', sort=False):
                self.rf = self.single_response_fitter(data.to_numpy().T, onsets, TR=self.TR, **kwargs)
                for field, destination, time_fields in [
                    ('hrf_profiles', 'profiles', ['covariate', 'time']),
                    ('predictions', 'predictions', ['t'])]:
                    frame = getattr(self.rf, field).copy()
                    if frame.shape[1] != len(data.columns):
                        raise ValueError('Parametric output columns do not match ROI inputs')
                    frame.columns = data.columns
                    frame = frame.reset_index()
                    for name, value in selection.items(): frame[name] = value
                    frame['event_type'] = event
                    self.ddict_ev[destination].append(frame.set_index(identities + ['event_type'] + time_fields)[list(data.columns)])
                estimates = self.rf.estimates.copy().reset_index(drop=True)
                if len(estimates) != len(data.columns):
                    raise ValueError('Parametric estimates do not match ROI inputs')
                estimates['vox'], estimates['event_type'] = list(data.columns), event
                for name, value in selection.items(): estimates[name] = value
                self.ddict_ev['pars'].append(estimates.set_index(identities + ['event_type', 'vox']))
        self.ddict_sub = self._concat_dict(self.ddict_ev)
        self.tc_subjects = self.ddict_sub['profiles']
        self.ev_predictions = self.ddict_sub['predictions']
        self.estimates = self.ddict_sub['pars']
        contexts = [key for key in identities if key != 'run']
        subject_means = self.tc_subjects.groupby(level=contexts + ['event_type', 'covariate', 'time']).mean()
        condition = [key for key in contexts if key != 'subject'] + ['event_type', 'covariate', 'time']
        self.grouper = subject_means.groupby(level=condition)
        self.tc_condition = self.grouper.mean()
        self.sem_condition, self.std_condition = self.grouper.sem(), self.grouper.std()
        self.time = np.sort(self.tc_condition.index.get_level_values('time').unique().to_numpy())


class NideconvFitter(InitFitter):
    """NideconvFitter

    A wrapper class around :class:`nideconv.GroupResponseFitter` for streamlined and flexible fMRI deconvolution.
    This class simplifies reproducibility, handles pandas indexing issues, and allows for efficient multiple 
    deconvolutions. Designed to work with **functional MRI (fMRI) data**, **event onsets**, and grey matter ribbon voxels.

    **Main Features**
    - Supports **basis sets** (`'fourier'` or `'fir'`).
    - Implements **ordinary least squares (OLS) fitting**.
    - Handles **confounds** and **covariates**.
    - Includes **multiple visualization functions**.

    Parameters
    ----------
    func : pd.DataFrame
        fMRI data indexed by `subject`, `run`, and `t`.
    onsets : pd.DataFrame
        Dataframe with event onsets indexed by `subject`, `run`, and `event_type`.
    TR : float, optional
        Repetition time in seconds. Default is `0.105`.
    confounds : pd.DataFrame, optional
        Confound regression matrix matching `func`. Default is `None`.
    basis_sets : str, optional
        Type of basis set (`'fourier'` or `'fir'`). Default is `'fourier'`.
    fit_type : str, optional
        Fitting method (`'ols'` or `'ridge'`). Default is `'ols'`.
    n_regressors : int or str, optional
        Number of regressors (`'tr'` syncs regressors with TR). Default is `9`.
    add_intercept : bool, optional
        Include an unmodulated event response alongside supplied event
        covariates. Without covariates, unit event amplitudes are always used.
        This does not control the constant run baseline. Default is `False`.
    lump_events : bool, optional
        If `True`, merges all event types. Default is `False`.
    interval : list, optional
        Time window for fitting regressors. Default is `[0, 20]`.
    osf : int, optional
        Oversampling factor for design matrix. Default is `20`.
    fit : bool, optional
        If `True`, fits the model upon initialization. Default is `True`.
    covariates : dict, optional
        Covariates for each event (dict of numpy arrays). Default is `None`.
    conf_intercept : bool, optional
        Include one constant baseline per acquisition. The baseline is not
        penalized in ridge fits. Default is `True`.
    event_models : dict, optional
        Per-event overrides, matched by exact event name or an unambiguous
        shell-style wildcard (for example 'CS_*'). Exact names take priority.
        Each dictionary may override basis_sets (or basis_set), n_regressors,
        interval, duration, covariates, add_intercept and fir_bin_mode. duration='from_onsets'
        retains TSV durations; duration=0 creates onset-locked impulses. When
        supplied, every observed event must match a rule. All events are fitted
        jointly in each acquisition. Effective settings are in model_spec.
    on_rank_deficient : {"raise", "skip"}, optional
        OLS policy for singular designs (default "raise"). All run models are
        retained in all_fitters and diagnostics in fit_diagnostics. With
        "skip", only successful runs enter profiles and predictions. Ridge
        continues to allow singular designs. To inspect a raised failure,
        construct with fit=False before calling fit(), or use error.fitter.
    drop_zero_regressors : bool, optional
        OLS only: omit exactly zero design columns before checking rank and
        fitting (default False). Keep the full design and coefficient labels;
        unidentifiable coefficients and affected profile samples are NaN.
        Other linear dependencies still follow on_rank_deficient.
    fir_bin_mode : {'legacy', 'equal'}, optional
        FIR bin construction, overridable per event. 'equal' requires the
        corresponding nideconv update and uses equal-width half-open bins.
        Default 'legacy' preserves existing designs and supports older nideconv.
    **kwargs
        Additional parameters for model initialization.

    Example
    ----------
    .. code-block:: python

        from lazyfmri import dataset, fitting
        df_func = dataset.fetch_fmri()
        df_onsets = dataset.fetch_onsets()

        nd_fit = fitting.NideconvFitter(
            df_func,
            df_onsets,
            basis_sets='fourier',
            n_regressors=4,
            TR=0.105,
            interval=[0,20]
        )
    """

    def __init__(
        self,
        func,
        onsets,
        TR=0.105,
        confounds=None,
        basis_sets="fourier",
        fit_type="ols",
        n_regressors=9,
        add_intercept=False,
        merge=False,
        verbose=False,
        lump_events=False,
        interval=[0, 20],
        osf=20,
        fit=True,
        covariates=None,
        conf_intercept=True,
        event_models=None,
        on_rank_deficient="raise",
        drop_zero_regressors=False,
        fir_bin_mode="legacy",
        **kwargs):

        if on_rank_deficient not in ("raise", "skip"):
            raise ValueError("on_rank_deficient must be raise or skip")
        self.on_rank_deficient = on_rank_deficient
        if not isinstance(drop_zero_regressors, (bool, np.bool_)):
            raise ValueError("drop_zero_regressors must be boolean")
        if drop_zero_regressors and fit_type.lower() != "ols":
            raise ValueError("drop_zero_regressors currently supports OLS only")
        self.drop_zero_regressors = drop_zero_regressors
        if fir_bin_mode not in ("legacy", "equal"):
            raise ValueError("fir_bin_mode must be legacy or equal")
        self.fir_bin_mode = fir_bin_mode
        self.event_models = event_models
        if not np.isfinite(TR) or TR <= 0:
            raise ValueError('TR must be finite and positive')
        if not isinstance(osf, (int, np.integer)) or osf < 1:
            raise ValueError('osf must be a positive integer')
        self.func = func
        self.onsets = onsets
        self.confounds = confounds
        self.basis_sets = basis_sets
        self.fit_type = fit_type.lower()
        if self.fit_type not in ('ols', 'ridge'):
            raise ValueError("fit_type must be 'ols' or 'ridge'")
        self.n_regressors = n_regressors
        self.add_intercept = add_intercept
        self.verbose = verbose
        self.lump_events = lump_events
        self.TR = TR
        self.fs = 1 / self.TR
        self.interval = list(interval)
        self.merge = merge
        self.osf = osf
        self.do_fit = fit
        self.covariates = covariates
        self.conf_icpt = conf_intercept

        # check if interval is multiple of TR
        if isinstance(self.interval[-1], str):
            self.interval[-1] = int(self.interval[-1]) * self.TR

        # format arrays as dataframe
        super().__init__(
            self.func,
            self.onsets,
            self.TR,
            merge=self.merge
        )

        # merge all events?
        if self.lump_events:
            self.used_onsets = self.melt_events()
        else:
            self.used_onsets = self.onsets.copy()

        # get n regressors
        self.derive_n_regressors()

        # update kwargs
        self.__dict__.update(kwargs)
        self._resolve_event_models()

        # get the model
        if self.fit_type == "ols":
            self.define_model()

        # specify the events
        self.define_events()

        # # fit
        if self.do_fit:
            self.fit()

        # set plotting defaults
        self.set_plotting_defaults()

    def set_plotting_defaults(self):

        """set_plotting_defaults

        Initializes default settings for plot visualization.

        Returns
        -------
        None
            Stores plotting defaults in the object attributes.

        Example
        ----------
        .. code-block:: python

            fitter.set_plotting_defaults()
        """


        # some plotting defaults
        self.plotting_defaults = plotting.Defaults()
        if not hasattr(self, "font_size"):
            self.font_size = self.plotting_defaults.font_size

        if not hasattr(self, "label_size"):
            self.label_size = self.plotting_defaults.label_size

        if not hasattr(self, "tick_width"):
            self.tick_width = self.plotting_defaults.tick_width

        if not hasattr(self, "tick_length"):
            self.tick_length = self.plotting_defaults.tick_length

        if not hasattr(self, "axis_width"):
            self.axis_width = self.plotting_defaults.axis_width

    def melt_events(self):
        """Lump event types while retaining all acquisition identifiers."""
        self.lumped_onsets = self.onsets.copy().reset_index()
        self.lumped_onsets['event_type'] = 'stim'
        self.lumped_onsets = self.lumped_onsets.set_index(self.onsets.index.names)
        return self.lumped_onsets

    def derive_n_regressors(self):
        """Validate the legacy global basis and resolve its regressor count."""
        self.allowed_basis_sets = ['fir', 'fourier', 'dct', 'legendre', 'canonical_hrf',
            'canonical_hrf_with_time_derivative', 'canonical_hrf_with_time_derivative_dispersion']
        self.tmp_regressors = round((self.interval[1] - self.interval[0]) / self.TR)
        self.n_regressors = self._event_regressor_count(self.basis_sets, self.n_regressors, self.interval)

    def _event_regressor_count(self, basis, count, interval):
        allowed = ['fir', 'fourier', 'dct', 'legendre', 'canonical_hrf',
            'canonical_hrf_with_time_derivative', 'canonical_hrf_with_time_derivative_dispersion']
        if basis not in allowed:
            raise ValueError(f'Unrecognized basis {basis!r}; choose from {allowed}')
        if basis.startswith('canonical_hrf'):
            return {'canonical_hrf': 1, 'canonical_hrf_with_time_derivative': 2,
                    'canonical_hrf_with_time_derivative_dispersion': 3}[basis]
        if isinstance(count, str):
            tokens = count.lower().split()
            count = round((interval[1] - interval[0]) / self.TR)
            if len(tokens) == 3 and tokens[0] == 'tr':
                operations = {'x': operator.mul, '*': operator.mul, '/': operator.truediv,
                              '+': operator.add, '-': operator.sub}
                if tokens[1] not in operations:
                    raise ValueError('Use tr, or tr <x, /, +, -> <number>')
                count = int(operations[tokens[1]](count, float(tokens[2])))
            elif tokens != ['tr']:
                raise ValueError('Use a positive regressor count or tr <operation> <number>')
        if not isinstance(count, (int, np.integer)) or count < 1:
            raise ValueError('n_regressors must resolve to a positive integer')
        if basis in ('fourier', 'legendre') and count % 2 == 0:
            count += 1
        return int(count)

    def _event_spec(self, event, ix=0):
        if hasattr(self, 'event_models_'):
            return self.event_models_[str(event)]
        return dict(basis_sets=getattr(self, 'basis_sets', 'fir'), n_regressors=self.n_regressors,
            interval=list(self.interval), duration='from_onsets',
            fir_bin_mode=getattr(self, 'fir_bin_mode', 'legacy'),
            covariates=self.covariates[ix] if isinstance(getattr(self, 'covariates', None), list) else getattr(self, 'covariates', None),
            add_intercept=self.add_intercept[ix] if isinstance(getattr(self, 'add_intercept', None), list) else getattr(self, 'add_intercept', False))

    def _resolve_event_models(self):
        """Resolve exact names before unambiguous shell-style wildcard patterns.

        All observed events must match when event_models is supplied. Numeric
        duration overrides are written only into used_onsets, never caller data.
        model_spec records every effective event configuration.
        """
        models = self.event_models
        if models is not None and (not isinstance(models, dict) or not models):
            raise ValueError('event_models must be a nonempty dictionary or None')
        events = utils.get_unique_ids(self.used_onsets, id='event_type')
        resolved = {}
        used_rules = set()
        for ix, event in enumerate(events):
            name = str(event)
            base = self._event_spec(event, ix)
            if models is not None:
                if name in models:
                    rule = name
                else:
                    matches = [pattern for pattern in models if fnmatch.fnmatchcase(name, pattern)]
                    if len(matches) != 1:
                        raise ValueError(f'{name}: expected one matching event model, found {matches}')
                    rule = matches[0]
                override = models[rule]
                if not isinstance(override, dict):
                    raise ValueError(f'{rule}: event model must be a dictionary')
                override = dict(override)
                if 'basis_set' in override:
                    if 'basis_sets' in override:
                        raise ValueError('Use only one of basis_set and basis_sets')
                    override['basis_sets'] = override.pop('basis_set')
                if set(override) - set(base):
                    raise ValueError(f'{rule}: unknown model fields {set(override) - set(base)}')
                base.update(override)
                used_rules.add(rule)
            if base['fir_bin_mode'] not in ('legacy', 'equal'):
                raise ValueError(f'{name}: fir_bin_mode must be legacy or equal')
            interval = list(map(float, base['interval']))
            if len(interval) != 2 or not np.isfinite(interval).all() or interval[1] <= interval[0]:
                raise ValueError(f'{name}: interval must contain two increasing finite times')
            base['interval'] = interval
            base['n_regressors'] = self._event_regressor_count(base['basis_sets'], base['n_regressors'], interval)
            duration = base['duration']
            if duration is None:
                duration = 0.
            if duration != 'from_onsets':
                if not isinstance(duration, (int, float, np.number)) or not np.isfinite(duration) or duration < 0:
                    raise ValueError(f'{name}: duration must be nonnegative or from_onsets')
                mask = self.used_onsets.index.get_level_values('event_type') == event
                if 'duration' not in self.used_onsets:
                    self.used_onsets['duration'] = np.nan
                self.used_onsets.loc[mask, 'duration'] = float(duration)
            base['duration'] = duration
            resolved[name] = base
        if models is not None:
            unused = set(models) - used_rules
            if unused:
                warnings.warn(f'Event model rules not used by this dataset: {sorted(unused)}', RuntimeWarning)
        self.event_models_ = resolved
        self.fir_events = [name for name, spec in resolved.items() if spec['basis_sets'] == 'fir']
        self.is_hybrid = len({spec['basis_sets'] for spec in resolved.values()}) > 1
        self.model_spec = pd.DataFrame.from_dict(resolved, orient='index').rename_axis('event_type')

    @staticmethod
    def _prediction_betas(fitter):
        """Zero omitted coefficients only for predictions on the observed design."""
        betas = fitter.betas.copy()
        dropped = getattr(fitter, 'dropped_zero_regressors', [])
        if len(dropped):
            betas.loc[dropped] = 0.
        return betas

    @staticmethod
    def _profiles_from_fitter(fitter, oversample=None):
        """Retain estimable FIR samples without treating missing bins as zero."""
        if not len(getattr(fitter, 'dropped_zero_regressors', [])):
            return fitter.get_timecourses(oversample=oversample)
        if oversample is None:
            oversample = fitter.oversample_design_matrix
        profiles = []
        for name, event in fitter.events.items():
            basis = event.get_basis_function(oversample=oversample)
            for covariate, coefficients in event.betas.groupby(level='covariate', sort=False):
                coefficients = coefficients.droplevel(['event type', 'covariate']).reindex(basis.columns)
                missing = coefficients.isna().to_numpy()
                values = basis.to_numpy().dot(coefficients.fillna(0.).to_numpy())
                affected = (basis.to_numpy() != 0.).astype(int).dot(missing.astype(int)) > 0
                values[affected] = np.nan
                frame = pd.DataFrame(values, index=basis.index, columns=coefficients.columns)
                frame['event type'], frame['covariate'] = name, covariate
                profiles.append(frame.reset_index().set_index(['event type', 'covariate', 'time']))
        return pd.concat(profiles)

    @staticmethod
    def get_event_predictions_from_fitter(fitter, intercept=True):
        """Predict each event, optionally adding only the constant run baseline.

        Event profiles exclude this baseline; unrelated nuisance regressors
        are never added to individual event predictions.
        """
        output = []
        labels = fitter.X.columns.get_level_values(0)
        for event in fitter.events:
            selected = labels == event
            if intercept:
                selected |= np.asarray([column[0] == 'confounds' and column[1] == 'intercept'
                                        for column in fitter.X.columns])
            design = fitter.X.loc[:, selected]
            prediction = design.dot(NideconvFitter._prediction_betas(fitter).loc[design.columns]).copy()
            prediction.index.name = 't'
            prediction = prediction.reset_index()
            prediction['event_type'] = event
            output.append(prediction)
        return pd.concat(output, ignore_index=True) if output else pd.DataFrame()

    def _interpolate_timecourse(self, tc, time=None, centers=None, target_time=None, event_type=None):
        """Linearly interpolate FIR bin-center estimates, for display only.

        Bin locations depend on the fitted interval and number of regressors,
        never on amplitude changes. No extrapolation outside the centers is used.
        Existing calls with only tc remain supported.
        """
        tc = np.asarray(tc, dtype=float)
        time = np.asarray(self.time if time is None else time, dtype=float)
        if tc.ndim != 1 or time.ndim != 1 or len(tc) != len(time) or not len(tc):
            raise ValueError('tc and time must be nonempty, equally sized 1D arrays')
        order = np.argsort(time)
        time, tc = time[order], tc[order]
        if not np.isfinite(time).all() or np.any(np.diff(time) <= 0):
            raise ValueError('FIR time coordinates must be finite and unique')
        if centers is None:
            centers = self._fir_bin_centers(event_type)
        centers = np.asarray(centers, dtype=float)
        spec = self._event_spec(event_type) if event_type is not None else dict(interval=self.interval, n_regressors=self.n_regressors)
        width = (float(spec['interval'][1]) - float(spec['interval'][0])) / int(spec['n_regressors'])
        if np.any(np.min(np.abs(time[:, None] - centers[None, :]), axis=0) >= width / 2):
            raise ValueError('FIR profile grid does not sample every bin; increase evaluation oversampling')
        # Choose the sample nearest each bin center. This recovers the plateau's
        # coefficient without merging equal adjacent coefficients or edge mixing.
        indices = np.abs(time[:, None] - centers[None, :]).argmin(axis=0)
        values = tc[indices]
        if target_time is None:
            target_time = np.unique(np.r_[centers, time[(time >= centers[0]) & (time <= centers[-1])]])
        target_time = np.asarray(target_time, dtype=float)
        if np.any(target_time < centers[0]) or np.any(target_time > centers[-1]):
            raise ValueError('FIR interpolation cannot extrapolate outside bin centers')
        return np.interp(target_time, centers, values), target_time

    def _fir_bin_centers(self, event_type=None):
        """Use actual FIR basis support; retain a uniform fallback for array-only callers."""
        table = None
        if hasattr(self, 'model') and hasattr(self.model, '_get_response_fitters'):
            table = self.model._get_response_fitters()
        if table is None:
            table = getattr(self, 'fitters', getattr(self, 'sub_df', None))
        if isinstance(table, (pd.Series, pd.DataFrame)):
            for fitter in table.to_numpy().reshape(-1):
                for event_name, event in getattr(fitter, 'events', {}).items():
                    if event_type is not None and str(event_name) != str(event_type):
                        continue
                    if getattr(event, 'basis_set', None) != 'fir':
                        continue
                    basis = event.get_basis_function(oversample=getattr(self, 'osf', 20))
                    times = basis.index.to_numpy(dtype=float)
                    centers = []
                    for column in basis:
                        support = times[np.abs(basis[column].to_numpy()) > 0]
                        if not len(support):
                            raise ValueError('FIR basis contains an unsampled regressor')
                        centers.append((support[0] + support[-1]) / 2)
                    centers = np.asarray(centers)
                    if np.any(np.diff(centers) <= 0):
                        raise ValueError('FIR basis supports are not ordered')
                    return centers
        spec = self._event_spec(event_type) if event_type is not None else dict(interval=self.interval, n_regressors=self.n_regressors)
        count = int(spec['n_regressors'])
        start, stop = map(float, spec['interval'])
        if count < 1 or count != spec['n_regressors'] or not np.isfinite([start, stop]).all() or stop <= start:
            raise ValueError('FIR requires a positive integer regressor count and increasing interval')
        edges = np.linspace(start, stop, count + 1)
        return (edges[:-1] + edges[1:]) / 2


    def interpolate_fir_subjects(self):
        """Interpolate each acquisition/event/covariate on a common FIR grid."""
        return self.interpolate_fir_condition(self.tc_subjects)

    def interpolate_fir_condition(self, obj):
        """Interpolate FIR events only; retain other event curves unchanged."""
        frame = obj.copy()
        if 'event type' in frame.index.names:
            frame.index = frame.index.set_names([
                'event_type' if name == 'event type' else name for name in frame.index.names])
        if 'time' not in frame.index.names:
            raise ValueError('Profiles require a time index')
        levels = [name for name in frame.index.names if name != 'time']
        cache, output = {}, []
        for key, group in frame.groupby(level=levels, sort=False, dropna=False):
            key = key if isinstance(key, tuple) else (key,)
            event = str(dict(zip(levels, key))['event_type'])
            spec = self._event_spec(event)
            if spec['basis_sets'] != 'fir':
                output.append(group)
                continue
            if event not in cache:
                centers = self._fir_bin_centers(event)
                event_mask = frame.index.get_level_values('event_type') == event
                times = np.sort(frame.loc[event_mask].index.get_level_values('time').unique().to_numpy(dtype=float))
                target = np.unique(np.r_[centers, times[(times >= centers[0]) & (times <= centers[-1])]])
                cache[event] = centers, target
            centers, target = cache[event]
            group = group.droplevel(levels).sort_index()
            if not group.index.is_unique:
                raise ValueError('Duplicate time points within a profile')
            values = {}
            for column in frame.columns:
                values[column], _ = self._interpolate_timecourse(group[column].to_numpy(),
                    time=group.index.to_numpy(), centers=centers, target_time=target, event_type=event)
            result = pd.DataFrame(values, index=pd.Index(target, name='time')).reset_index()
            for name, value in zip(levels, key):
                result[name] = value
            output.append(result.set_index(frame.index.names))
        return pd.concat(output) if output else frame.iloc[:0].copy()

    def format_fitters(self):

        """format_fitters

        Converts fitted model output into a structured dataframe for easier indexing and retrieval.

        Returns
        -------
        None
            Stores formatted fitters in the object attribute `fitters`.

        Example
        ----------
        .. code-block:: python

            fitter.format_fitters()
            print(fitter.fitters)
        """

        # pass
        self.fitters = (self.sub_df.copy() if self.fit_type == 'ridge'
                        else self.model._get_response_fitters())
        if not isinstance(self.fitters, pd.DataFrame):
            self.fitters = pd.DataFrame(self.fitters)

        self.fitters = self.check_for_run_index(self.fitters)

    @staticmethod
    def get_curves_from_fitter(fitter, index=True, icpt=False):
        """Extract each basis contribution using label-aligned coefficients.

        Legacy output names event_type/covariate/t are retained; covariate names
        continue to identify basis regressors. Multiple event covariates are
        disambiguated as <covariate>:<regressor>.
        """
        basis = fitter.get_basis_functions()
        betas = fitter.betas
        output = []
        for event in fitter.events:
            event_basis = basis.xs(event, level='event type', drop_level=False)
            event_rows = event_basis.index.get_level_values('time')
            event_columns = [column for column in basis.columns if column[0] == event and column in betas.index]
            covariates = list(dict.fromkeys(column[1] for column in event_columns))
            for column in event_columns:
                selected = [column]
                if icpt:
                    # Include the fitted intercept, not unrelated nuisance signals.
                    selected = [c for c in basis.columns if c not in event_columns
                        and c in betas.index and c[0] not in fitter.events and c[-1] == 'intercept'] + selected
                weights = betas.loc[selected].to_numpy()
                basis_values = event_basis.loc[:, selected].to_numpy()
                values = basis_values.dot(np.nan_to_num(weights, nan=0.))
                values[(basis_values != 0.).astype(int).dot(np.isnan(weights).astype(int)) > 0] = np.nan
                result = pd.DataFrame(values, columns=betas.columns)
                result['event_type'] = event
                result['covariate'] = str(column[2]) if len(covariates) == 1 else f'{column[1]}:{column[2]}'
                result['t'] = event_rows.to_numpy()
                # get_basis_functions may contain repeated time rows for separate
                # covariates. Only use rows for this column's covariate.
                row_covariates = event_basis.index.get_level_values('covariate')
                result = result.loc[row_covariates == column[1]]
                output.append(result)
        if not output:
            return pd.DataFrame()
        result = pd.concat(output, ignore_index=True)
        return result.set_index(['event_type', 'covariate', 't']) if index else result

    def get_basisset_timecourses(self):
        """Collect every acquisition and every ROI-specific ridge fitter."""
        if not hasattr(self, 'fitters'):
            if self.fit_type == 'ridge':
                self.fitters = self.sub_df.copy()
            else:
                self.format_fitters()
        output = []
        identities = list(self.fitters.index.names)
        for key, row in self.fitters.iterrows():
            key = key if isinstance(key, tuple) else (key,)
            curves = [self.get_curves_from_fitter(fitter, index=True, icpt=self.conf_icpt)
                      for fitter in row if fitter is not None]
            curves = pd.concat(curves, axis=1)
            if not curves.columns.is_unique:
                raise ValueError('Duplicate ROI columns from basis fitters')
            curves = curves.reset_index()
            for name, value in zip(identities, key):
                curves[name] = value
            output.append(curves)
        self.sub_basis = pd.concat(output, ignore_index=True).set_index(
            [identities[0], 'event_type'] + identities[1:] + ['covariate', 't'])

    def get_predictions_per_event(self):
        """Keep full acquisition identities when collecting fitted predictions."""
        if not hasattr(self, 'fitters'):
            self.format_fitters()
        event_predictions, full_predictions = [], []
        identities = list(self.fitters.index.names)
        for key, row in self.fitters.iterrows():
            key = key if isinstance(key, tuple) else (key,)
            fitter = row.iloc[0]
            full = fitter.X.dot(self._prediction_betas(fitter)).copy()
            full.columns = fitter.input_signal.columns
            full.index.name = 'time'
            full = full.reset_index()
            events = self.get_event_predictions_from_fitter(fitter, intercept=self.conf_icpt)
            for name, value in zip(identities, key):
                full[name] = value
                events[name] = value
            full_predictions.append(full)
            if not events.empty:
                event_predictions.append(events)
        self.sub_pred_full = pd.concat(full_predictions, ignore_index=True).set_index(identities + ['time'])
        self.ev_predictions = (pd.concat(event_predictions, ignore_index=True)
            .set_index([identities[0], 'event_type'] + identities[1:] + ['t'])) if event_predictions else pd.DataFrame()


    def timecourses_condition(self):
        """Fetch profiles and calculate equal-subject group means and uncertainty.

        tc_sem/tc_std retain their within-subject, across-run interpretation.
        sem_condition/std_condition describe the distribution of subject means.
        Raw FIR profiles remain unchanged; *_interp attributes are display-only.
        """
        if hasattr(self, 'fit_diagnostics') and self.fitters.empty:
            raise RuntimeError('No successfully fitted runs; inspect fit_diagnostics and all_fitters')
        self.profile_covariate = self.covariates if isinstance(self.covariates, str) else 'intercept'
        if self.fit_type == 'ols':
            # Avoid nideconv's reset_index().mean(), which averages string run IDs.
            if any(len(getattr(row.iloc[0], 'dropped_zero_regressors', []))
                   for _, row in getattr(self, 'fitters', pd.DataFrame()).iterrows()):
                parts = {key: self._profiles_from_fitter(row.iloc[0], self.osf)
                         for key, row in self.fitters.iterrows()}
                profiles = pd.concat(parts, names=list(self.fitters.index.names))
            else:
                profiles = self.model.get_timecourses()
        else:
            profiles = self.tc_subjects.copy()
            self.fitters = self.sub_df.copy()
        profiles = self.check_for_run_index(profiles)
        names = ['event_type' if name == 'event type' else name for name in profiles.index.names]
        profiles.index = profiles.index.set_names(names)
        identity = [key for key in ('subject', 'session', 'task', 'run') if key in names]
        self.tc_subjects = self.change_name_set_index(
            profiles, index=identity + ['event_type', 'covariate', 'time'])
        if any(not pd.api.types.is_numeric_dtype(dtype) for dtype in self.tc_subjects.dtypes):
            raise ValueError('Time-course signal columns must be numeric; remove acquisition metadata from signal columns')
        within = [key for key in identity if key != 'run'] + ['event_type', 'covariate', 'time']
        self.obj_grouped = self.tc_subjects.groupby(level=within)
        self.tc_mean = self.obj_grouped.mean()
        self.tc_sem = self.obj_grouped.sem()
        self.tc_std = self.obj_grouped.std()
        self.grouper = [key for key in within if key != 'subject']
        group = self.tc_mean.groupby(level=self.grouper)
        self.tc_condition = group.mean()
        self.sem_condition = group.sem()
        self.std_condition = group.std()
        self.time = np.sort(self.tc_condition.index.get_level_values('time').unique().to_numpy())
        if getattr(self, 'fir_events', None) or self.basis_sets == 'fir':
            self.tc_condition_interp = self.interpolate_fir_condition(self.tc_condition)
            self.tc_subjects_interp = self.interpolate_fir_subjects()
        try:
            self.rsq_ = self.model.get_rsq()
        except (AttributeError, TypeError, ValueError):
            self.rsq_ = None
        if self.fit_type == 'ols':
            self.fitters = self.model._get_response_fitters()
            self.format_fitters()
            self.get_predictions_per_event()

    @classmethod
    def check_for_run_index(self, df, loc=1):

        """check_for_run_index

        Ensures that the run index exists in a dataframe, adding a `"run"` column if missing.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe.
        loc : int, optional
            Position to insert the `"run"` column, by default `1`.

        Returns
        -------
        pd.DataFrame
            Dataframe with `"run"` column added if missing.

        Example
        ----------
        .. code-block:: python

            df_checked = NideconvFitter.check_for_run_index(df)
        """

        # reset index
        old_idx = list(df.index.names)

        tmp = df.copy()
        if "run" not in old_idx:
            tmp = df.reset_index()
            old_idx.insert(loc, "run")
            tmp["run"] = 1

            tmp.set_index(old_idx, inplace=True)

        # return
        return tmp

    def change_name_set_index(self, df, index=["subject", "event_type", "covariate", "time"]):

        """change_name_set_index

        Renames the `"event type"` column to `"event_type"` and resets the dataframe index.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe.
        index : list or str, optional
            List of index levels to set, by default `["subject", "event_type", "covariate", "time"]`.

        Returns
        -------
        pd.DataFrame
            Reformatted dataframe with updated index.

        Example
        ----------
        .. code-block:: python

            formatted_df = fitter.change_name_set_index(df)
        """

        # reset index
        tmp = df.reset_index()

        # remove space in event type column
        if "event type" in list(tmp.columns):
            tmp = tmp.rename(columns={"event type": "event_type"})

        # set index
        if isinstance(index, str):
            index = [index]

        tmp.set_index(index, inplace=True)

        # return
        return tmp

    def define_model(self, **kwargs):

        """define_model

        Defines the `nideconv.GroupResponseFitter` model and initializes it with the appropriate parameters.

        Parameters
        ----------
        **kwargs : dict
            Additional arguments passed to the `GroupResponseFitter`.

        Returns
        -------
        None
            Stores the initialized model in the object attribute `model`.

        Example
        ----------
        .. code-block:: python

            fitter.define_model()
        """


        identities = [key for key in ('subject', 'session', 'task', 'run') if key in self.func.index.names]
        if any(key not in self.used_onsets.index.names for key in identities):
            raise ValueError('Onsets must include the same acquisition identifiers as functional data')
        signal_keys = self.func.reset_index()[identities].drop_duplicates()
        event_keys = self.used_onsets.reset_index()[identities].drop_duplicates()
        matched = signal_keys.merge(event_keys, on=identities, how='left', indicator=True)
        missing = matched.loc[matched['_merge'] == 'left_only', identities]
        if not missing.empty:
            raise ValueError(f'Functional acquisitions without matching events: {missing.to_dict("records")}')
        self.model = nd.GroupResponseFitter(
            self.func,
            self.used_onsets,
            input_sample_rate=self.fs,
            confounds=self.confounds,
            oversample_design_matrix=self.osf,
            add_intercept=self.conf_icpt,
            concatenate_runs=None,
            **kwargs
        )

    def add_event(self, *args, **kwargs):
        """add_event

        Adds a new event to the response fitter.

        Parameters
        ----------
        *args : tuple
            Positional arguments passed to the `add_event` function.
        **kwargs : dict
            Additional keyword arguments for defining the event.

        Returns
        -------
        None
            Updates the fitted model with the new event.

        Example
        ----------
        .. code-block:: python

            fitter.add_event("stimulus", basis_set="fourier", n_regressors=4)
        """

        self.model.add_event(*args, **kwargs)

    def define_events(self):

        """define_events

        Adds events to the deconvolution model, specifying the basis set and regression parameters.

        Returns
        -------
        None
            Stores event definitions in the fitted model.

        Example
        ----------
        .. code-block:: python

            fitter.define_events()
        """

        if getattr(self, 'event_models', None) is not None:
            utils.verbose('Selected per-event basis functions (joint model):', self.verbose)
            for event, spec in self.event_models_.items():
                utils.verbose(
                    f"  {event}: basis={spec['basis_sets']}; "
                    f"n_regressors={spec['n_regressors']}; "
                    f"interval={spec['interval']}; duration={spec['duration']}",
                    self.verbose)
        else:
            utils.verbose(
                f"Selected '{self.basis_sets}'-basis sets (with {self.n_regressors} regressors)",
                self.verbose)

        # define events
        self.cond = utils.get_unique_ids(self.used_onsets, id="event_type")
        self.run_ids = utils.get_unique_ids(self.used_onsets, id="run")

        # Every event is added to the same acquisition-specific regression.
        if self.fit_type.lower() == 'ols':
            for ix, event in enumerate(self.cond):
                spec = self._event_spec(event, ix)
                covariates = spec['covariates']
                covariate_names = [covariates] if isinstance(covariates, str) else covariates
                # Upstream otherwise duplicates an explicitly supplied intercept.
                add_intercept = spec['add_intercept'] and (
                    covariate_names is None or 'intercept' not in covariate_names)
                self.model.add_event(str(event), basis_set=spec['basis_sets'],
                    n_regressors=spec['n_regressors'], interval=spec['interval'],
                    add_intercept=add_intercept, covariates=covariates,
                    **({'fir_bin_mode': spec['fir_bin_mode']} if spec['fir_bin_mode'] != 'legacy' else {}))

        else:
            utils.verbose('Setting up models ridge-regression', self.verbose)
            identities = [key for key in ('subject', 'session', 'task', 'run') if key in self.func.index.names]
            self.sub_ids = self.func.index.get_level_values('subject').unique().tolist()
            self.run_ids = self.func.index.get_level_values('run').unique().tolist()
            fitter_rows, profiles, predictions, event_predictions = [], [], [], []
            for key, acquisition in self.func.groupby(level=identities, sort=False):
                key = key if isinstance(key, tuple) else (key,)
                selection = dict(zip(identities, key))
                fitters, profile_columns, prediction_columns, event_columns = {}, [], [], []
                for column in acquisition.columns:
                    fitter = self.make_response_fitter(acquisition[[column]], selection=selection)
                    fitters[column] = fitter
                    if not self.do_fit:
                        continue
                    self._fit_ridge_response(fitter)
                    profile_columns.append(fitter.get_timecourses())
                    prediction = fitter.predict_from_design_matrix().copy()
                    prediction.columns = [column]
                    prediction_columns.append(prediction)
                    events = self.get_event_predictions_from_fitter(fitter, intercept=self.conf_icpt)
                    if not events.empty:
                        event_columns.append(events.set_index(['event_type', 't'])[[column]])
                fitter_rows.append(pd.DataFrame([fitters], index=pd.MultiIndex.from_tuples([key], names=identities)))
                for columns, destination in [(profile_columns, profiles), (prediction_columns, predictions), (event_columns, event_predictions)]:
                    if not columns:
                        continue
                    result = pd.concat(columns, axis=1)
                    if destination is predictions:
                        result.index.name = 'time'
                    result = result.reset_index()
                    for name, value in selection.items():
                        result[name] = value
                    destination.append(result)
            self.sub_df = pd.concat(fitter_rows)
            if profiles:
                self.tc_subjects = pd.concat(profiles, ignore_index=True).set_index(identities + ['event type', 'covariate', 'time'])
            else:
                self.tc_subjects = pd.DataFrame()
            self.sub_ev_df = (pd.concat(event_predictions, ignore_index=True).set_index(
                [identities[0], 'event_type'] + identities[1:] + ['t'])) if event_predictions else pd.DataFrame()
            self.ev_predictions = self.sub_ev_df.copy()
            self.sub_pred_full = (pd.concat(predictions, ignore_index=True).set_index(
                identities + ['time'])) if predictions else pd.DataFrame()

    @staticmethod
    def _fit_ridge_response(fitter, cv=20, alphas=None):
        """Fit one signal with an unpenalized acquisition baseline.

        sklearn estimates the baseline within each CV training fold. Restore
        its coefficient to the original design so downstream predictions and
        event coefficient labels retain their existing layout.
        """
        X = fitter.X.to_numpy(dtype=float)
        y = fitter.input_signal.to_numpy(dtype=float)
        if y.ndim != 2 or y.shape[1] != 1 or len(y) < 2:
            raise ValueError('Ridge response fitting requires one signal and at least two samples')
        if not np.isfinite(X).all() or not np.isfinite(y).all():
            raise ValueError('Ridge inputs must contain finite numeric values')
        baseline = np.asarray([column[0] == 'confounds' and column[1] == 'intercept'
                               for column in fitter.X.columns])
        if baseline.sum() > 1:
            raise ValueError('Expected at most one acquisition baseline')
        if baseline.any() and not np.all(X[:, baseline] == 1.):
            raise ValueError('The acquisition baseline must be a constant column of ones')
        predictors = X[:, ~baseline]
        alphas = np.logspace(7, 0, 20) if alphas is None else alphas
        fitter.rcv = linear_model.RidgeCV(alphas=alphas, fit_intercept=bool(baseline.any()),
                                         cv=min(cv, len(y)))
        fitter.rcv.fit(predictors, y[:, 0])
        betas = np.zeros((X.shape[1], 1))
        betas[~baseline, 0] = np.asarray(fitter.rcv.coef_).ravel()
        if baseline.any():
            betas[baseline, 0] = fitter.rcv.intercept_
        prediction = X.dot(betas)
        fitter.betas = betas
        fitter._send_betas_to_regressors()
        fitter._residuals = pd.DataFrame(y - prediction, index=fitter.input_signal.index,
                                        columns=fitter.input_signal.columns)
        fitter.sse = (fitter._residuals ** 2).sum(axis=0)
        fitter.ssquares = fitter.sse.copy()

    def make_response_fitter(self, data, run=None, cov=None, subject=None, selection=None):
        """Build one acquisition-specific model, with event-specific covariates."""
        selection = dict(selection or {})
        for field, value in [('subject', subject), ('run', run)]:
            if value is not None:
                selection[field] = value
        for field in ('subject', 'session', 'task', 'run'):
            if field in data.index.names and field not in selection:
                values = data.index.get_level_values(field).unique()
                if len(values) != 1:
                    raise ValueError(f'Expected one {field} in response fitter input')
                selection[field] = values[0]
        if self._select_acquisition(self.used_onsets, selection).empty:
            raise ValueError(f'No events match functional acquisition {selection}')
        model = nd.ResponseFitter(input_signal=data, sample_rate=self.fs,
            add_intercept=self.conf_icpt, oversample_design_matrix=self.osf)
        if self.confounds is not None:
            confounds = self._select_acquisition(self.confounds, selection)
            if len(confounds) != len(data) or not confounds.index.equals(data.index):
                raise ValueError('Confounds and functional data must have matching acquisition/time indices')
            model.add_confounds('confounds', confounds.to_numpy())
        for ix, event in enumerate(self.cond):
            event_frame = self._select_acquisition(self.used_onsets, selection).reset_index()
            event_frame = event_frame.loc[event_frame['event_type'] == event]
            if event_frame.empty:
                continue
            spec = self._event_spec(event, ix)
            event_cov = spec['covariates']
            if cov is not None:
                event_cov = cov
            covariates = event_frame[event_cov] if event_cov is not None else None
            if isinstance(covariates, pd.Series):
                covariates = covariates.to_frame()
            intercept = spec['add_intercept']
            if covariates is not None:
                covariates = covariates.reset_index(drop=True).copy()
                if intercept and 'intercept' not in covariates:
                    covariates.insert(0, 'intercept', 1.)
            duration = event_frame['duration'].to_numpy() if 'duration' in event_frame else None
            if duration is not None and not np.isfinite(duration).all():
                duration = None
            model.add_event(str(event), onsets=event_frame['onset'].to_numpy(),
                basis_set=spec['basis_sets'], n_regressors=spec['n_regressors'], interval=spec['interval'],
                covariates=covariates, durations=duration,
                **({'fir_bin_mode': spec['fir_bin_mode']} if spec.get('fir_bin_mode', 'legacy') != 'legacy' else {}))
        return model

    @staticmethod
    def _select_acquisition(frame, selection):
        """Select exact labels without numeric coercion or substring matching."""
        selected = frame
        for name, value in selection.items():
            if name in selected.index.names:
                selected = selected.loc[selected.index.get_level_values(name) == value]
            elif name in selected.columns:
                selected = selected.loc[selected[name] == value]
            elif len(selection) and name in ('session', 'task'):
                raise ValueError(f'Acquisition metadata {name!r} is missing from input')
        return selected

    def make_onsets_for_response_fitter(self, i, run=None, subject=None, selection=None):
        selection = dict(selection or {})
        if subject is not None:
            selection['subject'] = subject
        if run is not None:
            selection['run'] = run
        frame = self._select_acquisition(self.used_onsets, selection).reset_index()
        for key in ('subject', 'session', 'task', 'run'):
            if key in frame and frame[key].nunique() > 1:
                raise ValueError(f'Onsets span multiple {key} labels; specify the acquisition')
        return frame.loc[frame['event_type'] == i, 'onset']

    def fit(self):
        """Fit OLS runs independently, preserving models and failure diagnostics.

        Rank is checked before fitting. A skipped singular run never contributes
        coefficients, predictions or group averages. Other fitting failures
        always raise; the exception retains this object as error.fitter.
        """
        if self.fit_type == 'ols':
            if not hasattr(self.model, 'all_response_fitters'):
                self.model.all_response_fitters = self.model.response_fitters.copy()
            self.model.response_fitters = self.model.all_response_fitters.copy()
            self.format_fitters()
            self.all_fitters = self.fitters.copy()
            records, successful, failures = [], [], []
            for key, row in self.all_fitters.iterrows():
                fitter = row.iloc[0]
                record = dict(n_samples=fitter.X.shape[0], n_columns=fitter.X.shape[1],
                              rank=None, nullity=None, condition_number=np.nan,
                              singular_values=None, zero_columns=None, dropped_columns=(),
                              n_columns_fitted=fitter.X.shape[1],
                              status='failed', error='')
                try:
                    X = fitter.X.to_numpy(dtype=float)
                    if not np.isfinite(X).all():
                        raise ValueError('Design contains nonfinite values')
                    zero = np.all(X == 0., axis=0)
                    dropped = zero if getattr(self, 'drop_zero_regressors', False) else np.zeros(X.shape[1], dtype=bool)
                    fitter.dropped_zero_regressors = fitter.X.columns[dropped].copy()
                    active = X[:, ~dropped]
                    record.update(dropped_columns=tuple(fitter.X.columns[dropped]),
                                  n_columns_fitted=active.shape[1])
                    singular = np.linalg.svd(active, compute_uv=False)
                    # Match the cutoff used by nideconv's lstsq(rcond=-1).
                    tolerance = np.finfo(float).eps * singular[0] if len(singular) else 0.
                    rank = int(np.sum(singular > tolerance))
                    record.update(rank=rank, nullity=active.shape[1] - rank,
                        original_nullity=X.shape[1] - rank,
                        singular_values=tuple(singular),
                        zero_columns=tuple(fitter.X.columns[np.all(X == 0., axis=0)]),
                        condition_number=(float(singular[0] / singular[-1])
                            if rank == active.shape[1] and len(singular) and singular[-1] > 0 else np.inf))
                    if active.shape[1] == 0:
                        raise ValueError('No nonzero design columns remain')
                    if rank < active.shape[1]:
                        record.update(status='rank_deficient', error='Rank-deficient design')
                    else:
                        original_design = fitter.X
                        original_events = fitter.events
                        try:
                            fitter.X = original_design.loc[:, ~dropped]
                            active_events = set(fitter.X.columns.get_level_values(0))
                            fitter.events = {name: event for name, event in original_events.items()
                                             if name in active_events}
                            fitter.fit(type='ols')
                            if not hasattr(fitter, 'betas') or not np.isfinite(np.asarray(fitter.betas, dtype=float)).all():
                                raise ValueError('Missing or nonfinite fitted coefficients')
                            coefficients = fitter.betas.reindex(original_design.columns)
                        finally:
                            fitter.X = original_design
                            fitter.events = original_events
                        fitter.betas = coefficients
                        fitter._send_betas_to_regressors()
                        if dropped.any():
                            fitter._residuals = fitter.input_signal - fitter.X.dot(self._prediction_betas(fitter))
                        record['status'] = 'success'
                except Exception as error:
                    record.update(status='failed', error=str(error))
                records.append(record)
                successful.append(record['status'] == 'success')
                if record['status'] != 'success':
                    failures.append(f"{key}: {record['error']}")
            self.fit_diagnostics = pd.DataFrame(records, index=self.all_fitters.index)
            self.failed_runs = self.fit_diagnostics.loc[self.fit_diagnostics['status'] != 'success'].copy()
            self.fitters = self.all_fitters.loc[successful].copy()
            # Keep native nideconv aggregation restricted to successful runs.
            # Every original ResponseFitter remains accessible in all_fitters.
            self.model.response_fitters = self.model.all_response_fitters.loc[successful].copy()
            self.failed_fitters = self.all_fitters.loc[np.logical_not(successful)].copy()
            policy = getattr(self, 'on_rank_deficient', 'raise')
            fatal = (self.fit_diagnostics['status'] == 'failed').any()
            if failures and (policy == 'raise' or fatal):
                error = RuntimeError('Response fitting failed: ' + '; '.join(failures))
                error.fitter = self
                raise error
            if failures:
                warnings.warn(f'Skipped {len(failures)} rank-deficient run(s); inspect fit_diagnostics and all_fitters',
                              RuntimeWarning)
        else:
            if not self.do_fit:
                self.do_fit = True
                self.define_events()
            self.fitters = self.sub_df.copy()
        utils.verbose('Fitting completed', self.verbose)

    def plot_average_per_event(
        self,
        add_offset: bool = True,
        axs=None,
        title: str = "Average HRF across events",
        save_as: str = None,
        error_type: str = "sem",
        ttp: bool = False,
        ttp_lines: bool = False,
        ttp_labels: list = None,
        events: list = None,
        fwhm: bool = False,
        fwhm_lines: bool = False,
        fwhm_labels: list = None,
        inset_ttp: list = [0.75, 0.65, 0.3],
        inset_fwhm: list = [0.75, 0.65, 0.3],
        reduction_factor: float = 1.3,
        **kwargs):

        """plot_average_per_event

        Plot the average across runs and voxels for each event in your data. Allows the option to have time-to-peak or
        full-width half max (FWHM) plots as insets. This makes the most sense if you have multiple events, otherwise you have
        1 bar..

        Parameters
        ----------
        add_offset: bool, optional
            Shift the HRFs to have the baseline at zero, by default True. Theoretically, this should already happen if your
            baseline is estimated properly, but for visualization and quantification purposes this is alright
        axs: <AxesSubplot:>, optional
            Matplotlib axis to store the figure on, by default None
        title: str, optional
            Plot title, by default None, by default "Average HRF across events"
        save_as: str, optional
            Save the plot as a file, by default None
        error_type: str, optional
            Which error type to use across runs/voxels, by default "sem"
        ttp: bool, optional
            Plot the time-to-peak on the inset axis, by default False
        ttp_lines: bool, optional
            Plot lines on the original axis with HRFs to indicate the maximum amplitude, by default False
        ttp_labels: list, optional
            Which labels to use for the inset axis; this can be different than your event names (e.g., if you want to round
            numbers), by default None
        events: list, optional
            List that decides the order of the events to plot, by default None. By default, it takes the event names, but
            sometimes you want to flip around the order.
        fwhm: bool, optional
            Plot the full-width half-max (FWHM) on the inset axis, by default False
        fwhm_lines: bool, optional
            Plot lines on the original axis with HRFs to indicate the maximum amplitude, by default False
        fwhm_labels: list, optional
            Which labels to use for the inset axis; this can be different than your event names (e.g., if you want to round
            numbers), by default None
        inset_ttp: list, optional
            Where to put your TTP-axis, by default [0.75, 0.65, 0.3]. Height will be scaled by the number of events
        inset_fwhm: list, optional
            Where to put your FWHM-axis, by default [0.75, 0.65, 0.3, 0.3]. Width will be scaled by the number of events
        reduction_factor: float, optional
            Reduction factor of the font size in the inset axis, by default 1.3

        Example
        ----------
        .. code-block:: python

            # do the fitting
            nd_fit = fitting.NideconvFitter(
                df_ribbon, # dataframe with functional data
                df_onsets,  # dataframe with onsets
                basis_sets='canonical_hrf_with_time_derivative',
                TR=0.105,
                interval=[-3,17],
                add_intercept=True,
                verbose=True)

            # plot TTP with regular events + box that highlights stimulus onset
            fig,axs = plt.subplots(figsize=(8,8))
            nd_fit.plot_average_per_event(
                xkcd=plot_xkcd,
                x_label="time (s)",
                y_label="magnitude (%)",
                add_hline='default',
                ttp=True,
                lim=[0,6],
                ticks=[0,3,6],
                ttp_lines=True,
                y_label2="size (°)",
                x_label2="time-to-peak (s)",
                title="regular events",
                ttp_labels=[f"{round(float(ii),2)}°" for ii in nd_fit.cond],
                add_labels=True,
                fancy=True,
                cmap='inferno')
            # plot simulus onset
            axs.axvspan(0,1, ymax=0.1, color="#cccccc")

            # plot FWHM and flip the events
            nd_fit.plot_average_per_event(
                x_label="time (s)",
                y_label="magnitude (%)",
                add_hline='default',
                fwhm=True,
                fwhm_lines=True,
                lim=[0,5],
                ticks=[i for i in range(6)],
                fwhm_labels=[f"{round(float(ii),2)}°" for ii in nd_fit.cond[::-1]],
                events=nd_fit.cond[::-1],
                add_labels=True,
                x_label2="size (°)",
                y_label2="FWHM (s)",
                fancy=True,
                cmap='inferno')
        """

        self.__dict__.update(kwargs)
        inset_ttp, inset_fwhm = list(inset_ttp), list(inset_fwhm)

        if axs is None:
            if not hasattr(self, "figsize"):
                self.figsize = (8, 8)
            _, axs = plt.subplots(figsize=self.figsize)

        if not hasattr(self, "tc_condition"):
            self.timecourses_condition()

        if not hasattr(self, 'tc_subjects'):
            self.timecourses_condition()
        frame = self.tc_subjects
        covariate = getattr(self, 'profile_covariate', 'intercept')
        frame = frame.xs(covariate, level='covariate')
        # Avoid averaging distinct sessions/tasks implicitly.
        for field in ('session', 'task'):
            if field in frame.index.names and frame.index.get_level_values(field).nunique() > 1:
                raise ValueError(f'Select one {field} before plotting an event average')
        per_subject = frame.mean(axis=1).groupby(level=['subject', 'event_type', 'time']).mean()
        if add_offset:
            per_subject = per_subject.groupby(level=['subject', 'event_type'], group_keys=False).apply(
                lambda profile: profile - profile.sort_index(level='time').iloc[0])
        grouped = per_subject.groupby(level=['event_type', 'time'])
        self.avg_across_runs_voxels = grouped.mean()
        self.event_indices = None
        events = self.cond if events is None else list(events)
        self.event_avg, self.event_sem, self.event_std = [], [], []
        self.time = np.sort(per_subject.index.get_level_values('time').unique().to_numpy())
        for event in events:
            self.event_avg.append(grouped.mean().xs(event).reindex(self.time).to_numpy())
            self.event_sem.append(grouped.sem().xs(event).reindex(self.time).to_numpy())
            self.event_std.append(grouped.std().xs(event).reindex(self.time).to_numpy())
        # Keep historical intermediate attributes available to callers.
        self.tc_df = frame
        self.avg_across_runs = frame.groupby(level=['event_type', 'time']).mean()

        # reorder base on indices again
        if isinstance(self.event_indices, list):
            for tt, gg in zip(
                ["avg", "sem", "std"],
                    [self.event_avg, self.event_sem, self.event_std]):
                reordered = [gg[i] for i in self.event_indices]
                setattr(self, f"event_{tt}", reordered)

        # decide error type
        if error_type == "sem":
            self.use_error = self.event_sem.copy()
        elif error_type == "std":
            self.use_error = self.event_std.copy()
        else:
            self.use_error = None

        # plot
        plotter = plotting.LazyLine(
            self.event_avg,
            xx=self.time,
            axs=axs,
            error=self.use_error,
            title=title,
            save_as=save_as,
            **kwargs)

        if hasattr(self, "font_size"):
            self.old_font_size = plotter.font_size
            self.old_label_size = plotter.label_size
            self.font_size = plotter.font_size / reduction_factor
            self.label_size = plotter.label_size / reduction_factor

        if ttp:

            # make bar plot, use same color-coding
            if isinstance(ttp_labels, (list, np.ndarray)):
                ttp_labels = ttp_labels
            else:
                ttp_labels = events

            # scale height by nr of events
            if len(inset_ttp) < 4:
                inset_ttp.append(len(ttp_labels) * 0.05)

            left, bottom, width, height = inset_ttp
            ax2 = axs.inset_axes([left, bottom, width, height])
            self.plot_ttp(
                df=self.avg_across_runs_voxels,
                axs=ax2,
                hrf_axs=axs,
                ttp_labels=ttp_labels,
                ttp_lines=ttp_lines,
                font_size=self.font_size,
                label_size=self.label_size,
                sns_offset=2
            )

        if fwhm:

            # make bar plot, use same color-coding
            if isinstance(fwhm_labels, (list, np.ndarray)):
                fwhm_labels = fwhm_labels
            else:
                fwhm_labels = events

            # scale height by nr of events
            if len(inset_fwhm) < 4:
                inset_fwhm.insert(2, len(fwhm_labels) * 0.05)

            left, bottom, width, height = inset_fwhm
            ax2 = axs.inset_axes([left, bottom, width, height])
            self.plot_fwhm(
                df=self.avg_across_runs_voxels,
                axs=ax2,
                hrf_axs=axs,
                fwhm_labels=fwhm_labels,
                fwhm_lines=fwhm_lines,
                **dict(
                    kwargs,
                    font_size=self.font_size,
                    label_size=self.label_size,
                    sns_offset=2))

        if hasattr(self, "old_font_size"):
            self.font_size = self.old_font_size

        if hasattr(self, "old_label_size"):
            self.label_size = self.old_label_size

    def plot_ttp(
            self,
            df=None,
            axs=None,
            hrf_axs=None,
            ttp_lines=False,
            ttp_labels=None,
            figsize=(8, 8),
            ttp_ori='h',
            split="event_type",
            lines_only=False,
            par_kw={},
            **kwargs):

        if not isinstance(axs, mpl.axes._axes.Axes):
            _, axs = plt.subplots(figsize=figsize)

        # unstack series | assume split is on event_type
        if isinstance(df, pd.Series):
            df = pd.DataFrame(df).unstack(level=split)
            x_lbl = split
            x_data = self.cond
        else:
            x_lbl = "vox"
            x_data = list(df.columns)

        self.df_pars = HRFMetrics(
            df,
            TR=self.TR,
            **par_kw
        ).return_metrics()
        self.df_pars[x_lbl] = x_data

        # decide on color based on split
        if not hasattr(self, "color"):
            if not hasattr(self, "cmap") and "cmap" not in list(kwargs.keys()):
                cmap = "viridis"
            else:
                if "cmap" in list(kwargs.keys()):
                    cmap = kwargs["cmap"]
                else:
                    cmap = self.cmap
            colors = sns.color_palette(cmap, self.df_pars.shape[0])
        else:
            colors = self.color

        if ttp_lines:
            # heights need to be adjusted for by axis length
            if not isinstance(hrf_axs, mpl.axes._axes.Axes):
                raise ValueError(
                    "Need an axes-object containing HRF profiles to draw lines on")

            ylim = hrf_axs.get_ylim()
            tot = sum(list(np.abs(ylim)))
            start = (0 - ylim[0]) / tot

            for ix, ii in enumerate(self.df_pars["time_to_peak"].values):
                hrf_axs.axvline(
                    ii,
                    ymin=start,
                    ymax=self.df_pars["magnitude"].values[ix] / tot + start,
                    color=colors[ix],
                    linewidth=0.5)

        if not lines_only:
            self.ttp_plot = plotting.LazyBar(
                data=self.df_pars,
                x=x_lbl,
                y="time_to_peak",
                labels=ttp_labels,
                sns_ori=ttp_ori,
                axs=axs,
                # error=None,
                **kwargs
            )

    def plot_fwhm(
            self,
            df,
            axs=None,
            hrf_axs=None,
            fwhm_lines=False,
            fwhm_labels=None,
            split="event_type",
            figsize=(8, 8),
            fwhm_ori='v',
            par_kw={},
            **kwargs):

        if axs is None:
            fig, axs = plt.subplots(figsize=figsize)

        # unstack series | assume split is on event_type
        if isinstance(df, pd.Series):
            df = pd.DataFrame(df).unstack(level=split)
            x_lbl = split
            x_data = self.cond
        else:
            x_lbl = "vox"
            x_data = list(df.columns)

        self.df_pars = HRFMetrics(
            df,
            TR=self.TR,
            **par_kw
        ).return_metrics()
        self.df_pars[x_lbl] = x_data

        # decide on color based on split
        if not hasattr(self, "color"):
            if not hasattr(self, "cmap"):
                cmap = "viridis"
            else:
                cmap = self.cmap
            colors = sns.color_palette(cmap, self.df_pars.shape[0])
        else:
            colors = self.color

        if fwhm_lines:
            # heights need to be adjusted for by axis length
            xlim = hrf_axs.get_xlim()
            tot = sum(list(np.abs(xlim)))
            for ix, ii in enumerate(self.df_pars["fwhm"].values):
                start = (
                    self.df_pars["half_rise_time"].values[ix] - xlim[0]) / tot
                half_m = self.df_pars["half_max"].values[ix]
                hrf_axs.axhline(
                    half_m,
                    xmin=start,
                    xmax=start + ii / tot,
                    color=colors[ix],
                    linewidth=0.5)

        self.fwhm_labels = fwhm_labels
        self.fwhm_plot = plotting.LazyBar(
            data=self.df_pars,
            x=x_lbl,
            y="fwhm",
            palette=colors,
            sns_ori=fwhm_ori,
            axs=axs,
            error=None,
            **kwargs)


    def _event_plot_statistics(self, add_offset=False):
        frame = self.tc_subjects.xs(getattr(self, 'profile_covariate', 'intercept'), level='covariate')
        for field in ('session', 'task'):
            if field in frame.index.names and frame.index.get_level_values(field).nunique() > 1:
                raise ValueError(f'Select one {field} before plotting profiles')
        subject = frame.groupby(level=['subject', 'event_type', 'time']).mean()
        if add_offset:
            subject = subject.groupby(level=['subject', 'event_type'], group_keys=False).apply(
                lambda group: group - group.sort_index(level='time').iloc[0])
        group = subject.groupby(level=['event_type', 'time'])
        return group.mean(), group.sem(), group.std()

    def plot_average_per_voxel(
            self,
            add_offset: bool = True,
            axs=None,
            n_cols: int = 4,
            fig_kwargs: dict = {},
            figsize: tuple = None,
            make_figure: bool = True,
            labels: list = None,
            save_as: str = None,
            sharey: bool = False,
            skip_x: list = None,
            skip_y: list = None,
            title: list = None,
            **kwargs):
        """plot_average_per_voxel

        Plot the average across runs for each voxel in your dataset. Generally, this plot is used to plot HRFs across depth.
        If you have multiple events, we'll create a grid of `n_cols` wide (from which the rows are derived), with the HRFs for
        each event in the subplot. The legend will be put in the first subplot. If you only have 1 event, you can say
        `n_cols=None` to put the average across events for all voxels in 1 plot.

        Parameters
        ----------
        add_offset: bool, optional
            Shift the HRFs to have the baseline at zero, by default True. Theoretically, this should already happen if your
            baseline is estimated properly, but for visualization and quantification purposes this is alright
        axs: <AxesSubplot:>, optional
            Matplotlib axis to store the figure on, by default None
        n_cols: int, optional
            Decides the number of subplots on the x-axis, by default 4. If you have 1 event, specify `n_cols=None`
        wspace: float, optional
            Decide the width between subplots, by default 0
        figsize: tuple, optional
            Figure size, by default (24,5*nr_of_rows) or (8,8) if `n_cols=None`
        make_figure: bool, optional
            Actually create the plot or just fetch the data across depth, by default True
        labels: list, optional
            Which labels to use for the inset axis; this can be different than your event names (e.g., if you want to round
            numbers), by default None
        save_as: str, optional
            Save to file, by default None
        sharey: bool, optional
            Save all y-axes the same, by default False. Can be nice if you want to see the progression across depth

        Example
        ----------
        .. code-block:: python

            nd_fit.plot_average_per_voxel(
                labels=[f"{round(float(ii),2)} dva" for ii in nd_fit.cond],
                wspace=0.2,
                cmap="inferno",
                line_width=2,
                font_size=font_size,
                label_size=16,
                sharey=True
            )
        """

        if not hasattr(self, "tc_condition"):
            self.timecourses_condition()

        plot_mean, plot_sem, _ = self._event_plot_statistics(add_offset=add_offset)
        cols = list(plot_mean.columns)
        cols_id = np.arange(0, len(cols))
        if n_cols is not None:

            if isinstance(axs, (list, np.ndarray)):
                if len(axs) != len(cols):
                    raise ValueError(
                        f"For this option {len(cols)} axes are required, {len(axs)} were specified")
            else:
                # initiate figure
                if len(cols) > 10:
                    raise Exception(
                        f"{len(cols)} were requested. Maximum number of plots is set to 30")

                n_rows = int(np.ceil(len(cols) / n_cols))
                if not isinstance(figsize, tuple):
                    figsize = (24, 5 * n_rows)

                fig = plt.figure(figsize=figsize, constrained_layout=True)
                gs = fig.add_gridspec(n_rows, n_cols, **fig_kwargs)
        else:
            if not isinstance(figsize, tuple):
                figsize = (8, 8)

        self.all_voxels_in_event = []
        self.all_error_in_voxels = []
        for ix, col in enumerate(cols):

            # fetch data from specific voxel for each stimulus size
            self.voxel_in_events = []
            self.error_in_voxels = []
            for idc, stim in enumerate(self.cond):
                col_data = plot_mean[col].xs(stim).to_numpy(copy=True)
                err_data = plot_sem[col].xs(stim).to_numpy(copy=True)


                self.voxel_in_events.append(col_data[np.newaxis, :])
                self.error_in_voxels.append(err_data[np.newaxis, :])

            # this one is in case we want the voxels in 1 figure
            self.all_voxels_in_event.append(
                np.concatenate(
                    self.voxel_in_events, axis=0)[
                    np.newaxis, :])
            self.all_error_in_voxels.append(
                np.concatenate(
                    self.error_in_voxels, axis=0)[
                    np.newaxis, :])

        self.arr_voxels_in_event = np.concatenate(
            self.all_voxels_in_event, axis=0)
        self.arr_error_in_event = np.concatenate(
            self.all_error_in_voxels, axis=0)

        # try to find min/max across voxels. Use error if there were more than
        # 1 run
        top = self.arr_voxels_in_event
        bottom = self.arr_voxels_in_event
        if len(self.run_ids) > 1:
            top += self.arr_error_in_event
            bottom -= self.arr_error_in_event

        if make_figure:
            if n_cols is None:
                if labels:
                    labels = cols.copy()
                else:
                    labels = None

                vox_data = [self.arr_voxels_in_event[ii, 0, :]
                            for ii in range(self.arr_voxels_in_event.shape[0])]
                vox_error = [self.arr_error_in_event[ii, 0, :]
                             for ii in range(self.arr_voxels_in_event.shape[0])]

                if isinstance(axs, mpl.axes._axes.Axes):
                    self.pl = plotting.LazyLine(
                        vox_data,
                        xx=self.time,
                        error=vox_error,
                        axs=axs,
                        labels=labels,
                        add_hline='default',
                        **kwargs)
                else:
                    self.pl = plotting.LazyLine(
                        vox_data,
                        xx=self.time,
                        error=vox_error,
                        figsize=figsize,
                        labels=labels,
                        add_hline='default',
                        **kwargs)
            else:
                for ix, col in enumerate(cols):
                    if isinstance(axs, (np.ndarray, list)):
                        ax = axs[ix]
                        new_axs = True
                    else:
                        new_axs = False
                        ax = fig.add_subplot(gs[ix])

                    if ix == 0:
                        label = labels
                    else:
                        label = None

                    vox_data = [self.arr_voxels_in_event[ix, ii, :]
                                for ii in range(len(self.cond))]
                    vox_error = [self.arr_error_in_event[ix, ii, :]
                                 for ii in range(len(self.cond))]

                    if sharey:
                        ylim = [np.nanmin(bottom), np.nanmax(top)]
                    else:
                        ylim = None

                    if isinstance(title, (str, list)):
                        if isinstance(title, str):
                            title = [title for _ in cols]

                        add_title = title[ix]
                    else:
                        add_title = col

                    self.pl = plotting.LazyLine(
                        vox_data,
                        xx=self.time,
                        error=vox_error,
                        axs=ax,
                        labels=label,
                        add_hline='default',
                        y_lim=ylim,
                        title=add_title,
                        **kwargs)

                    if not new_axs:
                        if ix in cols_id[::n_cols]:
                            ax.set_ylabel(
                                "Magnitude (%change)", fontsize=self.font_size)

                        if ix in np.arange(n_rows * n_cols)[-n_cols:]:
                            ax.set_xlabel("Time (s)", fontsize=self.font_size)
                    else:
                        if not isinstance(skip_x, list):
                            skip_x = [False for _ in cols]

                        if not isinstance(skip_y, list):
                            skip_y = [False for _ in cols]

                        if not skip_x[ix]:
                            ax.set_xlabel("Time (s)", fontsize=self.font_size)

                        if not skip_y[ix]:
                            ax.set_ylabel(
                                "Magnitude (%change)", fontsize=self.font_size)

                plt.tight_layout()

        if save_as:
            fig.savefig(
                save_as,
                dpi=300,
                bbox_inches='tight')

    def plot_hrf_across_depth(
            self,
            axs=None,
            figsize: tuple = (8, 8),
            cmap: str = 'viridis',
            color: Union[str, tuple] = None,
            ci_color: Union[str, tuple] = "#cccccc",
            ci_alpha: float = 0.6,
            save_as: str = None,
            invert: bool = False,
            **kwargs):
        """plot_hrf_across_depth

        Plot the magnitude of the HRF across depth as points with a seaborn regplot through it. The points can be colored with
        `color` according to the HRF from :func:`lazyfmri.fitting.NideconvFitter.plot_average_across_voxels`, or they can
        be given 1 uniform color. The linear fit can be colored using `ci_color`, for which the default is light gray.

        Parameters
        ----------
        axs: <AxesSubplot:>, optional
            Matplotlib axis to store the figure on, by default None
        figsize: tuple, optional
            Figure size, by default (8,8)
        cmap: str, optional
            Color map for the data points, by default 'viridis'
        color: str, tuple, optional
            Don't use a color map for the data points, but a uniform color instead, by default None. `cmap` takes precedence!
        ci_color: str, tuple, optional
            Color of the linear fit with seaborn's regplot, by default "#cccccc"
        ci_alpha: float, optional
            Alpha of linear fit, by default 0.6
        save_as: str, optional
            Save as file, by default None
        invert: bool, optional
            By default, we'll assume your input data represents voxels from CSF to WM. This flag can flip that around, by
            default False

        Example
        ----------
        .. code-block:: python
                
            # lump events together
            lumped = fitting.NideconvFitter(
                df_ribbon,
                df_onsets,
                basis_sets='fourier',
                n_regressors=4,
                lump_events=True,
                TR=0.105,
                interval=[-3,17])

            # plot
            lumped.plot_hrf_across_depth(x_label="depth [%]")

            # make a combined plot of HRFs and magnitude
            fig = plt.figure(figsize=(16, 8))
            gs = fig.add_gridspec(1, 2)
                >>>
            ax = fig.add_subplot(gs[0])
            lumped.plot_average_per_voxel(
                n_cols=None,
                axs=ax,
                labels=True,
                x_label="time (s)",
                y_label="magnitude",
                set_xlim_zero=False)
            ax.set_title("HRF across depth (collapsed stimulus events)", fontsize=lumped.pl.font_size)
                >>>
            ax = fig.add_subplot(gs[1])
            lumped.plot_hrf_across_depth(
                axs=ax,
                order=1,
                x_label="depth [%]")
            ax.set_title("Maximum value HRF across depth", fontsize=lumped.pl.font_size)

        """

        if not hasattr(self, "all_voxels_in_event"):
            self.plot_timecourses(make_figure=False)

        self.max_vals = np.array([np.amax(self.all_voxels_in_event[ii])
                                 for ii in range(len(self.all_voxels_in_event))])

        if axs is None:
            fig, axs = plt.subplots(figsize=figsize)

        if isinstance(cmap, str):
            color_list = sns.color_palette(cmap, len(self.max_vals))
        else:
            color_list = [color for _ in self.max_vals]

        self.depths = np.linspace(0, 100, num=len(self.max_vals))
        if invert:
            self.max_vals = self.max_vals[::-1]

        self.pl = plotting.LazyCorr(
            self.depths,
            self.max_vals,
            color=ci_color,
            axs=axs,
            x_ticks=[0, 50, 100],
            points=False,
            scatter_kwargs={"cmap": cmap},
            **kwargs)

        for ix, mark in enumerate(self.max_vals):
            axs.plot(
                self.depths[ix],
                mark,
                'o',
                color=color_list[ix],
                alpha=ci_alpha)

        for pos, tag in zip([(0.02, 0.02), (0.85, 0.02)], ["pial", "wm"]):
            axs.annotate(
                tag,
                pos,
                fontsize=self.pl.font_size,
                xycoords="axes fraction"
            )

        if save_as:
            fig.savefig(save_as, dpi=300, bbox_inches='tight')

    def plot_areas_per_event(
            self,
            colors=None,
            save_as=None,
            add_offset=True,
            error_type="sem",
            axs=None,
            events=None,
            **kwargs):

        if not hasattr(self, "tc_condition"):
            self.timecourses_condition()

        plot_mean, plot_sem, plot_std = self._event_plot_statistics(add_offset=add_offset)
        n_cols = len(list(self.cond))
        figsize = (n_cols * 6, 6)

        if axs is None:
            fig = plt.figure(figsize=figsize)
            gs = fig.add_gridspec(1, n_cols)

        if error_type == "std":
            err = plot_std.copy()
        elif error_type == "sem":
            err = plot_sem.copy()
        else:
            err = plot_std.copy()

        if not isinstance(events, list) and not isinstance(events, np.ndarray):
            events = self.cond
        else:
            if self.verbose:
                print(f"Flipping events to {events}")

        for ix, event in enumerate(events):

            # add axis
            if axs is None:
                ax = fig.add_subplot(gs[ix])
            else:
                ax = axs

            for key in list(kwargs.keys()):
                if ix != 0:
                    if key == "y_ticks":
                        kwargs[key] = []
                    elif key == "y_label":
                        kwargs[key] = None

            event_df = utils.select_from_df(
                plot_mean, expression=f"event_type = {event}")
            error_df = utils.select_from_df(
                err, expression=f"event_type = {event}")

            self.data_for_plot = []
            self.error_for_plot = []
            for ii, dd in enumerate(list(self.tc_condition.columns)):

                # get the timecourse
                col_data = np.squeeze(
                    utils.select_from_df(
                        event_df,
                        expression='ribbon',
                        indices=[ii]).to_numpy(copy=True))
                col_error = np.squeeze(
                    utils.select_from_df(
                        error_df,
                        expression='ribbon',
                        indices=[ii]).to_numpy(copy=True))


                self.data_for_plot.append(col_data)
                self.error_for_plot.append(col_error)

            if not isinstance(error_type, str):
                self.error_for_plot = None

            plotting.LazyLine(
                self.data_for_plot,
                xx=self.time,
                error=self.error_for_plot,
                axs=ax,
                **kwargs)

            if axs is None:
                if save_as:
                    fig.savefig(save_as, dpi=300, bbox_inches='tight')


class CVDeconv(InitFitter):

    def __init__(
        self,
        func,
        onsets,
        TR=0.105):

        """CVDeconv

        Wrapper class around :class:`nideconv.NideconvFitter` to run deconvolution in a cross-validated r2-framework. Inputs
        are similar to :class:`nideconv.NideconvFitter`. The basic gist of the class is to find all possible run-specific
        combinations, average those, and run deconvolution. This is repeated until all combinations have been used, resulting
        in a final dataframe containing the r2-values for each subject.

        Parameters
        ----------
        func: pd.DataFrame
            Dataframe as per the output of :func:`lazyfmri.dataset.Datasets.fetch_fmri()`, containing the fMRI data
            indexed on subject, run, and t.
        onsets: pd.DataFrame
            Dataframe as per the output of :func:`lazyfmri.dataset.Datasets.fetch_onsets()`, containing the onset timings
            data indexed on subject, run, and event_type.
        TR: float, optional
            Repetition time, by default 0.105. Use to calculate the sampling frequency (1/TR)

        Example
        ----------

        .. code-block:: python
            # import stuff
            from lazyfmri import fitting
            cv_ = fitting.CVDeconv(
                func,
                onsets,
                TR=1.32
            )
            # run the crossvalidation
            cv_.crossvalidate(
                basis_sets="Fourier",
                n_regressors=9,
                interval=[-2,24]
            )
            # final dataframe:
            cr = cv_.r2_

        """
        self.func = func
        self.onsets = onsets
        self.TR = TR

        # format functional data and onset dataframes
        super().__init__(
            self.func,
            self.onsets,
            self.TR,
            merge=False
        )

    def return_combinations(self, split=2):

        """return_combinations

        Generates all possible unique combinations of run indices for cross-validation.

        Parameters
        ----------
        split : int, optional
            The number of runs to be included in each split, by default `2`.

        Returns
        -------
        list of tuples
            A list containing tuples, where each tuple represents a unique combination of runs.

        Raises
        ------
        ValueError
            If there are fewer than 3 runs, as cross-validation is ineffective with fewer than 3 runs.

        Example
        ----------
        .. code-block:: python

            cv = CVDeconv(func, onsets, TR=1.32)
            run_combinations = cv.return_combinations(split=2)
        """

        self.run_ids = utils.get_unique_ids(self.func, id="run")
        if len(self.run_ids) < 2 or not isinstance(split, (int, np.integer)) or not 1 <= split < len(self.run_ids):
            raise ValueError('Cross-validation requires at least two runs and 1 <= split < number of runs')

        # get all possible combination of runs
        return list(combinations(self.run_ids, split))

    def single_fitter(self, func, onsets, **kwargs):

        """single_fitter

        Creates an instance of :class:`lazyfmri.fitting.NideconvFitter` for a given functional dataset and onset times.

        Parameters
        ----------
        func : pd.DataFrame
            Functional fMRI data in the form of a pandas DataFrame.
        onsets : pd.DataFrame
            Onset timings corresponding to the events in the functional data.
        **kwargs : dict, optional
            Additional parameters passed to :class:`lazyfmri.fitting.NideconvFitter`.

        Returns
        -------
        :class:`lazyfmri.fitting.NideconvFitter`
            An instance of :class:`lazyfmri.fitting.NideconvFitter` fitted to the given data.

        Example
        ----------
        .. code-block:: python

            fitter = cv.single_fitter(df_func, df_onsets, basis_sets="Fourier", n_regressors=6)
        """

        if "fit" not in list(kwargs.keys()):
            kwargs["fit"] = True

        # run single fitter and fetch timecourses
        fit_obj = NideconvFitter(
            func,
            onsets,
            TR=self.TR,
            **kwargs
        )

        if kwargs["fit"]:
            fit_obj.timecourses_condition()

        return fit_obj

    # def out_of_set_prediction(self):
    def predict_out_of_set(self, src, trg):
        """Average labeled training coefficients, separately for each acquisition context.

        Held-out runs are scored separately, then averaged equally. A regressor
        absent from any training fit is not silently filled with a zero coefficient.
        """
        sources = src if isinstance(src, list) else [src]
        targets = [trg] if isinstance(trg, (str, int, float, np.integer)) else list(trg)
        if not sources or not targets:
            raise ValueError('Specify training fitters and held-out runs')
        training = []
        for fitted in sources:
            table = fitted.fitters
            if isinstance(table, pd.Series):
                table = table.to_frame()
            for key, row in table.iterrows():
                key = key if isinstance(key, tuple) else (key,)
                metadata = dict(zip(table.index.names, key))
                if metadata.get('run') in targets:
                    raise ValueError('Held-out run occurs in training fitters')
                training.append((metadata, row))
        full = self.full_fitters
        if isinstance(full, pd.Series):
            full = full.to_frame()
        scores = []
        for key, row in full.iterrows():
            key = key if isinstance(key, tuple) else (key,)
            metadata = dict(zip(full.index.names, key))
            if metadata.get('run') not in targets:
                continue
            context = {k: v for k, v in metadata.items() if k != 'run'}
            eligible = [record for record in training if all(record[0].get(k) == v for k, v in context.items())]
            if not eligible:
                raise ValueError(f'No training fits for {context}')
            # OLS has one multivariate fitter; ridge has one fitter per signal.
            values = {}
            for column, target_fitter in row.items():
                fitters = [record[1][column] for record in eligible]
                template = fitters[0].betas
                coefficients = []
                for fitter in fitters:
                    if set(fitter.betas.index) != set(template.index) or set(fitter.betas.columns) != set(template.columns):
                        raise ValueError('Training designs have different regressor/ROI labels; use a consistent model')
                    coefficients.append(fitter.betas.reindex(index=template.index, columns=template.columns).to_numpy(dtype=float))
                self.betas = pd.DataFrame(np.mean(coefficients, axis=0), index=template.index, columns=template.columns)
                design = target_fitter.X
                if set(design.columns) != set(self.betas.index):
                    raise ValueError('Training and held-out designs have different regressor labels')
                prediction = design.dot(self.betas.reindex(design.columns))
                observed = target_fitter.input_signal.reindex(columns=prediction.columns)
                if observed.shape != prediction.shape:
                    raise ValueError('Held-out predictions and data have different shapes')
                for roi in prediction.columns:
                    values[roi] = metrics.r2_score(observed[roi].to_numpy(), prediction[roi].to_numpy())
            scores.append({**context, **values})
        if not scores:
            raise ValueError('No requested held-out acquisitions found')
        result = pd.DataFrame(scores)
        contexts = [key for key in ('subject', 'session', 'task') if key in result]
        roi_columns = [key for key in result if key not in contexts]
        return result.groupby(contexts, as_index=False)[roi_columns].mean()[roi_columns + contexts]

    def crossvalidate(self, split=2, **kwargs):

        """crossvalidate

        Performs cross-validation using :class:`lazyfmri.fitting.NideconvFitter` by iteratively leaving out runs, training on the remaining runs,
        and predicting the left-out runs.

        Parameters
        ----------
        split : int, optional
            The number of runs included in each training fold, by default `2`.
        **kwargs : dict, optional
            Additional parameters passed to :class:`lazyfmri.fitting.NideconvFitter`.

        Returns
        -------
        None
            The method updates `self.r2_` with the R² values for each fold.

        Example
        ----------
        .. code-block:: python

            cv.crossvalidate(split=2, basis_sets="Fourier", n_regressors=9, interval=[-2,24])
        """

        # for later reference of design matrices
        self.full_model = self.single_fitter(
            self.func,
            self.onsets,
            fit=False,
            **kwargs  # all other parameters for NideconvFitter
        )

        # get fitters
        self.full_fitters = (self.full_model.model._get_response_fitters()
                            if hasattr(self.full_model, 'model') else self.full_model.sub_df.copy())

        self.combos = self.return_combinations(split=split)

        # loop through them
        self.fitters = []
        self.r2_ = []
        for ix, combo in enumerate(self.combos):
            unfitted_runs = [i for i in self.run_ids if i not in combo]

            self.fold_f = self.func.loc[self.func.index.get_level_values('run').isin(combo)]
            self.fold_o = self.onsets.loc[self.onsets.index.get_level_values('run').isin(combo)]

            fit_ = self.single_fitter(
                self.fold_f,
                self.fold_o,
                **kwargs  # all other parameters for NideconvFitter
            )

            self.fitters.append(fit_)

            # predict non-fitted run(s)
            df = self.predict_out_of_set(fit_, unfitted_runs)
            df["fold"] = ix
            self.r2_.append(df)

        self.r2_ = pd.concat(self.r2_)
        self.r2_ = self.r2_.set_index([key for key in ('subject', 'session', 'task', 'fold') if key in self.r2_])


def fwhm_lines(
        fwhm_list,
        axs,
        cmap="viridis",
        color=None,
        **kwargs):

    if not isinstance(fwhm_list, list):
        fwhm_list = [fwhm_list]

    # heights need to be adjusted for by axis length
    if not isinstance(color, (str, tuple)):
        colors = sns.color_palette(cmap, len(fwhm_list))
    else:
        colors = [color for _ in range(len(fwhm_list))]

    xlim = axs.get_xlim()
    tot = sum(list(np.abs(xlim)))
    for ix, ii in enumerate(fwhm_list):
        start = (ii.t0_ - xlim[0]) / tot
        axs.axhline(
            ii.half_max,
            xmin=start,
            xmax=start + ii.fwhm / tot,
            color=colors[ix],
            linewidth=0.5,
            **kwargs)


class HRFMetrics():

    def __init__(
        self,
        hrf,
        **kwargs
        ):
        """HRFMetrics

        A class for extracting various parameters from a given profile, optimized for the hemodynamic response function (HRF)
        but applicable to any time-series profile. It supports both `pandas.DataFrame` and `numpy.ndarray` as input, automatically
        converting them to a DataFrame for consistent processing. By default, parameters are extracted from the **largest peak** 
        (positive or negative) in the profile. If needed, you can **force** parameter extraction from positive (`force_pos=True`) 
        or negative (`force_neg=True`) responses. These options can be applied per column when using a DataFrame.

        **Extracted Parameters**
        The following HRF-related parameters are extracted and formatted into a DataFrame:
        
        - **magnitude**: Largest response in the profile.
        - **magnitude_ix**: Time index at which the largest response occurs.
        - **fwhm**: Full-width at half-maximum (FWHM).
        - **fwhm_obj**: Instance of `linescanning.fitting.FWHM`, storing additional FWHM properties.
        - **time_to_peak**: Time taken to reach the peak response.
        - **half_rise_time**: Time taken to reach half of the maximum response.
        - **half_max**: Half of the maximum response.
        - **rise_slope**: Slope of the rising phase towards the peak.
        - **onset_time**: Time at which the rising phase begins.
        - **positive_area**: Area-under-the-curve (AUC) of the positive response.
        - **undershoot**: AUC of the negative undershoot.

        Parameters
        ----------
        hrf : pd.DataFrame or np.ndarray
            Input profile, either 1D or 2D. If a 2D DataFrame is provided, each column represents a separate time-series.
        TR : float, optional
            Repetition time (TR) in seconds. Required if `hrf` is a NumPy array to construct time indices correctly.
        force_pos : bool or list, optional
            Force extraction from the **positive** response peak. Can be applied per column when using a DataFrame. Default is `False`.
        force_neg : bool or list, optional
            Force extraction from the **negative** response peak. Can be applied per column when using a DataFrame. Default is `False`.
        plot : bool, optional
            If `True`, plots the HRF profile when parameter extraction fails (useful for debugging). Default is `False`.
        nan_policy : bool, optional
            If `True`, failed extractions return NaN. Default is `False`.
            Flat or variance-excluded profiles always return NaN, with reasons
            available in the returned dataframe's attrs['qc'].
        debug : bool, optional
            If `True`, prints additional debugging information during parameter extraction. Default is `False`.
        progress : bool, optional
            If `True`, displays a progress bar using `alive_progress`. Default is `False`.
        thr_var : int or float, optional
            Variance threshold for input profiles. Time-series with variance below this threshold are ignored. Default is `0`.
        shift : int or float, optional
            Optional baseline sample count to subtract before extraction.
            None leaves the profile unchanged; 0 or 1 subtracts its first sample.
            Default is None. Baseline correction never modifies the input.
        vox_as_index : bool, optional
            If `True`, uses DataFrame column indices as voxel identifiers. Otherwise, original column names are retained. Default is `False`.
        col_name : str, optional
            Custom name for the added index when processing 2D DataFrames. Example: `"roi"` for region-based processing. Default is `"vox"`.
        peak : int, optional
            Select the nth qualifying peak in chronological order. None uses
            the largest absolute response, subject to force_pos/force_neg.
            If fewer than n peaks qualify, the largest response is used.

        Example
        ----------
        .. code-block:: python

            from lazyfmri import fitting

            # Extract HRF metrics from a profile
            metrics_kws = {}  # Additional parameters
            metrics = fitting.HRFMetrics(some_profile).return_metrics()

        """

        # get metrics
        self.metrics, self.final_hrf = self._get_metrics(hrf, **kwargs)

    def return_metrics(self):
        """return_metrics

        Returns the extracted HRF parameters in a formatted DataFrame.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing extracted HRF metrics, such as peak amplitude, time-to-peak, full-width half-maximum (FWHM), 
            area under the curve (AUC), and slope-related features.

        Example
        ----------
        .. code-block:: python

            metrics_df = HRFMetrics(some_profile).return_metrics()
            print(metrics_df)
        """

        return self.metrics.reset_index(drop=True)

    @staticmethod
    def _check_negative(hrf, force_pos=False, force_neg=False):
        """_check_negative

        Determines whether the largest peak in the HRF is negative or positive.

        Parameters
        ----------
        hrf : pd.DataFrame
            Input HRF time-series.
        force_pos : bool, optional
            If `True`, forces the HRF to be considered positive, by default `False`.
        force_neg : bool, optional
            If `True`, forces the HRF to be considered negative, by default `False`.

        Returns
        -------
        bool
            `True` if the HRF is negative (dominant negative peak), `False` otherwise.

        Example
        ----------
        .. code-block:: python

            is_negative = HRFMetrics._check_negative(hrf_data)
        """

        # check if positive or negative is largest
        if abs(hrf.min(axis=0).values) > hrf.max(axis=0).values:
            negative = True
        else:
            negative = False

        if force_pos:
            negative = False

        if force_neg:
            negative = True
        return negative

    def _get_metrics(
        self,
        hrf,
        TR: float = None,
        force_pos: Union[list, bool] = False,
        force_neg: Union[list, bool] = False,
        debug: bool = False,
        progress: bool = False,
        thr_var: Union[int, float] = 0,
        vox_as_index: bool = False,
        col_name: str = "vox",
        incl: Union[str, list] = [
            "magnitude",
            "magnitude_ix",
            "fwhm",
            "fwhm_obj",
            "time_to_peak",
            "half_rise_time",
            "half_max",
            "rise_slope",
            "onset_time",
            "positive_area",
            "undershoot",
            "auc_simple_pos",
            "auc_simple_neg",
            "auc_simple_total",
            "auc_simple_pos_norm",
            "auc_simple_neg_norm",
            "auc_simple_total_norm",            
            "1st_deriv_magnitude",
            "1st_deriv_time_to_peak",
            "2nd_deriv_magnitude",
            "2nd_deriv_time_to_peak",
        ],
        **kwargs
        ):

        """Extract independent ROI metrics with explicit QC for excluded inputs.

        Excluded/failed metrics are NaN, never fabricated zeros. QC reasons are
        stored in metrics.attrs['qc']; the historical metric columns are kept.
        """
        frame = self._verify_input(hrf, TR=TR, shift=None).copy()
        columns = list(frame.columns)
        requested = [incl] if isinstance(incl, str) else list(incl)
        def flags(value):
            if isinstance(value, (list, tuple, np.ndarray)):
                if len(value) != len(columns):
                    raise ValueError('Peak-direction flags must match all input ROI columns')
                return list(value)
            return [value] * len(columns)
        positive, negative = flags(force_pos), flags(force_neg)
        rows, quality = [], {}
        final_hrf = frame.copy()
        nan_policy = kwargs.get('nan_policy', False)
        context = alive_bar(len(columns), force_tty=True) if progress else nullcontext(lambda: None)
        with context as advance:
            for ix, column in enumerate(columns):
                profile = frame[column]
                values = profile.to_numpy(dtype=float)
                reason = None
                if not np.isfinite(values).all():
                    reason = 'nonfinite_signal'
                    if not nan_policy:
                        raise ValueError(f'Nonfinite HRF values in {column}')
                elif len(values) < 2 or np.var(values, ddof=1) <= thr_var:
                    reason = 'below_variance_threshold'
                record = {name: np.nan for name in requested}
                if reason is None:
                    try:
                        result, _, final_hrf = self._get_single_hrf_metrics(
                            profile, TR=TR, force_pos=positive[ix], force_neg=negative[ix],
                            debug=debug, **kwargs)
                        unknown = set(requested) - set(result.columns)
                        if unknown:
                            raise ValueError(f'Unknown HRF metrics: {sorted(unknown)}')
                        record = result.iloc[0].reindex(requested).to_dict()
                        reason = 'ok'
                        numeric = [v for v in record.values() if isinstance(v, (float, np.floating))]
                        if any(not np.isfinite(v) for v in numeric):
                            reason = 'partial_metrics'
                    except (ValueError, FloatingPointError, ZeroDivisionError) as exc:
                        if not nan_policy:
                            raise ValueError(f'HRF metric extraction failed for {column}: {exc}') from exc
                        reason = str(exc)
                rows.append(record)
                quality[column] = reason
                advance()
        metrics = pd.DataFrame(rows, columns=requested)
        if len(columns) > 1:
            metrics[col_name] = np.arange(len(columns)) if vox_as_index else columns
        if 'magnitude_ix' in metrics:
            metrics['magnitude_ix'] = pd.array(metrics['magnitude_ix'], dtype='Int64')
        metrics.attrs['qc'] = quality
        return metrics, final_hrf

    @staticmethod
    def _get_time(hrf):
        """_get_time

        Retrieves the time axis from the HRF DataFrame.

        Parameters
        ----------
        hrf : pd.DataFrame
            The input HRF time-series.

        Returns
        -------
        np.ndarray
            Array representing the time values.
        str
            Column name representing the time index (either `"time"` or `"t"`).

        Raises
        ------
        ValueError
            If no valid time column is found.

        Example
        ----------
        .. code-block:: python

            time_values, time_col = HRFMetrics._get_time(hrf_data)
        """
        
        try:
            tmp_df = hrf.reset_index()
        except BaseException:
            tmp_df = hrf.copy()

        try:
            return tmp_df["time"].values, "time"
        except BaseException:
            try:
                return tmp_df["t"].values, "t"
            except BaseException:
                raise ValueError(
                    "Could not find time dimension. Dataframe should contain 't' or 'time' column..")

    @classmethod
    def _verify_input(
        self,
        hrf,
        TR=None,
        shift=None
        ):

        """_verify_input

        Verifies and formats the input HRF time-series.

        Parameters
        ----------
        hrf : np.ndarray or pd.DataFrame
            Input HRF time-series.
        TR : float, optional
            Repetition time in seconds, required if input is a NumPy array.
        shift : int or float, optional
            If specified, shifts the profile to avoid negative baselines.

        Returns
        -------
        pd.DataFrame
            Formatted HRF time-series.

        Example
        ----------
        .. code-block:: python

            formatted_hrf = HRFMetrics._verify_input(hrf_data, TR=1.5)
        """

        if isinstance(hrf, np.ndarray):
            if hrf.ndim > 1:
                hrf = hrf.squeeze()

            if not isinstance(TR, (int, float)):
                raise ValueError(
                    "Please specify repetition time of this acquisition to construct time axis")

            time_axis = list(np.arange(0, hrf.shape[0]) * TR)
            hrf = pd.DataFrame({"voxel": hrf})
            hrf["time"] = time_axis
            hrf = hrf.set_index(["time"])

        elif isinstance(hrf, pd.Series):
            hrf = pd.DataFrame(hrf)

        # Work on a copy: metric baseline correction must not alter input data.
        hrf = hrf.copy()
        if isinstance(shift, (int, np.integer)):
            if shift < 0:
                raise ValueError('shift must be nonnegative or None')
            samples = max(1, shift)
            baseline = hrf.iloc[:samples].mean(axis=0)
            hrf = hrf - baseline
        return hrf

    @classmethod
    def plot_profile_for_debugging(
        self,
        hrf,
        extra=[],
        axs=None,
        figsize=(5, 5),
        peaks=None,
        **kwargs):

        """plot_profile_for_debugging

        Plots the HRF time-series for debugging purposes.

        Parameters
        ----------
        hrf : pd.DataFrame
            The input HRF time-series.
        axs : matplotlib.axes._axes.Axes, optional
            Matplotlib axis for plotting, by default `None`.
        figsize : tuple, optional
            Figure size, by default `(5, 5)`.
        **kwargs : dict
            Additional keyword arguments for plotting.

        Example
        ----------
        .. code-block:: python

            HRFMetrics.plot_profile_for_debugging(hrf_data)
        """

        if not isinstance(axs, mpl.axes._axes.Axes):
            _, axs = plt.subplots(figsize=figsize)

        time, time_col = self._get_time(hrf)

        kwargs = utils.update_kwargs(
            kwargs,
            "title",
            {
                "title": list(hrf.columns)[0],
                "fontweight": "bold"
            }
        )
        
        if len(extra)>0:
            data = [
                hrf.values.squeeze()
            ] + extra

            colors = ["r"]+sns.color_palette("viridis", len(extra))
            line_width = [3]+[1 for i in extra]
        else:
            data = hrf.values.squeeze()
            colors = "r"
            line_width = 3

        hmin, hmax = hrf.values.squeeze().min(), hrf.values.squeeze().max()
        
        y_lim = [
            hmin-abs(hmin*0.5),
            hmax*10
        ]

        kk = {
            "line_width": line_width,
            "color": colors,
            "xx": list(time),
            "x_label": "time (s)",
            "y_label": "magnitude",
            "axs": axs,
            "add_hline": 0,
            # "y_lim": y_lim
        }

        for k, v in kk.items():
            kwargs = utils.update_kwargs(
                kwargs,
                k,
                v
            )

        pl = plotting.LazyLine(
            data,
            **kwargs
        )
        
        if isinstance(peaks, (list, float)):
            peaks = [int(i) for i in peaks]
            if isinstance(pl.xx, list):
                t_ax = np.array(pl.xx)
            else:
                t_ax = pl.xx.copy()

            pl.axs.plot(
                t_ax[peaks],
                hrf.values.squeeze()[peaks],
                'kx'
            )

        return pl
            

    @classmethod
    def _get_riseslope(
        self,
        hrf,
        force_pos=False,
        force_neg=False,
        nan_policy=False,
        peak=None,
        resample_to_shape=500,
        t_start=0,
        max_scaler=0.05,
        noise_scaler=1        
        ):

        """_get_riseslope

        Computes the slope of the rising phase of the HRF.

        Parameters
        ----------
        hrf : pd.DataFrame
            The input HRF time-series.
        force_pos : bool, optional
            If `True`, forces extraction from a positive response, by default `False`.
        force_neg : bool, optional
            If `True`, forces extraction from a negative response, by default `False`.
        nan_policy : bool, optional
            If `True`, assigns `NaN` if extraction fails, by default `False`.
        peak : int, optional
            Specifies which peak to use, by default `None`.

        Returns
        -------
        float
            The estimated rise slope.
        float
            Time at which the rise slope occurs.

        Example
        ----------
        .. code-block:: python

            slope, onset_time = HRFMetrics._get_riseslope(hrf_data)
        """

        # fetch time stamps
        time, time_col = self._get_time(hrf)

        # find slope corresponding to amplitude
        mag = self._get_amplitude(
            hrf,
            force_pos=force_pos,
            force_neg=force_neg,
            peak=peak,
            resample_to_shape=resample_to_shape,
            t_start=t_start,
            max_scaler=max_scaler,
            noise_scaler=noise_scaler            
        )

        negative = False
        if mag["amplitude"] < 0:
            negative = True

        # limit search to where index of highest amplitude
        diff = np.diff(hrf.values.squeeze()[:mag["t_ix"]]) / np.diff(time[:mag["t_ix"]])

        # find minimum/maximum depending on whether HRF is negative or not
        try:
            if not force_pos:
                if negative:
                    val = np.array([np.amin(diff[:mag["t_ix"]])])
                else:
                    val = np.array([np.amax(diff[:mag["t_ix"]])])
            else:
                val = np.array([np.amax(diff[:mag["t_ix"]])])

            final_val = val[0]
            val_ix = utils.find_nearest(diff[:mag["t_ix"]], final_val)[0]
            val_t = time[val_ix]

        except Exception as e:  # noqa: F821
            if not nan_policy:
                pl = self.plot_profile_for_debugging(hrf)
                raise ValueError(f"Could not extract rise-slope from this profile: {e}")  # noqa: F821
            else:
                val_t = final_val = np.nan

        return final_val, val_t, np.nan

    @classmethod
    def _resample(self, hrf, resample_to_shape=500):
        """Interpolate values against actual time coordinates, including uneven grids."""
        time, time_col = self._get_time(hrf)
        time = np.asarray(time, dtype=float)
        values = hrf.to_numpy(dtype=float)
        if len(time) < 2 or not np.isfinite(time).all() or not np.isfinite(values).all():
            raise ValueError('HRF resampling requires at least two finite observations')
        order = np.argsort(time)
        time, values = time[order], values[order]
        if np.any(np.diff(time) <= 0):
            raise ValueError('HRF time coordinates must be unique')
        if resample_to_shape is None:
            target = time
        else:
            if int(resample_to_shape) != resample_to_shape or resample_to_shape < 2:
                raise ValueError('resample_to_shape must be an integer of at least 2')
            target = np.linspace(time[0], time[-1], int(resample_to_shape))
        data = np.column_stack([np.interp(target, time, values[:, col]) for col in range(values.shape[1])])
        return target, pd.DataFrame(data, columns=hrf.columns, index=pd.Index(target, name=time_col))
    
    @classmethod
    def _get_riseslope_siero(
        self,
        hrf,
        force_pos=False,
        force_neg=False,
        nan_policy=False,
        window=[0.2, 0.8],
        t_start=0,
        resample_to_shape=500,
        reference="mag",
        peak=None,
        max_scaler=0.05,
        noise_scaler=1,
        verify_slope=False,
        **kwargs
        ):

        """_get_riseslope_siero

        Computes the rise slope of the HRF based on the method by Siero and Tian, which fits a linear model to the rising phase 
        between 20% and 80% of the peak response. This provides a robust estimate of the steepness of the HRF rise.

        Parameters
        ----------
        hrf : pd.DataFrame
            The input HRF time-series.
        force_pos : bool, optional
            If `True`, forces extraction from a positive response, by default `False`.
        force_neg : bool, optional
            If `True`, forces extraction from a negative response, by default `False`.
        nan_policy : bool, optional
            If `True`, assigns `NaN` if extraction fails, by default `False`.
        window : list of float, optional
            Specifies the fraction of the peak response to use for the slope calculation. The default `[0.2, 0.8]` uses the interval
            between 20% and 80% of the peak.
        reference : str, optional
            Specifies whether to use `"mag"` (magnitude-based peak) or `"fwhm"` (half-maximum-based point) as the reference for 
            rise slope calculation. Default is `"mag"`.
        peak : int, optional
            Specifies which peak to use when multiple peaks are present, by default `None`.

        Returns
        -------
        tuple
            - `slope_val` (float): The estimated rise slope value.
            - `intsc` (float): The estimated intercept time, indicating when the rising phase begins.

        Example
        ----------
        .. code-block:: python

            slope, intercept_time = HRFMetrics._get_riseslope_siero(hrf_data)
        """

        # fetch time stamps
        # hrf = glm.resample_stim_vector(hrf, resample_to_shape)
        time, time_col = self._get_time(hrf)

        # resample
        time_rs, hrf_rs = self._resample(
            hrf,
            resample_to_shape=resample_to_shape
        )

        # correct for 0
        time_corr, hrf_corr = self._correct_t0(
            hrf_rs,
            t_start=t_start
        )

        # find index where t=0
        t0_idx = utils.find_nearest(time_rs, t_start)[0]

        # find slope based on fwhm
        if reference not in ["magnitude", "mag"]:
            fwhm_obj = self._get_fwhm(
                hrf_rs,
                force_pos=force_pos,
                force_neg=force_neg,
                nan_policy=nan_policy,
                peak=peak,
                max_scaler=max_scaler,
                noise_scaler=noise_scaler
            )

            # get interval around fwhm
            if isinstance(fwhm_obj["obj"], float):
                if nan_policy:
                    slope_val = fwhm_t0 = np.nan
                else:
                    raise TypeError(
                        "Could not extract fwhm from profile, so rise slope could not be determined")
            else:
                fwhm_t0 = fwhm_obj["obj"].t0_
                t0 = fwhm_t0 * window[0]
                t1 = fwhm_t0 * window[1]

        else:
            # use Siero/Tian method (fit between 20-80% of max)
            mag = self._get_amplitude(
                hrf,
                force_pos=force_pos,
                force_neg=force_neg,
                peak=peak,
                resample_to_shape=resample_to_shape,
                t_start=t_start,
                max_scaler=max_scaler,
                noise_scaler=noise_scaler                
            )

            mag_ix = mag["t_ix"]
            ttp = mag["t"]
            mag_window = [mag["amplitude"]*i for i in window]
            
            relevant_samples = hrf_rs.values.squeeze()[t0_idx:mag_ix]
            if mag_ix<t0_idx or len(relevant_samples) == 0:
                # peak likely before 0
                mag = self._get_amplitude(
                    hrf_rs,
                    force_pos=force_pos,
                    force_neg=force_neg,
                    peak=peak,
                    resample_to_shape=resample_to_shape,
                    t_start=time_rs[0],
                    max_scaler=max_scaler,
                    noise_scaler=noise_scaler                    
                )

                t0_idx = 0
                mag_ix = mag["t_ix"]
                ttp = mag["t"]
                mag_window = [mag["amplitude"]*i for i in window]
                
                relevant_samples = hrf_rs.values.squeeze()[t0_idx:mag_ix]      

            mag_indices = [
                utils.find_nearest(
                    relevant_samples,
                    i
                )[0]+t0_idx for i in mag_window
            ]

            # print(mag)
            # print(mag_window)
            # print(mag_indices)

            t0, t1 = [time_rs[i] for i in mag_indices]
            # print(t0, t1)

        slope_val, intercept, intsc, y_hat = self.get_slope_between_points(
            time_rs,
            hrf_rs.values.squeeze(),  # the magnitude signal
            t0,
            t1
        )

        if not np.isfinite(intsc) or verify_slope:
            
            # this y_lim will scale other subplots too
            tmp = hrf_rs.values.squeeze()
            y_lim = [
                tmp.min()-abs(tmp.min()*0.5),
                tmp.max()+(tmp.max()*0.1)
            ]

            kwargs = utils.update_kwargs(
                kwargs,
                "y_lim",
                y_lim
            )
            pl = self.plot_profile_for_debugging(
                hrf_rs,
                extra=[y_hat],
                add_vline={"pos": [t0, t1, ttp], "color": "k"},
                **kwargs
            )

            if not np.isfinite(intsc) and not nan_policy:
                raise ValueError("Could not extract onset time from profile; regression fit doesn't cross 0 ")

        return slope_val, intsc

    @classmethod
    def get_slope_between_points(
        self,
        time_array,
        signal_array,
        t0,
        t1
        ):
        # Find nearest indices to t0 and t1
        idx0 = np.argmin(np.abs(time_array - t0))
        idx1 = np.argmin(np.abs(time_array - t1))

        x0, x1 = time_array[idx0], time_array[idx1]
        y0, y1 = signal_array[idx0], signal_array[idx1]

        # Linear slope and intercept
        slope = (y1 - y0) / (x1 - x0) if x1 != x0 else np.nan
        intercept = y0 - slope * x0  # y = mx + b => b = y - mx

        # Calculate y_hat: the linear fit across the full time range
        y_hat = slope * time_array + intercept

        # Onset time: where line intersects y = 0
        intsc = -intercept / slope if slope != 0 else np.nan

        return slope, intercept, intsc, y_hat


    @classmethod
    def _correct_t0(
        self, 
        hrf,
        t_start=0
        ):

        time, time_col = self._get_time(hrf)

        # correct for 0
        hrf_corr = utils.select_from_df(
            hrf,
            expression=f"{time_col} >= {t_start}"
        )

        time_corr = np.array(utils.get_unique_ids(hrf_corr, id=time_col))

        return time_corr, hrf_corr

    @classmethod
    def _get_amplitude(self, hrf, force_pos=False, force_neg=False, peak=None,
        t_start=0, resample_to_shape=500, max_scaler=0.05, noise_scaler=1,
        verify_peaks=False, **kwargs):
        """Select peaks by time index, never by matching a repeated amplitude."""
        if force_pos and force_neg:
            raise ValueError('force_pos and force_neg are mutually exclusive')
        time, curve = self._resample(hrf, resample_to_shape=resample_to_shape)
        if curve.shape[1] != 1:
            raise ValueError('Peak extraction requires one profile at a time')
        valid = np.flatnonzero(time >= t_start)
        if not len(valid):
            raise ValueError('No HRF observations at or after t_start')
        signal_values = curve.iloc[:, 0].to_numpy()
        values = signal_values[valid]
        if peak is not None and (not isinstance(peak, (int, np.integer)) or peak < 1):
            raise ValueError('peak must be a positive integer or None')
        selected = None
        peaks = np.array([], dtype=int)
        if peak is not None:
            prominence = max(max_scaler * np.max(np.abs(values)), noise_scaler * np.std(values))
            positive = signal.find_peaks(values, prominence=prominence)[0]
            negative = signal.find_peaks(-values, prominence=prominence)[0]
            positive = positive[values[positive] > 0]
            negative = negative[values[negative] < 0]
            peaks = positive if force_pos else negative if force_neg else np.sort(np.r_[positive, negative])
            if len(peaks) >= peak:
                selected = int(peaks[peak - 1])
        if selected is None:
            selected = int(np.argmax(values) if force_pos else np.argmin(values) if force_neg else np.argmax(np.abs(values)))
        index = int(valid[selected])
        if verify_peaks:
            self.plot_profile_for_debugging(curve, peaks=valid[peaks].tolist(), **kwargs)
        return {'amplitude': signal_values[index], 't': time[index], 't_ix': index}

    @classmethod
    def _get_derivatives(
        self,
        hrf,
        **kwargs
        ):

        """_get_derivatives

        Computes the first and second derivatives of the HRF time-series.

        Parameters
        ----------
        hrf : pd.DataFrame
            The input HRF time-series.
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        dict
            Dictionary containing:
            - `"1st_deriv_magnitude"`: The first derivative peak magnitude.
            - `"1st_deriv_time_to_peak"`: Time to reach the first derivative peak.
            - `"2nd_deriv_magnitude"`: The second derivative peak magnitude.
            - `"2nd_deriv_time_to_peak"`: Time to reach the second derivative peak.

        Example
        ----------
        .. code-block:: python

            derivatives = HRFMetrics._get_derivatives(hrf_data)
        """

        # fetch time stamps
        time, time_col = self._get_time(hrf)

        kwargs = utils.update_kwargs(
            kwargs,
            "peak",
            1
        )

        # get derivatives
        first_derivative = np.gradient(
            hrf.values.squeeze(),
            time
        )[..., np.newaxis]

        second_derivative = np.gradient(
            first_derivative.squeeze(),
            time
        )[..., np.newaxis]

        # deriv
        ddict = {}
        for key, val in zip(["1st_deriv", "2nd_deriv"], [first_derivative, second_derivative]):
            df = pd.DataFrame(val, columns=[key])
            df[time_col] = time
            df.set_index([time_col], inplace=True)
            res = self._get_amplitude(
                df, 
                **kwargs
            )

            for i, e in res.items():
                ddict[f"{key}_{i}"] = e

        return ddict

    @classmethod
    def _get_auc(
        self,
        hrf,
        force_pos=False,
        force_neg=False,
        nan_policy=False,
        t_start=0,
        peak=None,
        resample_to_shape=500,
        max_scaler=0.05,
        noise_scaler=1        
        ):

        """_get_auc

        Computes the area under the curve (AUC) for the positive and undershoot phases of the HRF.

        Parameters
        ----------
        hrf : pd.DataFrame
            The input HRF time-series.
        force_pos : bool, optional
            If `True`, forces extraction from a positive response, by default `False`.
        force_neg : bool, optional
            If `True`, forces extraction from a negative response, by default `False`.
        nan_policy : bool, optional
            If `True`, assigns `NaN` if extraction fails, by default `False`.
        peak : int, optional
            Specifies which peak to use, by default `None`.

        Returns
        -------
        dict
            Dictionary containing:
            - `"pos_area"`: AUC of the positive HRF response.
            - `"undershoot"`: AUC of the negative undershoot.

        Example
        ----------
        .. code-block:: python

            auc_values = HRFMetrics._get_auc(hrf_data)
        """

        # get time
        time, time_col = self._get_time(hrf)

        # resample
        time_rs, hrf_rs = self._resample(
            hrf,
            resample_to_shape=resample_to_shape
        )

        # correct for 0
        time_corr, hrf_corr = self._correct_t0(
            hrf_rs,
            t_start=t_start
        )

        # get time
        dx = np.diff(time_rs)[0]

        # find index where t=0
        t0_idx = utils.find_nearest(time_rs, t_start)[0]

        # get amplitude of HRF
        mag = self._get_amplitude(
            hrf,
            force_pos=force_pos,
            force_neg=force_neg,
            peak=peak,
            t_start=t_start,
            resample_to_shape=resample_to_shape,
            max_scaler=max_scaler,
            noise_scaler=noise_scaler
        )

        # first check if the period before the peak has zero crossings
        zeros = np.zeros_like(hrf_rs.values.squeeze())[:mag["t_ix"]]
        xx = np.arange(0, zeros.shape[0])

        try:
            coords = utils.find_intersection(
                xx,
                zeros,
                hrf_rs.values.squeeze()[:mag["t_ix"]]
            )

            HAS_ZEROS_BEFORE_PEAK = True
        except BaseException:
            HAS_ZEROS_BEFORE_PEAK = False

        if HAS_ZEROS_BEFORE_PEAK:
            # if we found multiple coordinates, we have some bump/dip before
            # peak
            coords = sorted([int(i[0][0]) for i in coords])

            if len(coords) > 1:
                first_cross = coords[-1]
            else:
                # assume single zero point before peak
                first_cross = coords[0]

        else:
            # if no zeros before peak, take start of curve as starting curve
            first_cross = 0

        # print(f"1st crossing = {first_cross}")
        # find the next zero crossing after "first_cross" that is after peak
        # index
        zeros = np.zeros_like(hrf_rs.values.squeeze())
        xx = np.arange(0, zeros.shape[0])

        try:
            coords = utils.find_intersection(
                xx,
                zeros,
                hrf_rs.values.squeeze()
            )

            FAILED = False
        except BaseException:
            FAILED = True

        # if there's no zero-crossings, take end of interval
        third_cross = None
        if FAILED:
            second_cross = hrf_rs.shape[0]-1
        else:
            # filter coordinates for elements BEFORE peak
            coord_list = [
                int(i[0][0])
                for i in coords if int(i[0][0]) > mag["t_ix"]
            ]
            if len(coord_list) > 0:

                # sort
                coord_list.sort()
                # index first element as second zero
                second_cross = coord_list[0]

                # take last element as undershoot
                if len(coord_list) < 2:
                    third_cross = hrf_rs.values.squeeze().shape[0] - 1
            else:
                second_cross = hrf_rs.values.squeeze().shape[0] - 1

            # it's already at the end
            end_boundary = hrf_rs.values.squeeze().shape[0]
            if second_cross < end_boundary-2:
                # if multple coords after peak we might have negative undershoot
                if len(coord_list) > 1:
                    # check if sample are not too close to one another;AUC needs at
                    # least 2 samples
                    ix = 1
                    third_cross = coord_list[ix]
                    # print(coord_list)
                    # print(third_cross)
                    while (third_cross - second_cross) < 2:
                        ix += 1
                        if ix < len(coord_list):
                            third_cross = coord_list[ix]
                        else:
                            third_cross = end_boundary-1
            else:
                third_cross = None

        # print(f"3rd crossing = {third_cross}")
        # connect A and B (bulk of response)
        zeros = np.zeros_like(hrf_rs.iloc[first_cross:second_cross])
        xx = np.arange(0, zeros.shape[0])

        # print(f"first: {first_cross}\t| second: {second_cross}\t| third: {third_cross}")
        try:
            ab_area = metrics.auc(
                xx,
                hrf_rs.iloc[first_cross:second_cross]
                )*dx
        except Exception as e:
            if not nan_policy:
                pl = self.plot_profile_for_debugging(hrf)
                raise ValueError(f"Could not extract AUC from first {first_cross}-to-second {second_cross} crossing: {e}")
            else:
                ab_area = np.nan

        # check for under-/overshoot
        ac_area = np.nan
        if isinstance(third_cross, int):
            # print(f"1st cross: {first_cross}\t2nd cross: {second_cross}\t3rd cross: {third_cross}")
            zeros = np.zeros_like(hrf_rs.iloc[second_cross:third_cross])
            xx = np.arange(0, zeros.shape[0])

            try:
                ac_area = abs(
                    metrics.auc(
                        xx,
                        hrf_rs.iloc[second_cross:third_cross]
                    )
                )*dx
            except Exception as e:
                if not nan_policy:
                    pl = self.plot_profile_for_debugging(
                        hrf,
                        add_hline=0
                    )
                    raise ValueError(f"Could not extract AUC from second {second_cross}-to-third {third_cross} crossing: {e}")
                else:
                    ac_area = np.nan

        return {
            "pos_area": ab_area,
            "undershoot": ac_area
        }

    @classmethod
    def plot_hrf_auc(
        self,
        time,
        signal,
        t_start=0,
        figsize=(5,5)
        ):
        # Mask time and signal starting from t_start
        start_ix = np.argmin(np.abs(time - t_start))
        time_corr = time[start_ix:]
        signal_corr = signal[start_ix:]

        # Positive and negative masks
        pos_mask = signal_corr > 0
        neg_mask = signal_corr < 0

        pl = plotting.LazyLine(
            signal,
            xx=time,
            labels="signal",
            color="black", 
            line_width=3,
            add_hline=0,
            figsize=figsize,
            add_vline=start_ix,
            x_label="time (s)",
            y_label="signal",
            title="AUC visualization"
        )

        # Fill positive area
        pl.axs.fill_between(
            time_corr[pos_mask],
            0,
            signal_corr[pos_mask],
            color='tab:blue',
            alpha=0.4,
            label="Positive AUC"
        )

        # Fill negative area
        pl.axs.fill_between(
            time_corr[neg_mask],
            0,
            signal_corr[neg_mask],
            color='tab:red',
            alpha=0.4,
            label="Negative AUC"
        )

        return pl

    @classmethod
    def _get_auc_simple(
        self,
        hrf,
        t_start=0,
        resample_to_shape=500,
        nan_policy=True,
        normalize=False,
        verify_auc=False,
        **kwargs
        ):
        """
        Compute AUC for the positive and negative portions of an HRF time course.

        Parameters
        ----------
        hrf : pd.DataFrame
            DataFrame with time and signal columns.
        time_col : str, optional
            Name of the time column, by default "t".
        signal_col : str, optional
            Name of the signal column, by default "value".

        Returns
        -------
        dict
            Dictionary containing:
            - "pos_area": Area under positive part of the HRF.
            - "neg_area": Absolute area under negative part of the HRF.
        """

        # get time
        time, time_col = self._get_time(hrf)

        # resample
        time_rs, hrf_rs = self._resample(
            hrf,
            resample_to_shape=resample_to_shape
        )

        # correct for 0
        time_corr, hrf_corr = self._correct_t0(
            hrf_rs,
            t_start=t_start
        )

        hrf_corr_arr = hrf_corr.to_numpy(copy=True).squeeze()

        if normalize:
            scale = np.max(np.abs(hrf_corr_arr))
            if not np.isfinite(scale) or scale == 0:
                return {'pos_area': np.nan, 'neg_area': np.nan}
            hrf_corr_arr = hrf_corr_arr / scale

        # Separate positive and negative parts
        pos_mask = hrf_corr_arr > 0
        neg_mask = hrf_corr_arr < 0

        def compute_auc_safely(x, y):
            valid = np.isfinite(x) & np.isfinite(y)
            if np.any(valid) and x[valid].shape[0] > 1 and y[valid].shape[0] > 1:
                return abs(metrics.auc(x[valid], y[valid]))
            else:
                return 0.0  # No valid points → area is 0

        # Integrate the clipped curve on the entire axis. Removing opposite-
        # sign samples would bridge disconnected lobes and inflate the area.
        roots = FWHM._crossings(time_corr, hrf_corr_arr, 0.)
        integration_time = np.unique(np.r_[time_corr, roots])
        integration_values = np.interp(integration_time, time_corr, hrf_corr_arr)
        pos_area = compute_auc_safely(integration_time, np.maximum(integration_values, 0.))
        neg_area = compute_auc_safely(integration_time, np.minimum(integration_values, 0.))

        if verify_auc:
            extra_plot = []

            # Create masks and use NaNs for masking for plotting
            pos_curve = np.where(pos_mask, hrf_corr_arr, np.nan)
            neg_curve = np.where(neg_mask, hrf_corr_arr, np.nan)

            pl = self.plot_profile_for_debugging(
                hrf_corr,
                **kwargs
            )

            # Use fill_between on the Axes object to shade AUC regions
            ax = pl.axs  # Get the matplotlib Axes object

            ax.fill_between(
                time_corr,
                np.where(pos_mask, hrf_corr_arr, np.nan),
                y2=0,
                color="tab:green",
                alpha=0.3,
                label="Positive AUC"
            )

            ax.fill_between(
                time_corr,
                np.where(neg_mask, hrf_corr_arr, np.nan),
                y2=0,
                color="tab:red",
                alpha=0.3,
                label="Negative AUC"
            )

            ax.legend(
                loc="best",
                frameon=False
            )

        return {
            "pos_area": pos_area,
            "neg_area": neg_area,
            "total_area": pos_area + neg_area
        }

        
    @classmethod
    def _get_fwhm(
        self,
        hrf,
        force_pos=False,
        force_neg=False,
        nan_policy=False,
        add_fct=0.5,
        peak=None,
        t_start=0,
        resample_to_shape=500,
        max_scaler=0.05,
        noise_scaler=1      ,
        verify_fwhm=False,
        **kwargs  
        ):

        """_get_fwhm

        Computes the full-width at half-maximum (FWHM) of the HRF.

        Parameters
        ----------
        hrf : pd.DataFrame
            The input HRF time-series.
        force_pos : bool, optional
            If `True`, forces extraction from a positive response, by default `False`.
        force_neg : bool, optional
            If `True`, forces extraction from a negative response, by default `False`.
        nan_policy : bool, optional
            If `True`, assigns `NaN` if extraction fails, by default `False`.
        add_fct : float, optional
            Scaling factor for extending the search window for FWHM calculation, by default `0.5`.
        peak : int, optional
            Specifies which peak to use, by default `None`.

        Returns
        -------
        dict
            Dictionary containing:
            - `"fwhm"`: The computed FWHM value.
            - `"half_rise"`: Time at half of the rising phase.
            - `"half_max"`: The half-max amplitude value.
            - `"obj"`: Instance of `linescanning.fitting.FWHM`, storing additional properties.

        Example
        ----------
        .. code-block:: python

            fwhm_values = HRFMetrics._get_fwhm(hrf_data)
        """

        # fetch time stamps
        time, time_col = self._get_time(hrf)

        # get amplitude of HRF
        mag = self._get_amplitude(
            hrf,
            force_pos=force_pos,
            force_neg=force_neg,
            peak=peak,
            t_start=t_start,
            resample_to_shape=resample_to_shape,
            max_scaler=max_scaler,
            noise_scaler=noise_scaler            
        )

        negative = False
        if mag["amplitude"] < 0:
            negative = True

        # check time stamps
        if time.shape[0] != hrf.values.shape[0]:
            raise ValueError(
                f"Shape of time dimension ({time.shape[0]}) does not match dimension of HRF ({hrf.values.shape[0]})")

        # define index period around magnitude; add 20% to avoid FWHM errors
        end_ix = mag["t_ix"] + mag["t_ix"]
        end_ix += int(end_ix * add_fct)

        try:
            # get fwhm around max amplitude
            fwhm_val = FWHM(
                time,
                hrf.values,
                negative=negative,
                amplitude=mag["amplitude"],
                peak_time=mag["t"]
            )
            if fwhm_val.status != 'ok' and not nan_policy:
                raise ValueError(fwhm_val.status)

            fwhm_dict = {
                "fwhm": fwhm_val.fwhm,
                "half_rise": fwhm_val.t0_,
                "half_max": fwhm_val.half_max,
                "obj": fwhm_val
            }

            if verify_fwhm:
                pl = self.plot_profile_for_debugging(
                    hrf,
                    **kwargs
                )

                xlim = pl.axs.get_xlim()
                tot = sum(list(np.abs(xlim)))
                start = (fwhm_val.t0_-xlim[0])/tot
                pl.axs.axhline(
                    fwhm_val.half_max,
                    xmin=start,
                    xmax=start+fwhm_val.fwhm/tot,
                    color="k",
                    linewidth=0.5
                )

        except Exception as e:
            if not nan_policy:
                pl = self.plot_profile_for_debugging(hrf)
                raise ValueError(f"Could not extract FWHM from this profile: {e}")  # noqa: F821
            else:
                fwhm_dict = {
                    "fwhm": np.nan,
                    "half_rise": np.nan,
                    "half_max": np.nan,
                    "obj": np.nan
                }

        return fwhm_dict

    @classmethod
    def _get_single_hrf_metrics(
        self,
        hrf,
        TR=None,
        force_pos=False,
        force_neg=False,
        plot=False,
        nan_policy=False,
        debug=False,
        shift=None,
        peak=None,
        resample_to_shape=500,
        t_start=0,
        max_scaler=0.05,
        noise_scaler=1,
        verify_peaks=False,
        verify_slope=False,
        verify_auc=False,
        verify_fwhm=False,
        **kwargs
        ):

        """_get_single_hrf_metrics

        Extracts all HRF metrics for a single time-series profile.

        Parameters
        ----------
        hrf : pd.DataFrame
            The input HRF time-series.
        TR : float, optional
            Repetition time in seconds.
        force_pos : bool, optional
            If `True`, forces extraction from a positive response, by default `False`.
        force_neg : bool, optional
            If `True`, forces extraction from a negative response, by default `False`.
        plot : bool, optional
            If `True`, plots the HRF profile for debugging, by default `False`.
        nan_policy : bool, optional
            If `True`, assigns `NaN` if extraction fails, by default `False`.
        debug : bool, optional
            If `True`, prints debugging information, by default `False`.
        shift : int or float, optional
            Shifts the HRF profile before analysis, by default `None`.
        peak : int, optional
            Specifies which peak to use, by default `None`.
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing extracted HRF parameters.
        dict
            Dictionary containing FWHM-related metrics.
        pd.DataFrame
            Processed HRF time-series.

        Example
        ----------
        .. code-block:: python

            hrf_metrics, fwhm_info, processed_hrf = HRFMetrics._get_single_hrf_metrics(hrf_data)
        """

        # verify input type
        hrf = self._verify_input(
            hrf,
            TR=TR,
            shift=shift
        )
        
        # 1. Define verification flags
        verification_flags = {
            "verify_peaks": verify_peaks,
            "verify_slope": verify_slope,
            "verify_auc": verify_auc,
            "verify_fwhm": verify_fwhm
            # Add more flags as needed
        }

        # 2. Count how many are True
        active_flags = {k: v for k, v in verification_flags.items() if v}
        ncols = len(active_flags)

        # 3. Create subplots and map axes
        axs_dict = {}
        if ncols > 0:
            fig, axs = plt.subplots(
                ncols=ncols,
                figsize=(ncols * 5, 4),
                sharey=True
            )
            
            fig.suptitle(
                list(hrf.columns)[0],
                fontweight="bold",
                fontsize=26,
                y=1.12
            )

            # Ensure axs is iterable
            if ncols == 1:
                axs = [axs]

            for ax, flag in zip(axs, active_flags.keys()):
                axs_dict[flag] = ax

        
        utils.verbose(f"var\t= {abs(hrf.var().iloc[0])}", debug)
        # utils.verbose(f"TR\t= {TR}", debug)
        # utils.verbose(f"shift\t= {shift}", debug)

        orig_hrf = hrf.copy()

        if plot:
            pl = self.plot_profile_for_debugging(
                orig_hrf,
                add_hline=0,
            )

        # get magnitude and amplitude
        mag = self._get_amplitude(
            orig_hrf,
            force_pos=force_pos,
            force_neg=force_neg,
            peak=peak,
            t_start=t_start,
            resample_to_shape=resample_to_shape,
            max_scaler=max_scaler,
            noise_scaler=noise_scaler,
            verify_peaks=verify_peaks,
            axs=axs_dict.get("verify_peaks"),
            title="peaks"
        )

        # fwhm
        fwhm_obj = self._get_fwhm(
            orig_hrf,
            force_pos=force_pos,
            force_neg=force_neg,
            nan_policy=nan_policy,
            peak=peak,
            t_start=t_start,
            resample_to_shape=resample_to_shape,
            max_scaler=max_scaler,
            noise_scaler=noise_scaler,
            verify_fwhm=verify_fwhm,
            axs=axs_dict.get("verify_fwhm"),
            title="FWHM"            
        )

        # rise slope
        rise_slope = self._get_riseslope_siero(
            orig_hrf,
            force_pos=force_pos,
            force_neg=force_neg,
            nan_policy=nan_policy,
            peak=peak,
            t_start=t_start,
            resample_to_shape=resample_to_shape,
            max_scaler=max_scaler,
            noise_scaler=noise_scaler,
            verify_slope=verify_slope,
            axs=axs_dict.get("verify_slope"),
            title="slope & intercept"
        )

        # positive area + undershoot
        auc = self._get_auc(
            orig_hrf,
            force_pos=force_pos,
            force_neg=force_neg,
            nan_policy=nan_policy,
            peak=peak,
            t_start=t_start,
            resample_to_shape=resample_to_shape,
            max_scaler=max_scaler,
            noise_scaler=noise_scaler,
        )

        # positive area + undershoot
        simple_auc = self._get_auc_simple(
            orig_hrf,
            nan_policy=nan_policy,
            t_start=t_start,
            resample_to_shape=resample_to_shape,
            verify_auc=verify_auc,
            axs=axs_dict.get("verify_auc"),
            title="AUC"            
        )
        
        # normalized peak positive area + undershoot
        simple_auc_norm = self._get_auc_simple(
            orig_hrf,
            normalize=True,
            nan_policy=nan_policy,
            t_start=t_start,
            resample_to_shape=resample_to_shape
        )

        # positive area + undershoot
        if not isinstance(peak, int):
            peak = 1

        deriv = self._get_derivatives(
            hrf,
            force_pos=force_pos,
            force_neg=force_neg,
            peak=peak,
            t_start=t_start,
            resample_to_shape=resample_to_shape,
            max_scaler=max_scaler,
            noise_scaler=noise_scaler,
        )

        ddict = {
            "magnitude": [mag["amplitude"]],
            "magnitude_ix": [mag["t_ix"]],
            "fwhm": [fwhm_obj["fwhm"]],
            "fwhm_obj": [fwhm_obj["obj"]],
            "time_to_peak": [mag["t"]],
            "half_rise_time": [fwhm_obj["half_rise"]],
            "half_max": [fwhm_obj["half_max"]],
            "rise_slope": [rise_slope[0]],
            "onset_time": [rise_slope[1]],
            "positive_area": [auc["pos_area"]],
            "undershoot": [auc["undershoot"]],
            "auc_simple_pos": [simple_auc["pos_area"]],
            "auc_simple_neg": [simple_auc["neg_area"]],
            "auc_simple_total": [simple_auc["total_area"]],
            "auc_simple_pos_norm": [simple_auc_norm["pos_area"]],
            "auc_simple_neg_norm": [simple_auc_norm["neg_area"]],
            "auc_simple_total_norm": [simple_auc_norm["total_area"]],            
            "1st_deriv_magnitude": [deriv["1st_deriv_amplitude"]],
            "1st_deriv_time_to_peak": [deriv["1st_deriv_t"]],
            "2nd_deriv_magnitude": [deriv["2nd_deriv_amplitude"]],
            "2nd_deriv_time_to_peak": [deriv["2nd_deriv_t"]],
        }

        df = pd.DataFrame(ddict)

        return df, fwhm_obj, hrf


class FWHM():
    def __init__(self, x, hrf, resample=500, negative=False, amplitude=None, peak_time=None):
        """Measure crossings bracketing the selected peak; return NaN if incomplete."""
        self.x = np.asarray(x, dtype=float).reshape(-1)
        self.hrf = np.asarray(hrf, dtype=float).reshape(-1)
        self.resample, self.negative, self.amplitude = resample, negative, amplitude
        if len(self.x) != len(self.hrf) or len(self.x) < 2:
            raise ValueError('FWHM requires equally sized time/signal arrays with at least two points')
        if not np.isfinite(self.x).all() or not np.isfinite(self.hrf).all() or np.any(np.diff(self.x) <= 0):
            raise ValueError('FWHM requires finite values and increasing time coordinates')
        size = max(len(self.x), int(resample))
        self.use_x = np.unique(np.r_[self.x, np.linspace(self.x[0], self.x[-1], size)])
        self.use_rf = np.interp(self.use_x, self.x, self.hrf)
        extremum = np.argmin(self.use_rf) if negative else np.argmax(self.use_rf)
        peak_time = self.use_x[extremum] if peak_time is None else float(peak_time)
        self.half_max = (float(amplitude) if amplitude is not None else self.use_rf[extremum]) / 2
        roots = self._crossings(self.use_x, self.use_rf, self.half_max)
        self.hmx = [(np.array([root]), np.array([self.half_max])) for root in roots]
        before, after = roots[roots < peak_time], roots[roots > peak_time]
        self.status = 'ok' if len(before) and len(after) else 'missing_half_max_crossings'
        self.ts = [before[-1], after[0]] if self.status == 'ok' else []
        self.t0_, self.t1_ = self.ts if self.ts else (np.nan, np.nan)
        self.fwhm = self.t1_ - self.t0_

    @staticmethod
    def _crossings(x, y, level):
        delta = y - level
        roots = list(x[delta == 0])
        for i in np.flatnonzero(delta[:-1] * delta[1:] < 0):
            roots.append(x[i] + (x[i+1] - x[i]) * (-delta[i]) / (delta[i+1] - delta[i]))
        return np.unique(roots)

    def half_max_x(self, x, y, amplitude=None):
        x, y = np.asarray(x), np.asarray(y)
        amplitude = (np.min(y) if self.negative else np.max(y)) if amplitude is None else amplitude
        return [(np.array([root]), np.array([amplitude / 2])) for root in self._crossings(x, y, amplitude / 2)]

def error_function(
        parameters,
        args,
        data,
        objective_function):
    """
    Parameters
    ----------
    parameters : list or ndarray
        A tuple of values representing a model setting.
    args : dictionary
        Extra arguments to `objective_function` beyond those in `parameters`.
    data : ndarray
       The actual, measured time-series against which the model is fit.
    objective_function : callable
        The objective function that takes `parameters` and `args` and
        produces a model time-series.

    Returns
    -------
    error : float
        The residual sum of squared errors between the prediction and data.
    """
    return np.nan_to_num(
        np.sum((data - objective_function(parameters, **args))**2), nan=1e12)
    # return
    # 1-np.nan_to_num(pearsonr(data,np.nan_to_num(objective_function(*list(parameters),
    # **args)[0]))[0])


def double_gamma_with_d(a1, a2, b1, b2, c, d1, d2, x=None, negative=False):
    """Evaluate a causal HRF without moving onset zero or mutating x."""
    if x is None:
        raise ValueError('Provide HRF time coordinates with x')
    time = np.asarray(x, dtype=float)
    if min(a1, a2, b1, b2, d1, d2) <= 0:
        raise ValueError('HRF shape, scale and delay parameters must be positive')
    positive_time = np.maximum(time, 0.)
    curve = ((positive_time / d1)**a1 * np.exp(-(positive_time - d1) / b1)
             - c * (positive_time / d2)**a2 * np.exp(-(positive_time - d2) / b2))
    curve = np.where(time < 0, 0., curve)
    return -curve if negative else curve


def make_prediction(
    parameters,
    onsets=None,
    scan_length=None,
    TR=1.32,
    osf=1,
    cov_as_ampl=None,
    negative=False,
    interval=[0, 25]
):

    # this correction is needed to get the HRF in the correct time domain
    dt = 1 / (osf / TR)

    time_points = _sample_time_grid(interval, dt)
    hrf = [double_gamma_with_d(
        *parameters,
        x=time_points,
        negative=negative)]

    ev_ = glm.make_stimulus_vector(
        onsets,
        scan_length=scan_length,
        TR=TR,
        osf=osf,
        cov_as_ampl=cov_as_ampl
    )

    ev_conv = glm.convolve_hrf(
        hrf,
        ev_,
        TR=TR,
        osf=osf,
        time=time_points,
        # make_figure=True
    )

    ev_conv_rs = glm.resample_stim_vector(ev_conv, scan_length)
    key = list(ev_conv_rs.keys())[0]

    return ev_conv_rs[key].squeeze()


def iterative_search(
        data,
        onsets,
        starting_params=[6, 12, 0.9, 0.9, 0.35, 5.4, 10.8],
        bounds=None,
        method='trust-constr',
        constraints=None,
        cov_as_ampl=None,
        TR=1.32,
        osf=1,
        interval=[0, 25],
        xtol=1e-4,
        ftol=1e-4,
):
    """iterative_search

    Iterative search function to find the best set of parameters that describe the HRF across a full timeseries. During the
    optimization, each parameter is adjusted and a new prediction is formed until the variance explained of this prediction is
    maximized.

    Parameters
    ----------
    data: np.ndarray
        Input must be an array (voxels,timepoints)
    onsets: pd.DataFrame
        Dataframe indexed on subject, run, and event_type (as per :class:`lazyfmri.dataset.ParseExpToolsFile`)
    verbose: bool, optional
        Print progress to the terminal, by default False
    TR: float, optional
        Repetition time, by default 1.32
    osf: int, optional
        Oversampling factor to account for decimals in onset times, by default 100
    starting_params: list, optional
        Starting parameters of the HRF, by default [6,12,0.9,0.9,0.35,5.4,10.8]. This is generally a good start.
    n_jobs: int, optional
        Number of jobs to use, by default 1
    bounds: list, optional
        Specific bounds for each parameter, by default None
    resample_to_shape: int, optional
        Resample the final profiles to a certain shape, by default None
    xtol: float, optional
        x-tolerance of the fitter, by default 1e-4
    ftol: float, optional
        f-tolerance of the fitter, by default 1e-4
    interval: list, optional
        Interval used to define the full HRF profile, by default [0,30]
    read_index: bool, optional
        Copy the index from the input dataframe, by default False
    """
    # data = (voxels,time)
    if data.ndim > 1:
        data = data.squeeze()

    scan_length = data.shape[0]

    # args
    args = {
        "onsets": onsets,
        "scan_length": scan_length,
        "cov_as_ampl": cov_as_ampl,
        "TR": TR,
        "osf": osf,
        "interval": interval
    }

    # run for both negative and positive; return parameters of best r2
    res = {}
    for lbl, nnp in zip(["pos", "neg"], [False, True]):
        res[lbl] = {}
        args["negative"] = nnp
        output = minimize(
            error_function,
            starting_params,
            args=(
                args,
                data,
                make_prediction),
            method=method,
            bounds=bounds,
            tol=ftol,
            options=dict(xtol=xtol)
        )

        pred = make_prediction(
            output['x'],
            **args
        )

        res[lbl]["pars"] = output['x']
        res[lbl]["r2"] = metrics.r2_score(data, pred)

    r2_pos = res["pos"]["r2"]
    r2_neg = res["neg"]["r2"]
    if r2_pos > r2_neg:
        return res["pos"]["pars"], "pos"
    else:
        return res["neg"]["pars"], "neg"

    # return starting_params, "pos"


class FitHRFparams():
    """FitHRFparams

    Similar to Marco's pRF modeling, we can also estimate the best set of parameters that describe the HRF. This has slightly
    more degrees of freedom compared to using basis sets, but is not as unconstrained as FIR.

    Parameters
    ----------
    data: np.ndarray
        Input must be an array (voxels,timepoints)
    onsets: pd.DataFrame
        Dataframe indexed on subject, run, and event_type (as per :class:`lazyfmri.dataset.ParseExpToolsFile`)
    verbose: bool, optional
        Print progress to the terminal, by default False
    TR: float, optional
        Repetition time, by default 1.32
    osf: int, optional
        Oversampling factor to account for decimals in onset times, by default 100
    starting_params: list, optional
        Starting parameters of the HRF, by default [6,12,0.9,0.9,0.35,5.4,10.8]
    n_jobs: int, optional
        Number of jobs to use, by default 1
    bounds: list, optional
        Specific bounds for each parameter, by default None
    resample_to_shape: int, optional
        Resample the final profiles to a certain shape, by default None
    xtol: float, optional
        x-tolerance of the fitter, by default 1e-4
    ftol: float, optional
        f-tolerance of the fitter, by default 1e-4
    interval: list, optional
        Interval used to define the full HRF profile, by default [0,30]
    read_index: bool, optional
        Copy the index from the input dataframe, by default False
    """

    def __init__(
            self,
            data,
            onsets,
            verbose=False,
            TR=1.32,
            osf=1,
            starting_params=[6, 12, 0.9, 0.9, 0.35, 5.4, 10.8],
            n_jobs=1,
            bounds=None,
            resample_to_shape=None,
            xtol=1e-4,
            ftol=1e-4,
            method="trust-constr",
            cov_as_ampl=None,
            interval=[0, 30],
            read_index=False,
            parallel=True,
            **kwargs):

        self.data = data
        self.onsets = onsets
        self.verbose = verbose
        self.TR = TR
        self.osf = osf
        self.interval = list(interval)
        self.starting_params = starting_params
        self.n_jobs = n_jobs
        self.bounds = bounds
        self.xtol = xtol
        self.ftol = ftol
        self.cov_as_ampl = cov_as_ampl
        self.resample_to_shape = resample_to_shape
        self.read_index = read_index
        self.parallel = parallel
        self.method = method

        # set default bounds that can be updated with kwargs
        if not isinstance(self.bounds, list):
            self.bounds = [
                (4, 8),
                (10, 14),
                (0.8, 1.2),
                (0.8, 1.2),
                (0, 0.5),
                (0, 10),
                (5, 15)
            ]

        self.__dict__.update(kwargs)

        # set number of jobs if None is specified (default = 1)
        if not isinstance(self.n_jobs, int):
            self.n_jobs = self.data.shape[0]

    def iterative_fit(self):

        # data = (voxels,time)
        if self.data.ndim < 2:
            self.data = self.data[np.newaxis, ...]

        if self.parallel:
            self.tmp_results = Parallel(self.n_jobs, verbose=self.verbose)(
                delayed(iterative_search)(
                    self.data[i, :],
                    self.onsets,
                    starting_params=self.starting_params,
                    TR=self.TR,
                    bounds=self.bounds,
                    xtol=self.xtol,
                    ftol=self.ftol,
                    cov_as_ampl=self.cov_as_ampl,
                    osf=self.osf,
                    interval=self.interval,
                    method=self.method
                ) for i in range(self.data.shape[0])
            )
        else:
            self.tmp_results = []
            for i in range(self.data.shape[0]):
                tt = iterative_search(
                    self.data[i, :],
                    self.onsets,
                    starting_params=self.starting_params,
                    TR=self.TR,
                    bounds=self.bounds,
                    xtol=self.xtol,
                    ftol=self.ftol,
                    cov_as_ampl=self.cov_as_ampl,
                    osf=self.osf,
                    interval=self.interval,
                    method=self.method
                )
                self.tmp_results.append(tt)

        # parse into array
        self.iterative_search_params = np.array(
            [self.tmp_results[i][0] for i in range(self.data.shape[0])])
        self.prof_sign = [self.tmp_results[i][1]
                          for i in range(self.data.shape[0])]

        # self.iterative_search_params = np.array(self.starting_params)[np.newaxis,...]
        # self.prof_sign = None

        self.force_neg = [False for _ in range(self.data.shape[0])]
        self.force_pos = [True for _ in range(self.data.shape[0])]

        if isinstance(self.prof_sign, list):
            for ix, el in enumerate(self.prof_sign):
                if el == "neg":
                    self.force_neg[ix] = True
                    self.force_pos[ix] = False

        # also immediately create profiles
        self.profiles_from_parameters(
            resample_to_shape=self.resample_to_shape,
            negative=self.prof_sign,
            read_index=self.read_index)

        self.hrf_pars = HRFMetrics(
            self.hrf_profiles,
            force_neg=self.force_neg,
            force_pos=self.force_pos
        ).return_metrics()

        self.estimates = pd.DataFrame(self.iterative_search_params, columns=[
                                      "a1", "a2", "b1", "b2", "c", "d1", "d2"])

        self.estimates["subject"] = utils.get_unique_ids(
            self.hrf_profiles, id="subject")[0]
        self.estimates["run"] = utils.get_unique_ids(
            self.hrf_profiles, id="run")[0]
        self.estimates["event_type"] = utils.get_unique_ids(
            self.hrf_profiles, id="event_type")[0]

    def profiles_from_parameters(
            self,
            resample_to_shape=None,
            negative=None,
            read_index=False):

        assert hasattr(
            self, "iterative_search_params"), "No parameters found, please run iterative_fit()"

        dt = self.TR / self.osf
        time_points = _sample_time_grid(self.interval, dt)

        hrfs = []
        preds = []
        pars = []
        for i in range(self.iterative_search_params.shape[0]):
            neg = False
            if isinstance(negative, list):
                if negative[i] in ["negative", "neg"]:
                    neg = True

            pars = list(self.iterative_search_params[i, :])
            hrf = double_gamma_with_d(*pars, x=time_points, negative=neg)

            pred = make_prediction(
                pars,
                onsets=self.onsets,
                scan_length=self.data.shape[1],
                TR=self.TR,
                osf=self.osf,
                cov_as_ampl=self.cov_as_ampl,
                interval=self.interval,
                negative=neg
            )

            # resample to specified length
            if isinstance(resample_to_shape, int):
                hrf = glm.resample_stim_vector(hrf, resample_to_shape)

            hrfs.append(hrf)
            preds.append(pred)

        hrf_profiles = np.array(hrfs)
        predictions = np.array(preds)

        if isinstance(resample_to_shape, int):
            time_points = np.linspace(time_points[0], time_points[-1], num=resample_to_shape)

        self.hrf_profiles = pd.DataFrame(hrf_profiles.T)
        self.hrf_profiles["time"] = time_points
        self.hrf_profiles["covariate"] = "intercept"
        self.predictions = pd.DataFrame(predictions.T)
        self.predictions["t"] = list(
            np.arange(0, self.data.shape[1]) * self.TR)

        # read indices from onset dataframe
        self.prof_indices = ["covariate", "time"]
        self.pred_indices = ["t"]

        self.custom_indices = []
        for el in ["subject", "run", "event_type"]:
            try:
                el_in_df = utils.get_unique_ids(self.onsets, id=el)
            except BaseException:
                el_in_df = None

            if isinstance(el_in_df, list):
                self.custom_indices.append(el)

                for df in [self.hrf_profiles, self.predictions]:
                    df[el] = el_in_df[0]

        if len(self.custom_indices) > 0:
            self.prof_indices = self.custom_indices + self.prof_indices
            self.pred_indices = self.custom_indices + self.pred_indices

        self.hrf_profiles = self.hrf_profiles.set_index(self.prof_indices)
        self.predictions = self.predictions.set_index(self.pred_indices)


class Epoch(InitFitter):

    """Epoch

    A class to extract timepoints within a specified interval based on onset times for all subjects, tasks, and runs 
    in a dataframe. The extracted epochs can be indexed on `subject`, `run`, `event_type`, `epoch`, and `t` 
    to facilitate event-related analyses.

    Parameters
    ----------
    func : pd.DataFrame
        Dataframe containing the fMRI data indexed on subject, run, and time (`t`).
        Expected format: output of :func:`lazyfmri.dataset.Datasets.fetch_fmri()`.
    onsets : pd.DataFrame
        Dataframe containing the event onset timings indexed on subject, run, and event type.
        Expected format: output of :func:`lazyfmri.dataset.Datasets.fetch_onsets()`.
    TR : float, optional
        Repetition time (TR) in seconds. Used to calculate the sampling frequency.
        Default is `0.105` seconds.
    interval : list, optional
        Time interval around each event onset to extract the epoch data. 
        Default is `[-2, 14]`, meaning extraction starts 2 seconds before and ends 14 seconds after the event.
    merge : bool, optional
        Whether to concatenate runs before extracting epochs. Default is `False`.

    Example
    -------
    .. code-block:: python

        from lazyfmri import fitting

        # Instantiate the Epoch class
        sub_ep = fitting.Epoch(
            func,
            onsets,
            TR=0.105,
            interval=[-2, 14]
        )

        # Extract the epoched dataframe
        sub_df = sub_ep.df_epoch.copy()
    """

    def __init__(
        self,
        func,
        onsets,
        TR=0.105,
        interval=[-2, 14],
        merge=False,
        verbose=False
        ):

        self.func = func
        self.onsets = onsets
        self.TR = TR
        self.interval = list(interval)
        self.merge = merge
        self.verbose = verbose

        # prepare data
        super().__init__(
            self.func,
            self.onsets,
            self.TR,
            merge=self.merge
        )

        # epoch data based on present identifiers (e.g., task/run)
        self.epoch_input()

    @staticmethod
    def correct_baseline(d_, bsl=20, verbose=False):
        """Subtract each signal's initial-sample mean without changing the input."""
        values = np.asarray(d_, dtype=float)
        if not isinstance(bsl, (int, np.integer)) or not 1 <= bsl <= len(values):
            raise ValueError('bsl must be between 1 and the number of samples')
        corrected = values - values[:bsl].mean(axis=0)
        if isinstance(d_, pd.DataFrame):
            return pd.DataFrame(corrected, index=d_.index, columns=d_.columns)
        if isinstance(d_, pd.Series):
            return pd.Series(corrected, index=d_.index, name=d_.name)
        return corrected

    @staticmethod
    def _get_epoch_timepoints(interval, TR):
        """TR-spaced offsets within the half-open requested interval."""
        start, stop = map(float, interval)
        if not np.isfinite([start, stop, TR]).all() or TR <= 0 or stop <= start:
            raise ValueError('Specify a positive TR and increasing finite interval')
        first = int(np.ceil(start / TR - 1e-10))
        last = int(np.ceil(stop / TR - 1e-10))
        return np.arange(first, last) * TR

    def epoch_events(self, df_func, df_onsets, index=False):
        """Extract full epochs without wrapping negative indices or relabeling samples.

        Event onsets are aligned to their nearest acquired sample; offsets remain
        exact multiples of TR. Boundary epochs are skipped with a warning.
        """
        identities = [key for key in ('subject', 'session', 'task', 'run') if key in df_func.index.names]
        for key in identities:
            if df_func.index.get_level_values(key).nunique() != 1:
                raise ValueError('epoch_events requires one acquisition at a time')
        times = df_func.index.get_level_values('t').to_numpy(dtype=float)
        order = np.argsort(times)
        times, data = times[order], df_func.iloc[order]
        if len(times) > 1 and not np.allclose(np.diff(times), self.TR):
            raise ValueError('Epoch extraction requires regularly sampled data')
        relative_time = self._get_epoch_timepoints(self.interval, self.TR)
        offsets = np.rint(relative_time / self.TR).astype(int)
        output = []
        events = df_onsets.reset_index()
        for event, group in events.groupby('event_type', sort=False):
            if event in ('response', 'blink'):
                continue
            for epoch, (_, row) in enumerate(group.iterrows()):
                onset = float(row['onset'])
                if not np.isfinite(onset):
                    raise ValueError('Event onset must be finite')
                center = int(np.argmin(np.abs(times - onset)))
                indices = center + offsets
                if not len(indices) or indices.min() < 0 or indices.max() >= len(data) or onset < times[0] or onset > times[-1]:
                    warnings.warn(f'Skipping incomplete epoch: event={event}, onset={onset}', RuntimeWarning)
                    continue
                result = pd.DataFrame(data.iloc[indices].to_numpy(copy=True), columns=data.columns)
                for field in identities:
                    result[field] = data.index.get_level_values(field)[0]
                result['event_type'], result['epoch'], result['t'] = event, epoch, relative_time
                output.append(result)
        fields = identities + ['event_type', 'epoch', 't']
        result = pd.concat(output, ignore_index=True) if output else pd.DataFrame(columns=list(data.columns) + fields)
        return result.set_index(fields) if index else result

    def epoch_runs(
        self,
        df_func,
        df_onsets,
        index=False,
        ):

        """Extract epochs across multiple runs.

        Loops over all available runs and applies :func:`lazyfmri.fitting.Epoch.epoch_events()` to extract event-specific epochs for each run.

        Parameters
        ----------
        df_func : pd.DataFrame
            Functional data indexed on `subject`, `run`, and `t`.
        df_onsets : pd.DataFrame
            Onset data indexed on `subject`, `run`, and `event_type`.
        index : bool, optional
            Whether to set hierarchical index on the output dataframe. Default is `False`.

        Returns
        -------
        pd.DataFrame
            Extracted epochs for all runs combined.
        """

        # loop through runs
        self.run_ids = utils.get_unique_ids(df_func, id="run")
        # print(f"task-{task}\t| runs = {run_ids}")
        run_df = []
        for run in self.run_ids:

            expr = f"run = {run}"
            run_func = utils.select_from_df(df_func, expression=expr)
            run_stims = utils.select_from_df(df_onsets, expression=expr)

            # get epochs
            df = self.epoch_events(
                run_func,
                run_stims,
                index=index
            )

            run_df.append(df)

        run_df = pd.concat(run_df)

        return run_df

    def epoch_tasks(
        self,
        df_func,
        df_onsets
        ):

        """Extract epochs across multiple tasks.

        Loops over all tasks in the dataset, extracts task-specific functional data, and applies :func:`lazyfmri.fitting.Epoch.epoch_runs()` 
        to process all runs within each task.

        Parameters
        ----------
        df_func : pd.DataFrame
            Functional data indexed on `subject`, `task`, `run`, and `t`.
        df_onsets : pd.DataFrame
            Onset data indexed on `subject`, `task`, `run`, and `event_type`.

        Returns
        -------
        pd.DataFrame
            Extracted epochs for all tasks combined.
        """

        # read task IDs
        self.task_ids = utils.get_unique_ids(df_func, id="task")

        # loop through task IDs
        task_df = []
        for task in self.task_ids:

            # extract task-specific dataframes
            expr = f"task = {task}"
            task_func = utils.select_from_df(df_func, expression=expr)
            task_stims = utils.select_from_df(df_onsets, expression=expr)

            df = self.epoch_runs(
                task_func,
                task_stims,
                index=False
            )

            df["task"] = task
            task_df.append(df)

        return pd.concat(task_df)

    def epoch_subjects(self, df_func, df_onsets):
        """Group on all acquisition fields rather than subject/run alone."""
        identities = [key for key in ('subject', 'session', 'task', 'run') if key in df_func.index.names]
        output = []
        for key, data in df_func.groupby(level=identities, sort=False):
            key = key if isinstance(key, tuple) else (key,)
            events = df_onsets
            for field, value in zip(identities, key):
                events = events.loc[events.index.get_level_values(field) == value]
            output.append(self.epoch_events(data, events, index=True))
        self.idx = identities + ['event_type', 'epoch', 't']
        return pd.concat(output)

    def epoch_input(self):

        """Main function to extract epochs from input data.

        Calls :func:`lazyfmri.fitting.Epoch.epoch_subjects()` to extract epochs at the subject level and formats the resulting dataframe.
        The final output is stored in `self.df_epoch`.

        Returns
        -------
        None
            The extracted epoch dataframe is stored as `self.df_epoch`.
        """
        # run Epoching
        self.df_epoch = self.epoch_subjects(
            self.func,
            self.onsets
        )

        # copy column names
        self.df_epoch.columns = self.func.columns


class GLM(InitFitter):

    """GLM

    Class to run a GLM for all subjects, tasks, and runs in a dataframe.

    Parameters
    ----------
    func: pd.DataFrame
        Dataframe as per the output of :func:`lazyfmri.dataset.Datasets.fetch_fmri()`, containing the fMRI data indexed on
        subject, run, and t.
    onsets: pd.DataFrame
        Dataframe as per the output of :func:`lazyfmri.dataset.Datasets.fetch_onsets()`, containing the onset timings data
        indexed on subject, run, and event_type.
    TR: float, optional
        Repetition time, by default 0.105. Use to calculate the sampling frequency (1/TR)
    interval: list, optional
        Interval to fit the regressors over, by default [0,12]
    merge: bool, optional
        Concatenate the runs before extracting the epochs

    """

    def __init__(
        self,
        func,
        onsets,
        TR=0.105,
        merge=False,
        **kwargs
        ):

        self.func = func
        self.onsets = onsets
        self.TR = TR
        self.merge = merge

        # prepare data
        super().__init__(
            self.func,
            self.onsets,
            self.TR,
            merge=self.merge
        )

        # epoch data based on present identifiers (e.g., task/run)
        self.glm_input(**kwargs)

    @classmethod
    def get_copes(
        self,
        evs,
        derivative=False,
        dispersion=False,
        add_intercept=True,
        **kwargs
        ):

        copes = []
        if not dispersion and not derivative:
            copes = np.eye(len(evs))
        else:
            start = 0
            for i in enumerate(evs):
                if dispersion and derivative:
                    fc = 3
                elif dispersion or derivative:
                    fc = 2
                else:
                    fc = 1

                n_cols = len(evs) * fc
                arr = np.zeros((n_cols))
                arr[start] = 1  # Derivative coefficients are nuisance terms, not amplitude.
                copes.append(arr)

                start += fc

        copes = np.array(copes)
        if add_intercept:
            icpt = np.zeros((len(evs), 1))
            copes = np.hstack((icpt, copes))

        return copes

    @classmethod
    def single_glm(
        self,
        func,
        onsets,
        **kwargs
        ):

        defaults = {
            "add_intercept": True,
            "hrf_pars": "glover",
            "derivative": True,
            "TR": 0.105,
            "osf": 100
        }

        # update kwargs with defaults
        for key, val in defaults.items():
            kwargs = utils.update_kwargs(
                kwargs,
                key,
                val
            )

        # decide contrasts
        evs = utils.get_unique_ids(onsets, id="event_type")
        c_vec = self.get_copes(
            evs,
            **kwargs
        )

        fitter = glm.GenericGLM(
            onsets,
            func,
            **kwargs
        )

        fitter.create_design()
        fitter.fit(copes=c_vec)

        fm = self.format_output(fitter)
        return fm

    @classmethod
    def format_fm(self, glm_obj):

        # full model predictions = dot product of design matrix & betas
        pred_fm = glm_obj.results["x_conv"] @ glm_obj.results["betas"]

        # input will be dataframe, per definition, so copy indices
        return pd.DataFrame(pred_fm, index=glm_obj.orig.index)

    @classmethod
    def format_ev(self, glm_obj):

        evs = utils.get_unique_ids(glm_obj.onsets, id="event_type")
        dm_columns = glm_obj.results["dm"].columns.to_list()
        preds = []

        for ix, ev in enumerate(evs):

            beta_idx = [ix for ix, ii in enumerate(
                dm_columns) if ev in ii or "regressor" in ii or "intercept" in ii]

            # get betas for all voxels
            betas = glm_obj.results["betas"][beta_idx]
            event = glm_obj.results["x_conv"][:, beta_idx]
            ev_preds = event @ betas

            df_ev = pd.DataFrame(ev_preds, index=glm_obj.orig.index)
            idx = list(df_ev.index.names)
            df_ev["event_type"] = ev
            df_ev.reset_index(inplace=True)
            idx.insert(-1, "event_type")
            df_ev.set_index(idx, inplace=True)
            preds.append(df_ev)

        # get predictions
        return pd.concat(preds)

    @classmethod
    def format_r2(self, glm_obj):
        idx = {}
        for i in ["subject", "session", "task", "run"]:
            try:
                IDs = utils.get_unique_ids(glm_obj.orig, id=i)
            except BaseException:
                IDs = []

            if len(IDs) > 0:
                idx[i] = IDs[0]

        df_r2 = pd.DataFrame(glm_obj.results["r2"][np.newaxis, ...])
        for key, val in idx.items():
            df_r2[key] = val

        df_r2.set_index(list(idx.keys()), inplace=True)
        return df_r2

    @classmethod
    def format_tstats(self, glm_obj):
        idx = {}
        for i in ["subject", "session", "task", "run"]:
            try:
                IDs = utils.get_unique_ids(glm_obj.orig, id=i)
            except BaseException:
                IDs = []

            if len(IDs) > 0:
                idx[i] = IDs[0]

        evs = utils.get_unique_ids(glm_obj.onsets, id="event_type")
        df_t = pd.DataFrame(glm_obj.results["tstats"])
        for key, val in idx.items():
            df_t[key] = val

        df_t["event_type"] = evs
        ix_list = list(idx.keys())
        ix_list.insert(-1, "event_type")
        df_t.set_index(ix_list, inplace=True)
        return df_t

    @classmethod
    def format_obj(self, glm_obj):
        idx = {}
        for i in ["subject", "session", "task", "run"]:
            try:
                IDs = utils.get_unique_ids(glm_obj.orig, id=i)
            except BaseException:
                IDs = []

            if len(IDs) > 0:
                idx[i] = IDs[0]

        evs = utils.get_unique_ids(glm_obj.onsets, id="event_type")
        df_t = pd.DataFrame(glm_obj.results["tstats"])
        for key, val in idx.items():
            df_t[key] = val

        df_t["event_type"] = evs
        ix_list = list(idx.keys())
        ix_list.insert(-1, "event_type")
        df_t.set_index(ix_list, inplace=True)
        return df_t

    @classmethod
    def format_output(self, glm_obj):

        # full model
        df_fm = self.format_fm(glm_obj)

        # sort out ev-predictions
        df_em = self.format_ev(glm_obj)

        # format r2
        df_r2 = self.format_r2(glm_obj)

        # format r2
        df_tstats = self.format_tstats(glm_obj)

        ddict = {
            "obj": glm_obj,
            "full_model": df_fm,
            "ev_model": df_em,
            "r2": df_r2,
            "tstats": df_tstats
        }

        return ddict

    @classmethod
    def glm_runs(self, df_func, df_onsets, **kwargs):
        """Fit each complete acquisition separately, including session metadata."""
        identities = [key for key in ('subject', 'session', 'task', 'run') if key in df_func.index.names]
        result = []
        for key, data in df_func.groupby(level=identities, sort=False):
            key = key if isinstance(key, tuple) else (key,)
            events = NideconvFitter._select_acquisition(df_onsets, dict(zip(identities, key)))
            if events.empty:
                raise ValueError(f'No events for acquisition {key}')
            result.append(self.single_glm(data, events, **kwargs))
        return result

    @classmethod
    def glm_tasks(self, df_func, df_onsets, **kwargs):
        return self.glm_runs(df_func, df_onsets, **kwargs)

    @classmethod
    def glm_subjects(self, df_func, df_onsets, **kwargs):
        return self.glm_runs(df_func, df_onsets, **kwargs)

    def glm_input(self, **kwargs):

        # Keep the model sampling rate equal to the acquisition sampling rate.
        kwargs.setdefault('TR', self.TR)
        self.df_glm = self.glm_subjects(
            self.func,
            self.onsets,
            **kwargs
        )

        concat_elements = [
            "full_model",
            "ev_model",
            "r2",
            "tstats"
        ]

        self.glm_output = {}
        for conc in concat_elements:
            self.glm_output[conc] = []
            for i in self.df_glm:
                self.glm_output[conc].append(i[conc])

            if len(self.glm_output[conc]) > 0:
                self.glm_output[conc] = pd.concat(self.glm_output[conc])

    def get_result(self):
        return self.glm_output

    def find_max_r2(self, df=None):
        if not isinstance(df, pd.DataFrame):
            df = self.glm_output["r2"]

        max_r2 = np.amax(df.values)
        max_idx = np.where(df.values == max_r2)[-1][0]
        return max_r2, max_idx

    def find_r2(self, vox_nr, df=None):
        df = self.glm_output['r2'] if df is None else df
        values = df.iloc[:, vox_nr].to_numpy()
        return float(np.mean(values)) if len(values) else np.nan

    def plot_ev_predictions(
        self,
        data=None,
        subject=None,
        task=None,
        run=None,
        vox_nr="max",
        cmap="inferno",
        axs=None,
        figsize=(14, 4),
        full=False,
        full_only=False,
        full_color="k",
        r2_dec=4,
        **kwargs
        ):

        # get data
        preds = self.glm_output["ev_model"]
        r2 = self.glm_output["r2"]
        full_preds = self.glm_output["full_model"]
        evs = utils.get_unique_ids(preds, id="event_type")
        if not isinstance(data, pd.DataFrame):
            data = self.func.copy()

        # apply some filters
        expr = []
        if isinstance(subject, str):
            expr.append(f"subject = {subject}")

        if isinstance(task, str):
            expr.append(f"task = {task}")

        if isinstance(run, (int, str)):
            expr.append(f"run = {run}")

        if len(expr) > 0:
            preds = utils.multiselect_from_df(preds, expression=expr)
            r2 = utils.multiselect_from_df(r2, expression=expr)
            data = utils.multiselect_from_df(data, expression=expr)
            full_preds = utils.multiselect_from_df(full_preds, expression=expr)

        if not isinstance(axs, mpl.axes._axes.Axes):
            fig, axs = plt.subplots(figsize=figsize)

        # set defaults for actual datapoints
        markers = ['.']
        colors = ["#cccccc"]
        linewidth = [0.5]
        if not isinstance(vox_nr, int):
            r2_val, vox_nr = self.find_max_r2(df=r2)
        else:
            r2_val = self.find_r2(vox_nr, df=r2)

        signals = [data.iloc[:, vox_nr].values]
        labels = ['input signal']
        if not full_only:

            for ev in evs:

                # get predictions
                ev_preds = utils.select_from_df(
                    preds,
                    expression=f"event_type = {ev}"
                )

                signals.append(ev_preds.iloc[:, vox_nr].values)
                labels.append(f"Event '{ev}'")
                markers.append(None)
                linewidth.append(3)

            if isinstance(cmap, str):
                add_colors = sns.color_palette(cmap, len(evs))
            else:
                if isinstance(cmap, np.ndarray):
                    add_colors = list(cmap.squeeze())
                else:
                    add_colors = cmap

            colors = [*colors, *add_colors]
        else:
            full = True

        # append full model
        if full:
            signals.append(full_preds.iloc[:, vox_nr].values)
            labels.append("full model")
            markers.append(None)
            linewidth.append(2)
            colors.append(full_color)

        # set defaults
        kw_dict = {
            "y_label": "activity (au)",
            "x_label": "volumes",
            "labels": labels,
            "title": f"model fit vox {vox_nr+1}/{data.shape[1]} (r2={round(r2_val,r2_dec)})",
            "line_width": linewidth
        }

        for key, val in kw_dict.items():
            kwargs = utils.update_kwargs(
                kwargs,
                key,
                val
            )

        pl = plotting.LazyLine(
            signals,
            axs=axs,
            markers=markers,
            color=colors,
        )

        return pl
