from . import (
    glm,
    utils,
    plotting,
)
import lmfit
import warnings
import numpy as np
import pandas as pd
import nideconv as nd
import seaborn as sns
import matplotlib as mpl
from typing import Union
from scipy import signal
from sklearn import metrics
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from joblib import Parallel, delayed
from alive_progress import alive_bar

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
        self.concat_obj = self.concat_runs(self.func, self.onsets)

        # store new dataframes
        self.func = self.concat_obj["func"]
        self.onsets = self.concat_obj["onsets"]

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

        """prepare_onsets

        Prepares the onset DataFrame by ensuring proper indexing and adding necessary columns.

        Returns
        -------
        None
            Updates `self.onsets` with formatted data.

        Example
        ----------
        .. code-block:: python

            fitter.prepare_onsets()
        """

        # store copy of original data
        self.orig_onsets = self.onsets.copy()

        # put in dataframe
        self.old_index = []
        if isinstance(self.onsets, np.ndarray):
            self.onsets = pd.DataFrame(self.onsets)
        else:
            try:
                # check old index
                try:
                    self.old_index = list(self.onsets.index.names)
                except BaseException:
                    pass
                self.onsets.reset_index(inplace=True)
            except BaseException:
                pass

        # format dataframe with subject, run, and t
        try:
            self.task_ids = utils.get_unique_ids(self.onsets, id="task")
            self.final_index = [
                "subject",
                "task",
                "run",
                "event_type"
            ]

            self.final_elements = [1, None, 1, "stim"]

        except BaseException:
            self.task_ids = None
            self.final_index = [
                "subject",
                "run",
                "event_type"
            ]
            self.final_elements = [1, 1, "stim"]

        self.onset_index = [
            "subject",
            "run",
            "event_type"
        ]

        for key, val in zip(self.final_index, self.final_elements):
            if key not in list(self.onsets.columns):
                self.onsets[key] = val

        self.drop_cols = []
        if len(self.old_index) > 0:
            for ix in self.old_index:
                if ix not in self.final_index:
                    self.drop_cols.append(ix)

        if len(self.drop_cols) > 0:
            self.onsets = self.onsets.drop(self.drop_cols, axis=1)

        self.onsets.set_index(self.final_index, inplace=True)

    def prepare_data(self):

        """prepare_data

        Prepares the functional data DataFrame by ensuring proper indexing and adding necessary columns.

        Returns
        -------
        None
            Updates `self.func` with formatted data.

        Example
        ----------
        .. code-block:: python

            fitter.prepare_data()
        """

        # store copy of original data
        self.orig_data = self.func.copy()

        # put in dataframe
        self.time = self._get_timepoints()
        self.old_index = []
        if isinstance(self.func, np.ndarray):
            self.func = pd.DataFrame(self.func)
        else:
            try:
                # check old index
                try:
                    self.old_index = list(self.func.index.names)
                except BaseException:
                    pass
                self.func = self.func.reset_index()
            except BaseException:
                pass

        # format dataframe with subject, run, and t
        try:
            self.task_ids = utils.get_unique_ids(self.func, id="task")
            self.final_index = [
                "subject",
                "task",
                "run",
                "t"
            ]

            self.final_elements = [1, None, 1, self.time]

        except BaseException:
            self.task_ids = None
            self.final_index = [
                "subject",
                "run",
                "t"
            ]
            self.final_elements = [1, 1, self.time]

        for key, val in zip(self.final_index, self.final_elements):
            if key not in list(self.func.columns):
                self.func[key] = val

        self.drop_cols = []
        if len(self.old_index) > 0:
            for ix in self.old_index:
                if ix not in self.final_index:
                    self.drop_cols.append(ix)

        if len(self.drop_cols) > 0:
            self.func = self.func.drop(self.drop_cols, axis=1)

        self.func.set_index(self.final_index, inplace=True)

    @classmethod
    def concat_func(self, df):

        """concat_func

        Concatenates functional MRI data from multiple runs into a single dataframe. Adjusts time indices based on 
        TR and resets the run index.

        Parameters
        ----------
        df : pd.DataFrame
            Functional data dataframe indexed on `subject`, `run`, and `t`.

        Returns
        -------
        pd.DataFrame
            Concatenated functional data with updated indices.

        Example
        ----------
        .. code-block:: python

            func_concat = NideconvFitter.concat_func(func_df)
        """

        # check if time matches TR
        t = utils.get_unique_ids(df, id="t")
        tr = np.diff(t)[0]
        tr_list = list(np.arange(0, df.shape[0]) * tr)

        # construct new dataframe
        new_func = pd.DataFrame(df.values, columns=list(df.columns))
        new_func["subject"] = utils.get_unique_ids(df, id="subject")[0]
        new_func["t"], new_func["run"] = np.array(tr_list), 1
        new_func.set_index(["subject", "run", "t"], inplace=True)

        return new_func

    @classmethod
    def concat_onsets(
        self,
        onsets,
        func
        ):

        """concat_onsets

        Concatenates event onset data across multiple runs, updating onset times to reflect the new concatenated 
        timeline.

        Parameters
        ----------
        onsets : pd.DataFrame
            Onset data indexed on `subject`, `run`, and `event_type`.
        func : pd.DataFrame
            Functional data used to determine run durations.

        Returns
        -------
        pd.DataFrame
            Concatenated onsets with updated `onset` times.

        Example
        ----------
        .. code-block:: python

            onsets_concat = NideconvFitter.concat_onsets(onsets_df, func_df)
        """

        n_runs = utils.get_unique_ids(onsets, id="run")
        run_dfs = []
        for ix, ii in enumerate(n_runs):
            df = utils.select_from_df(
                onsets, expression=f"run = {ii}").reset_index()
            ff = utils.select_from_df(func, expression=f"run = {ii}")
            df["onset"] += (ix * ff.shape[0])
            df["run"] = 1
            run_dfs.append(df)

        new_onsets = pd.concat(run_dfs)
        new_onsets.set_index(["subject", "run", "event_type"], inplace=True)

        return new_onsets

    @classmethod
    def concat_runs(
        self,
        func,
        onsets
        ):

        """concat_runs

        Concatenates functional MRI data and event onsets across multiple runs for each subject. Calls 
        `concat_func` and `concat_onsets` internally.

        Parameters
        ----------
        func : pd.DataFrame
            Functional MRI data indexed by `subject`, `run`, and `t`.
        onsets : pd.DataFrame
            Event onset data indexed by `subject`, `run`, and `event_type`.

        Returns
        -------
        dict
            Dictionary containing concatenated functional (`"func"`) and onset (`"onsets"`) data.

        Example
        ----------
        .. code-block:: python

            concatenated_data = NideconvFitter.concat_runs(func_df, onsets_df)
            func_concat = concatenated_data["func"]
            onsets_concat = concatenated_data["onsets"]
        """

        sub_ids = utils.get_unique_ids(func, id="subject")
        sub_dfs = {}
        sub_dfs["func"] = []
        sub_dfs["onsets"] = []

        # loop through subjects
        for sub in sub_ids:

            # concatenate functional
            sub_func = utils.select_from_df(
                func, expression=f"subject = {sub}")
            sub_dfs["func"].append(self.concat_func(sub_func))

            # concatenate onsets
            sub_onsets = utils.select_from_df(
                onsets, expression=f"subject = {sub}")
            sub_dfs["onsets"].append(self.concat_onsets(sub_onsets, sub_func))

        for key, val in sub_dfs.items():
            sub_dfs[key] = pd.concat(val)

        return sub_dfs

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
            order = [tm, trg]

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

    def parameters_for_tc_subjects(
        self,
        **kwargs
        ):

        """parameters_for_tc_subjects

        Extracts HRF parameters for subject-specific timecourses.

        Parameters
        ----------
        *args : tuple, optional
            Additional arguments for :class:`lazyfmri.fitting.HRFMetrics`.
        **kwargs : dict, optional
            Additional keyword arguments for :class:`lazyfmri.fitting.HRFMetrics`.

        Returns
        -------
        None
            Updates `self.pars_subjects` with extracted parameters.

        Raises
        ------
        ValueError
            If `self.tc_subjects` is not available.

        Example
        ----------
        .. code-block:: python

            fitter.parameters_for_tc_subjects()
        """

        if not hasattr(self, "tc_subjects"):
            raise ValueError(
                f"{self} does not have 'tc_subjects' attribute, run fitter first")

        utils.verbose(
            f"Deriving parameters from {self} with 'HRFMetrics'",
            self.verbose
        )
        subj_ids = utils.get_unique_ids(self.tc_subjects, id="subject")

        subjs = []
        subjs_avg_run = []
        for sub in subj_ids:

            sub_df = utils.select_from_df(
                self.tc_subjects, expression=f"subject = {sub}")
            ev_ids = utils.get_unique_ids(sub_df, id="event_type")

            # loop through evs
            evs = []
            evs_avg = []
            for ev in ev_ids:

                # get event-specific dataframe
                ev_df = utils.select_from_df(
                    sub_df, expression=f"event_type = {ev}")
                run_ids = utils.get_unique_ids(sub_df, id="run")

                # loop through runs
                runs = []
                for run in run_ids:

                    # get run-specific dataframe
                    run_df = utils.select_from_df(
                        ev_df, expression=f"run = {run}")
                    pars = HRFMetrics(
                        run_df,
                        TR=self.TR,
                        **kwargs
                    ).return_metrics()

                    pars["event_type"], pars["run"] = ev, run
                    runs.append(pars)

                # also get parameters of average across runs
                idx = ["subject", "event_type"]
                if "time" in list(ev_df.index.names):
                    idx += ["time"]
                else:
                    idx += ["t"]

                avg_df = ev_df.groupby(idx).mean()
                avg_pars = HRFMetrics(
                    avg_df,
                    TR=self.TR,
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
            evs_avg["subject"] = sub
            subjs_avg_run.append(evs_avg)

            # single runs
            evs = pd.concat(evs)
            evs["subject"] = sub
            subjs.append(evs)

        df_conc = pd.concat(subjs)
        df_avg = pd.concat(subjs_avg_run)
        avg_idx = ["subject", "event_type"]
        idx = avg_idx + ["run"]

        add_idx = self.find_pars_index(df_conc)
        if isinstance(add_idx, str):
            idx += [add_idx]
            avg_idx += [add_idx]

        self.pars_subjects = df_conc.set_index(idx)
        self.avg_pars_subjects = df_avg.set_index(avg_idx)

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

    def parameters_for_tc_condition(
        self,
        *args,
        **kwargs
        ):

        """parameters_for_tc_condition

        Extracts HRF parameters for condition-averaged timecourses.

        Parameters
        ----------
        *args : tuple, optional
            Additional arguments for :class:`lazyfmri.fitting.HRFMetrics`.
        **kwargs : dict, optional
            Additional keyword arguments for :class:`lazyfmri.fitting.HRFMetrics`.

        Returns
        -------
        None
            Updates `self.pars_condition` with extracted parameters.

        Raises
        ------
        ValueError
            If `self.tc_condition` is not available.

        Example
        ----------
        .. code-block:: python

            fitter.parameters_for_tc_condition()
        """

        if not hasattr(self, "tc_condition"):
            raise ValueError(
                f"{self} does not have 'tc_condition' attribute, run fitter first")

        utils.verbose(
            f"Deriving condition-wise parameters from {self} with 'HRFMetrics'",
            self.verbose)
        ev_ids = utils.get_unique_ids(self.tc_condition, id="event_type")

        # loop through evs
        evs = []
        for ev in ev_ids:

            # get event-specific dataframe
            ev_df = utils.select_from_df(
                self.tc_condition, expression=f"event_type = {ev}")

            ev_pars = HRFMetrics(
                ev_df,
                TR=self.TR,
                *args,
                **kwargs
            ).return_metrics()

            ev_pars["event_type"] = ev
            evs.append(ev_pars)

        # concatenate
        df_conc = pd.concat(evs)
        df_conc["subject"] = "avg"

        # add indices
        idx = ["subject", "event_type"]
        add_idx = self.find_pars_index(df_conc)
        if isinstance(add_idx, str):
            idx += [add_idx]

        self.pars_condition = df_conc.set_index(idx)
    
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

    def fit(
        self,
        debug=False,
        **kwargs
        ):

        """fit

        Fits HRF parameters across subjects, runs, and event types.

        Parameters
        ----------
        debug : bool, optional
            If `True`, prints verbose messages during the fitting process, by default `False`.
        **kwargs : dict
            Additional keyword arguments for HRF fitting.

        Returns
        -------
        None
            Stores fitted HRF parameters and model outputs as object attributes:
            - `tc_subjects`: Extracted HRF profiles for each subject.
            - `ev_predictions`: Event-specific model predictions.
            - `estimates`: Estimated HRF parameters.
            - `tc_condition`: Mean HRF timecourses across conditions.
            - `sem_condition`: Standard error of the mean (SEM) of timecourses.
            - `std_condition`: Standard deviation of timecourses.

        Example
        ----------
        .. code-block:: python

            nd_fit.fit(debug=True)
            print(nd_fit.tc_subjects)
        """

        self.ddict_ev = self._set_dict()

        # loop through subject
        for sub in self.sub_ids:

            utils.verbose(f"Fitting '{sub}'", debug)

            # get subject specific onsets
            self.sub_onsets = utils.select_from_df(
                self.orig_onsets, expression=f"subject = {sub}")
            for run in self.run_ids:

                # subject and run specific data
                utils.verbose(f" run-'{run}'", debug)
                expr = (
                    f"subject = {sub}",
                    "&",
                    f"run = {run}"
                )
                self.run_func = utils.select_from_df(
                    self.orig_func, expression=expr)

                # loop through events
                for ev in self.ev_ids:

                    utils.verbose(f"  ev-'{ev}'", debug)

                    # run/event-specific onsets
                    expr = (
                        f"run = {run}",
                        "&",
                        f"event_type = {ev}"
                    )
                    self.ons = utils.select_from_df(
                        self.sub_onsets, expression=expr)
                    # print(f"  {self.run_func.shape}")

                    # do fit
                    self.rf = self.single_response_fitter(
                        self.run_func.T.values,
                        self.ons,
                        TR=self.TR,
                        **kwargs
                    )

                    # get profiles and full timecourse predictions
                    df = self.rf.hrf_profiles.copy()
                    pr = self.rf.predictions.copy()

                    es = self.rf.estimates.copy()
                    es["vox"] = list(self.func.columns)
                    es.set_index(["subject", "event_type",
                                 "run", "vox"], inplace=True)

                    # append
                    self.ddict_ev["predictions"].append(pr)
                    self.ddict_ev["profiles"].append(df)
                    self.ddict_ev["pars"].append(es)

        # concatenate all subjects
        self.ddict_sub = self._concat_dict(self.ddict_ev)

        # set final outputs
        self.tc_subjects = self.ddict_sub["profiles"]
        self.ev_predictions = self.ddict_sub["predictions"]
        self.estimates = self.ddict_sub["pars"]

        self.tc_subjects.columns = self.func.columns
        self.ev_predictions.columns = self.func.columns

        self.grouper = self.tc_subjects.groupby(
            level=['event_type', 'covariate', 'time'])
        self.tc_condition = self.grouper.mean()
        self.sem_condition = self.grouper.sem()
        self.std_condition = self.grouper.std()

        self.time = utils.get_unique_ids(self.tc_condition, id="time")


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
        Whether to fit an intercept. Default is `False`.
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
        Includes an intercept in confound model if `True`. Default is `True`.
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
        **kwargs):

        self.func = func
        self.onsets = onsets
        self.confounds = confounds
        self.basis_sets = basis_sets
        self.fit_type = fit_type
        self.n_regressors = n_regressors
        self.add_intercept = add_intercept
        self.verbose = verbose
        self.lump_events = lump_events
        self.TR = TR
        self.fs = 1 / self.TR
        self.interval = interval
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

        """melt_events

        Merges all events in the onset dataframe into a single generic event type `"stim"`.

        Returns
        -------
        pd.DataFrame
            Onset dataframe with all events combined into a single category.

        Example
        ----------
        .. code-block:: python

            lumped_onsets = fitter.melt_events()
        """


        self.lumped_onsets = self.onsets.copy().reset_index()
        self.lumped_onsets['event_type'] = 'stim'
        self.lumped_onsets = self.lumped_onsets.set_index(
            ['subject', 'run', 'event_type'])
        return self.lumped_onsets

    def derive_n_regressors(self):

        """derive_n_regressors

        Determines the number of regressors to use based on the specified basis set. If `"tr"` is specified, 
        automatically calculates regressors per TR.

        Returns
        -------
        None
            Stores the determined number of regressors in `self.n_regressors`.

        Example
        ----------
        .. code-block:: python

            fitter.derive_n_regressors()
            print(fitter.n_regressors)
        """


        self.allowed_basis_sets = [
            "fir",
            "fourier",
            "dct",
            "legendre",
            "canonical_hrf",
            "canonical_hrf_with_time_derivative",
            "canonical_hrf_with_time_derivative_dispersion"
        ]

        self.n_regressors
        if self.basis_sets not in self.allowed_basis_sets:
            raise ValueError(
                f"Unrecognized basis set '{self.basis_sets}'. Must be one of {self.allowed_basis_sets}")
        else:
            if self.basis_sets == "canonical_hrf":
                self.n_regressors = 1
            elif self.basis_sets == "canonical_hrf_with_time_derivative":
                self.n_regressors = 2
            elif self.basis_sets == "canonical_hrf_with_time_derivative_dispersion":
                self.n_regressors = 3
            else:
                # set 1 regressor per TR
                if isinstance(self.n_regressors, str):

                    self.tmp_regressors = round(
                        ((self.interval[1] + abs(self.interval[0]))) / self.TR)

                    if len(self.n_regressors) > 2:
                        # assume operation on TR
                        els = self.n_regressors.split(" ")

                        if len(els) != 3:
                            raise TypeError(
                                f"""Format of this input must be 'tr <operation> <value>' (WITH SPACES), not
                                '{self.n_regressors}'. Example: 'tr x 2' will use 2 regressors per TR""")

                        op = utils.str2operator(els[1])
                        val = float(els[-1])

                        self.n_regressors = int(op(self.tmp_regressors, val))
                    else:
                        self.n_regressors = self.tmp_regressors

    @staticmethod
    def get_event_predictions_from_fitter(fitter, intercept=True):

        """get_event_predictions_from_fitter

        Extracts event-specific predictions from a fitted model.

        Parameters
        ----------
        fitter : nideconv.GroupResponseFitter
            The fitted response model.
        intercept : bool, optional
            If `True`, includes intercept regressors in the predictions, by default `True`.

        Returns
        -------
        pd.DataFrame
            Dataframe containing predicted response for each event type.

        Example
        ----------
        .. code-block:: python

            event_predictions = NideconvFitter.get_event_predictions_from_fitter(fitter)
        """


        ev_pred = []
        for ev in fitter.events.keys():

            if intercept:
                X_stim = pd.concat(
                    [
                        fitter.X.xs("confounds", axis=1, drop_level=False),
                        fitter.X.xs(ev, axis=1, drop_level=False)
                    ],
                    axis=1
                )

                expr = (
                    f"event type = {ev}",
                    "or",
                    "event type = confounds"
                )
            else:
                X_stim = fitter.X.xs(ev, axis=1, drop_level=False)
                expr = f"event type = {ev}"

            betas = utils.select_from_df(fitter.betas, expression=expr)
            pred = X_stim.dot(betas).reset_index()
            pred.rename(columns={"time": "t"}, inplace=True)

            if "index" in list(pred.columns):
                pred.rename(columns={"index": "t"}, inplace=True)

            pred["event_type"] = ev
            ev_pred.append(pred)

        ev_pred = pd.concat(ev_pred, ignore_index=True)
        return ev_pred

    def _interpolate_timecourse(self, tc):
        """
        Interpolates a single FIR timecourse by identifying plateaus and averaging transition points.

        Parameters
        ----------
        tc : np.ndarray
            1D timecourse array.

        Returns
        -------
        np.ndarray
            Interpolated timecourse.
        np.ndarray
            Corresponding time values from self.time.
        """

        tc = np.asarray(tc)
        diff = np.diff(tc)
        gradient = np.sign(diff)
        idx = np.where(gradient != 0)[0]

        if len(idx) == 0:
            # If no change, return a flat timecourse
            return tc[:1], self.time[:1]

        new_idx = ((idx[1:] + idx[:-1]) / 2).astype(int)
        new_idx = np.insert(new_idx, 0, idx[0] // 2)
        new_idx = np.insert(new_idx, 0, 0)  # Ensure first point is included

        interp_vals = tc[new_idx]
        new_time = self.time[new_idx]

        return interp_vals, new_time


    def interpolate_fir_subjects(self):

        """interpolate_fir_subjects

        Interpolates the finite impulse response (FIR) model across subjects, resampling plateaus
        to obtain smoother timecourses.

        Returns
        -------
        pd.DataFrame
            Interpolated FIR timecourses indexed by `subject`, `run`, `event_type`, `covariate`, and `time`.

        Example
        ----------
        .. code-block:: python

            interpolated_fir = fitter.interpolate_fir_subjects()
        """

        # loop through subject IDs
        subjs = utils.get_unique_ids(self.tc_subjects, id="subject")
        sub_df = []
        for sub in subjs:

            # get subject specific dataframes
            sub_tmp = utils.select_from_df(
                self.tc_subjects, expression=f"subject = {sub}")
            run_ids = utils.get_unique_ids(sub_tmp, id="run")

            # loop through runs
            run_df = []
            for run in run_ids:

                # get run-specific dataframe
                run_tmp = utils.select_from_df(
                    sub_tmp, expression=f"run = {run}")

                # call the interpolate function
                run_interp = self.interpolate_fir_condition(run_tmp)

                # make indexing the same as input
                run_interp.columns = run_tmp.columns
                run_interp = run_interp.reset_index()
                run_interp["run"] = run

                # append interpolated run dataframe
                run_df.append(run_interp)

            # concatenate run dataframe and add subject inedx
            run_df = pd.concat(run_df)
            run_df["subject"] = sub
            sub_df.append(run_df)

        # final indexing
        sub_df = pd.concat(sub_df)
        sub_df.set_index(["subject", "run", "event_type",
                         "covariate", "time"], inplace=True)
        return sub_df

    def interpolate_fir_condition(self, obj):
        """
        Interpolates FIR model responses for all conditions in the input DataFrame.

        Parameters
        ----------
        obj : pd.DataFrame
            DataFrame with FIR model responses, indexed on ['event_type', 'covariate', 'time'].

        Returns
        -------
        pd.DataFrame
            Interpolated timecourses per condition and event type.
        """

        evs = utils.get_unique_ids(obj, id="event_type")
        interpolated_dfs = []

        for ev in evs:
            ev_df = utils.select_from_df(obj, expression=f"event_type = {ev}")
            covariate = utils.get_unique_ids(ev_df, id="covariate")[0]

            # Interpolate each timecourse column (e.g., voxel or component)
            interpolated_cols = []
            for col_idx in range(ev_df.shape[1]):
                tc = ev_df.iloc[:, col_idx].values
                interp_tc, new_t = self._interpolate_timecourse(tc)
                interpolated_cols.append(interp_tc[..., np.newaxis])

            # Stack interpolated columns
            data = np.concatenate(interpolated_cols, axis=1)
            df_interp = pd.DataFrame(data)
            df_interp["time"] = new_t
            df_interp["event_type"] = ev
            df_interp["covariate"] = covariate

            interpolated_dfs.append(df_interp)

        df_result = pd.concat(interpolated_dfs)
        df_result.set_index(["event_type", "covariate", "time"], inplace=True)

        return df_result

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
        self.fitters = self.model._get_response_fitters()
        if not isinstance(self.fitters, pd.DataFrame):
            self.fitters = pd.DataFrame(self.fitters)

        self.fitters = self.check_for_run_index(self.fitters)

    @staticmethod
    def get_curves_from_fitter(
        fitter,
        index=True,
        icpt=False,
        ):

        """get_curves_from_fitter

        Extracts the predicted response curves for each event type from the fitted model.

        Parameters
        ----------
        fitter : nideconv.GroupResponseFitter
            The fitted response model.
        index : bool, optional
            If `True`, maintains dataframe index structure, by default `True`.
        icpt : bool, optional
            If `True`, includes the intercept regressor in the predictions, by default `False`.

        Returns
        -------
        pd.DataFrame
            Dataframe containing event-specific response curves.

        Example
        ----------
        .. code-block:: python

            response_curves = NideconvFitter.get_curves_from_fitter(fitter)
        """


        betas = fitter.betas
        basis = fitter.get_basis_functions()
        regr = utils.get_unique_ids(betas, id="regressor")

        ev_df = []
        evs = utils.get_unique_ids(betas, id="event type")

        # full intercept was fitted
        if icpt:
            regr.pop(regr.index("intercept"))
            evs.pop(evs.index("confounds"))

        for ev in evs:
            # print(f"{ev}")
            expr = f"event type = {ev}"
            ev_beta = utils.select_from_df(betas, expression=expr)
            ev_basis = utils.select_from_df(basis, expression=expr)

            basis_curves = []
            for reg in regr:

                # if intercept was fitted, we need to extract the beta/basis
                # set for that too
                if icpt:

                    # basis sets
                    reg_basis = ev_basis.loc[:, ["confounds", ev]]
                    b1 = pd.DataFrame(
                        reg_basis.loc[:, ev].loc[:, "intercept"].loc[:, reg])
                    b2 = pd.DataFrame(
                        reg_basis.loc[:, "confounds"].loc[:, "intercept"])
                    reg1 = pd.concat([b2, b1], axis=1)

                    # betas
                    b1 = utils.select_from_df(
                        betas, expression="event type = confounds")
                    b2 = utils.select_from_df(
                        ev_beta, expression="regressor = {reg}")
                    bet1 = pd.concat([b1, b2])

                else:

                    reg1 = pd.DataFrame(
                        ev_basis.loc[:, ev].loc[:, "intercept"].loc[:, reg])
                    bet1 = utils.select_from_df(
                        ev_beta, expression=f"regressor = {reg}")

                # dot product
                tmp = reg1.values.dot(bet1.values)

                tmp = pd.DataFrame(
                    tmp, columns=bet1.columns, index=ev_basis.index)
                tmp.reset_index(inplace=True)
                tmp.rename(
                    columns={
                        "time": "t",
                        "event type": "event_type"
                    },
                    inplace=True
                )
                tmp["covariate"] = reg
                tmp.set_index(["event_type", "covariate", "t"], inplace=True)
                basis_curves.append(tmp)

            basis_curves = pd.concat(basis_curves)
            ev_df.append(basis_curves)

        ev_df = pd.concat(ev_df)
        if not index:
            ev_df.reset_index(inplace=True)

        return ev_df

    def get_basisset_timecourses(self):

        """get_basisset_timecourses

        Extracts the fitted basis set regressors from the model, retrieving timecourses for each event.

        Returns
        -------
        None
            Stores timecourses in object attributes:
            - `sub_basis`: Basis function timecourses indexed by subject, event type, and time.

        Example
        ----------
        .. code-block:: python

            fitter.get_basisset_timecourses()
            print(fitter.sub_basis)
        """


        # also get the predictions
        if self.fit_type == "ols":
            if not hasattr(self, "fitters"):
                self.fitters = self.model._get_response_fitters()
                self.format_fitters()

        sub_ids = utils.get_unique_ids(self.fitters, id="subject")
        sub_basis = []
        for sub in sub_ids:

            sub_fitters = utils.select_from_df(
                self.fitters, expression=f"subject = {sub}")

            # loop through runs
            self.sub_basis = []
            run_ids = utils.get_unique_ids(sub_fitters, id="run")
            for run in run_ids:

                # full model predictions
                run_fitter = utils.select_from_df(
                    self.fitters, expression=f"run = {run}").iloc[0][0]
                curves = self.get_curves_from_fitter(
                    run_fitter,
                    index=False,
                    icpt=self.conf_icpt
                )

                # append
                curves["run"] = run
                self.sub_basis.append(curves)

            # concatenate and index
            self.sub_basis = pd.concat(self.sub_basis)
            self.sub_basis["subject"] = sub

            sub_basis.append(self.sub_basis)

        # event-specific events
        self.sub_basis = pd.concat(sub_basis, ignore_index=True)
        self.sub_basis.set_index(
            ["subject", "event_type", "run", "covariate", "t"], inplace=True)

    def get_predictions_per_event(self):
        """
        Retrieves both full-model and event-specific predictions from the fitted model.

        Stores predictions in object attributes:
            - self.ev_predictions: Event-level predicted responses.
            - self.sub_pred_full: Full model predictions per subject.

        Returns
        -------
        None

        Example
        -------
        >>> fitter.get_predictions_per_event()
        >>> print(fitter.ev_predictions)
        """

        if self.fit_type == "ols":
            if not hasattr(self, "fitters"):
                self.fitters = self.model._get_response_fitters()
                self.format_fitters()

        sub_ids = utils.get_unique_ids(self.fitters, id="subject")

        all_ev_preds = []
        all_full_preds = []

        for sub in sub_ids:
            sub_fitters = utils.select_from_df(self.fitters, f"subject = {sub}")

            subject_ev_preds = []
            subject_full_preds = []

            run_ids = utils.get_unique_ids(sub_fitters, id="run")
            for run in run_ids:
                run_fitter = utils.select_from_df(
                    sub_fitters, f"run = {run}"
                ).iloc[0, 0]  # assumes fitter is in first cell

                # Full model prediction
                preds = run_fitter.predict_from_design_matrix()
                preds.columns = self.tc_condition.columns  # ensure consistent columns
                preds = preds.reset_index().rename(columns={"index": "time"})
                preds["run"] = run
                subject_full_preds.append(preds)

                # Event-specific predictions
                ev_pred = self.get_event_predictions_from_fitter(
                    run_fitter, intercept=self.conf_icpt
                )
                ev_pred["run"] = run
                subject_ev_preds.append(ev_pred)

            # Concatenate subject-level results
            subj_full_df = pd.concat(subject_full_preds, ignore_index=True)
            subj_ev_df = pd.concat(subject_ev_preds, ignore_index=True)
            subj_full_df["subject"] = subj_ev_df["subject"] = sub

            all_full_preds.append(subj_full_df)
            all_ev_preds.append(subj_ev_df)

        # Assign outputs to object attributes
        self.ev_predictions = pd.concat(all_ev_preds, ignore_index=True)
        self.ev_predictions.set_index(["subject", "event_type", "run", "t"], inplace=True)

        self.sub_pred_full = pd.concat(all_full_preds, ignore_index=True)
        self.sub_pred_full.set_index(["subject", "run", "time"], inplace=True)


    def timecourses_condition(self):
        """
        Computes condition-wise timecourses by averaging across runs and extracting
        subject-level and condition-level HRF responses from the model.

        Stores timecourse data in object attributes:
            - self.tc_condition: Condition-wise average across subjects.
            - self.tc_subjects: Subject-level timecourses.
            - self.tc_mean: Mean timecourse (subject-level).
            - self.tc_sem: Standard error of the mean.
            - self.tc_std: Standard deviation.
            - self.sem_condition: SEM at condition level.
            - self.std_condition: STD at condition level.
            - self.time: Extracted time axis.
            - self.rsq_: R² values from model (if available).
            - self.ev_predictions: Event-level predictions (if OLS fit).
        """

        # Default covariate
        if not isinstance(self.covariates, str):
            self.covariates = "intercept"

        utils.verbose(
            f"Fetching subject/condition-wise time courses from {self.model}",
            self.verbose
        )

        # --- Fetch timecourses ---
        if self.fit_type == "ols":
            self.tc_condition = self.model.get_conditionwise_timecourses()
            self.tc_subjects = self.model.get_timecourses()
        else:
            self.tc_condition = self.tc_subjects.groupby(level=["event type", "covariate", "time"]).mean()
            self.fitters = self.sub_df.copy()

        # --- Subject-level grouping ---
        self.obj_grouped = self.tc_subjects.groupby(
            level=["subject", "event type", "covariate", "time"])

        self.tc_mean = self.obj_grouped.mean()
        self.tc_sem = self.obj_grouped.sem()
        self.tc_std = self.obj_grouped.std()

        # --- Clean index names ---
        self.tc_mean = self.change_name_set_index(self.tc_mean)
        self.tc_sem = self.change_name_set_index(self.tc_sem)
        self.tc_std = self.change_name_set_index(self.tc_std)

        # --- Aggregate condition-level stats ---
        self.grouper = ["event_type", "covariate", "time"]
        self.sem_condition = self.tc_sem.groupby(level=self.grouper).mean()
        self.std_condition = self.tc_std.groupby(level=self.grouper).mean()

        # --- Finalize condition & subject-level dataframes ---
        self.tc_condition = self.change_name_set_index(
            self.tc_condition, index=self.grouper
        )

        self.tc_subjects = self.check_for_run_index(self.tc_subjects)
        self.tc_subjects = self.change_name_set_index(
            self.tc_subjects,
            index=["subject", "run", "event_type", "covariate", "time"]
        )

        # Drop unnecessary task column
        if "task" in self.tc_subjects.columns:
            self.tc_subjects.drop(columns=["task"], inplace=True)

        # --- Get time axis ---
        self.time = (
            self.tc_condition.groupby("time").mean().reset_index()["time"].values
        )

        # --- FIR-specific interpolation ---
        if self.basis_sets == "fir":
            self.tc_condition_interp = self.interpolate_fir_condition(self.tc_condition)
            self.tc_subjects_interp = self.interpolate_fir_subjects()

        # --- R² and predictions ---
        try:
            self.rsq_ = self.model.get_rsq()
        except Exception:
            self.rsq_ = None

        if self.fit_type == "ols":
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

        utils.verbose(
            f"Selected '{self.basis_sets}'-basis sets (with {self.n_regressors} regressors)",
            self.verbose)

        # define events
        self.cond = utils.get_unique_ids(self.used_onsets, id="event_type")
        self.run_ids = utils.get_unique_ids(self.used_onsets, id="run")

        # add events to model
        if self.fit_type.lower() == "ols":
            for ix, event in enumerate(self.cond):
                utils.verbose(f"Adding event '{event}' to model", self.verbose)

                if isinstance(self.covariates, list):
                    cov = self.covariates[ix]
                else:
                    cov = self.covariates

                if isinstance(self.add_intercept, list):
                    icpt = self.add_intercept[ix]
                else:
                    icpt = self.add_intercept

                self.model.add_event(
                    str(event),
                    basis_set=self.basis_sets,
                    n_regressors=self.n_regressors,
                    interval=self.interval,
                    add_intercept=icpt,
                    covariates=cov)

        else:

            utils.verbose("Setting up models ridge-regression", self.verbose)

            # get subjects
            try:
                self.sub_ids = utils.get_unique_ids(self.func, id="subject")
            except BaseException:
                self.sub_ids = [1]

            self.do_fit = True
            self.sub_df = []
            self.sub_ev_df = []
            self.tc_subjects = []
            for sub in self.sub_ids:

                try:
                    self.run_ids = [
                        int(i) for i in utils.get_unique_ids(self.func, id="run")]
                    self.run_df = []
                    self.run_pred_ev_df = []
                    self.run_prof_ev_df = []
                    set_indices = ["subject", "run"]
                    set_ev_idc = ["event_type", "run", "t"]
                    set_prof_idc = ["subject", "run", "event type", "covariate", "time"]
                except BaseException:
                    self.run_ids = [None]
                    self.run_df = None
                    self.run_pred_ev_df = None
                    self.run_prof_ev_df = None
                    set_indices = ["subject"]
                    set_ev_idc = ["event_type", "t"]
                    set_prof_idc = [
                        "subject", "event type", "covariate", "time"]

                # loop through runs, if available
                for run in self.run_ids:

                    # loop trough voxels (ridge only works for 1d data)
                    self.vox_df = []
                    self.ev_predictions = []
                    self.vox_prof = []
                    for ix, col in enumerate(list(self.func.columns)):

                        # select single voxel timecourse from main DataFrame
                        if isinstance(run, int):
                            self.vox_signal = pd.DataFrame(utils.select_from_df(
                                self.func[col], expression=f"run = {run}")[col])
                        else:
                            self.vox_signal = pd.DataFrame(self.func[col])

                        # make response fitter
                        if isinstance(self.covariates, list):
                            cov = self.covariates[ix]
                        else:
                            cov = self.covariates

                        self.rf = self.make_response_fitter(
                            self.vox_signal, run=run, cov=cov)

                        # fit immediately; makes life easier
                        if self.do_fit:
                            self.rf.fit(type="ridge")

                            # HRF profiles
                            self.vox_prof.append(
                                self.rf.get_timecourses().reset_index())

                            # timecourse predictions
                            self.ev_pred = self.get_event_predictions_from_fitter(
                                self.rf, intercept=self.conf_icpt)
                            self.ev_predictions.append(self.ev_pred)

                        self.rf_df = pd.DataFrame({col: self.rf}, index=[0])
                        self.vox_df.append(self.rf_df)

                    self.vox_df = pd.concat(self.vox_df, axis=1)

                    if len(self.ev_predictions) > 0:
                        self.ev_predictions = pd.concat(
                            self.ev_predictions, axis=1)

                    if len(self.vox_prof) > 0:
                        self.vox_prof = pd.concat(self.vox_prof, axis=1)

                    if isinstance(run, int):
                        self.vox_df["run"] = run
                        self.run_df.append(self.vox_df)

                        if isinstance(self.ev_predictions, pd.DataFrame):
                            self.ev_predictions["run"] = run
                            self.run_pred_ev_df.append(self.ev_predictions)

                        if isinstance(self.vox_prof, pd.DataFrame):
                            self.vox_prof["run"] = run
                            self.run_prof_ev_df.append(self.vox_prof)

                    else:
                        self.vox_df["subject"] = sub
                        self.sub_df.append(self.vox_df)

                        if isinstance(self.ev_predictions, pd.DataFrame):
                            self.sub_ev_df.append(self.ev_predictions)

                        if isinstance(self.vox_prof, pd.DataFrame):
                            self.vox_prof["subject"] = sub
                            self.tc_subjects.append(self.vox_prof)

                if isinstance(self.run_df, list):
                    self.run_df = pd.concat(self.run_df)
                    self.run_df["subject"] = sub
                    self.sub_df.append(self.run_df)

                if isinstance(self.run_pred_ev_df, list):
                    self.run_pred_ev_df = pd.concat(self.run_pred_ev_df)
                    self.sub_ev_df.append(self.run_pred_ev_df)

                if isinstance(self.run_prof_ev_df, list):
                    self.run_prof_ev_df = pd.concat(self.run_prof_ev_df)
                    self.run_prof_ev_df["subject"] = sub
                    self.tc_subjects.append(self.run_prof_ev_df)

            self.sub_df = pd.concat(self.sub_df).set_index(set_indices)

            if len(self.sub_ev_df) > 0:
                self.sub_ev_df = pd.concat(self.sub_ev_df)
                self.sub_ev_df.set_index(set_ev_idc, inplace=True)

            if len(self.tc_subjects) > 0:
                self.tc_subjects = pd.concat(self.tc_subjects)
                self.tc_subjects.set_index(set_prof_idc, inplace=True)

    def make_response_fitter(self, data, run=None, cov=None):

        # specify voxel-specific model
        model = nd.ResponseFitter(
            input_signal=data,
            sample_rate=self.fs,
            add_intercept=self.conf_icpt,
            oversample_design_matrix=self.osf
        )

        # get onsets
        for i in self.cond:
            model.add_event(
                str(i),
                onsets=self.make_onsets_for_response_fitter(i, run=run),
                basis_set=self.basis_sets,
                n_regressors=self.n_regressors,
                interval=self.interval,
                covariates=cov
            )

        return model

    def make_onsets_for_response_fitter(self, i, run=None):
        select_from_onsets = self.used_onsets.copy()
        if isinstance(run, int):
            select_from_onsets = utils.select_from_df(
                select_from_onsets, expression=f"run = {run}")
            drop_idcs = ["subject", "run"]
        else:
            drop_idcs = ["subject"]

        return select_from_onsets.reset_index().drop(
            drop_idcs, axis=1).set_index('event_type').loc[i].onset

    def fit(self):

        """fit

        Fits a deconvolution model to the functional MRI data based on subject, run, and event information. Uses 
        `single_response_fitter` to fit the data and store results, including timecourse predictions, profiles, 
        and parameter estimates.

        Parameters
        ----------
        debug : bool, optional
            If `True`, prints verbose messages during fitting, by default `False`.
        **kwargs : dict
            Additional parameters passed to the `single_response_fitter` function.

        Example
        ----------
        .. code-block:: python

            fitter = NideconvFitter(func, onsets, TR=1.32)
            fitter.fit(debug=True)
        """

        # fitting
        utils.verbose(
            f"Fitting with '{self.fit_type}' minimization",
            self.verbose)
        if self.fit_type.lower() == "ols":
            self.model.fit(type=self.fit_type)
            self.fitters = self.model._get_response_fitters()
            self.format_fitters()

        utils.verbose("Fitting completed", self.verbose)

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

        if axs is None:
            if not hasattr(self, "figsize"):
                self.figsize = (8, 8)
            _, axs = plt.subplots(figsize=self.figsize)

        if not hasattr(self, "tc_condition"):
            self.timecourses_condition()

        # average across runs
        self.tc_df = utils.select_from_df(
            self.tc_condition,
            expression=f"covariate = {self.covariates}")

        # average across runs
        self.avg_across_runs = self.tc_df.groupby(
            ["event_type", "time"]).mean()

        if not isinstance(events, (list, np.ndarray)):
            events = self.cond
            self.event_indices = None
        else:
            utils.verbose(f"Flipping events to {events}", self.verbose)
            self.avg_across_runs = pd.concat(
                [utils.select_from_df(
                    self.avg_across_runs,
                    expression=f"event_type = {ii}")
                    for ii in events])

            # get list of switched indices
            self.event_indices = [list(events).index(ii) for ii in self.cond]

        # average across voxels
        self.avg_across_runs_voxels = self.avg_across_runs.mean(axis=1)

        # parse into list so it's compatible with LazyLine (requires an array
        # of lists)
        self.event_avg = self.avg_across_runs_voxels.groupby(
            "event_type").apply(np.hstack).to_list()
        self.event_sem = self.avg_across_runs.sem(
            axis=1).groupby("event_type").apply(
            np.hstack).to_list()
        self.event_std = self.avg_across_runs.std(
            axis=1).groupby("event_type").apply(
            np.hstack).to_list()

        # reorder base on indices again
        if isinstance(self.event_indices, list):
            for tt, gg in zip(
                ["avg", "sem", "std"],
                    [self.event_avg, self.event_sem, self.event_std]):
                reordered = [gg[i] for i in self.event_indices]
                setattr(self, f"event_{tt}", reordered)

        # shift all HRFs to zero
        if add_offset:
            for ev in range(len(self.event_avg)):
                if self.event_avg[ev][0] > 0:
                    self.event_avg[ev] -= self.event_avg[ev][0]
                else:
                    self.event_avg[ev] += abs(self.event_avg[ev][0])

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

        cols = list(self.tc_condition.columns)
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
                col_data = self.tc_condition[col][stim].values
                err_data = self.sem_condition[col][stim].values

                if add_offset:
                    if col_data[0] > 0:
                        col_data -= col_data[0]
                    else:
                        col_data += abs(col_data[0])

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

        if not axs:
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

        n_cols = len(list(self.cond))
        figsize = (n_cols * 6, 6)

        if not axs:
            fig = plt.figure(figsize=figsize)
            gs = fig.add_gridspec(1, n_cols)

        if error_type == "std":
            err = self.std_condition.copy()
        elif error_type == "sem":
            err = self.sem_condition.copy()
        else:
            err = self.std_condition.copy()

        if not isinstance(events, list) and not isinstance(events, np.ndarray):
            events = self.cond
        else:
            if self.verbose:
                print(f"Flipping events to {events}")

        for ix, event in enumerate(events):

            # add axis
            if not axs:
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
                self.tc_condition, expression=f"event_type = {event}")
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
                        indices=[ii]).values)
                col_error = np.squeeze(
                    utils.select_from_df(
                        error_df,
                        expression='ribbon',
                        indices=[ii]).values)

                # shift to zero
                if add_offset:
                    if col_data[0] > 0:
                        col_data -= col_data[0]
                    else:
                        col_data += abs(col_data[0])

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

            if not axs:
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
        if len(self.run_ids) <= 2:
            raise ValueError(
                "Crossvalidating is somewhat pointless with with fewer than 3 runs..")

        # get all possible combination of runs
        return utils.unique_combinations(self.run_ids, l=split)

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

        """predict_out_of_set

        Predicts the responses for unseen (left-out) runs based on the beta weights from trained data.

        Parameters
        ----------
        src : :class:`lazyfmri.fitting.NideconvFitter` or list of :class:`lazyfmri.fitting.NideconvFitter`
            The fitted model(s) used to obtain beta weights.
        trg : int, str, float, or list
            The run(s) for which predictions should be generated.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing R² values for each subject, where columns represent the fMRI voxels or regions.

        Example
        ----------
        .. code-block:: python

            r2_scores = cv.predict_out_of_set(trained_fitter, [1, 2])
        """

        if isinstance(src, list):
            self.betas = pd.DataFrame(pd.concat([src.fitters.iloc[i].betas for i in range(
                src.fitters.shape[0])], axis=1).mean(axis=1))
            ref_obj = src[0]
        else:
            ref_obj = src
            self.betas = src.fitters.iloc[0].betas

        if isinstance(trg, (float, int, str)):
            trg = [trg]

        cv_r2 = []
        sub_df = []
        sub_ids = utils.get_unique_ids(ref_obj.func, id="subject")
        for sub in sub_ids:

            sub_fitters = utils.select_from_df(
                src.fitters, expression=f"subject = {sub}")

            if isinstance(src, list):
                self.betas = pd.DataFrame(pd.concat([sub_fitters.iloc[i][0].betas for i in range(
                    sub_fitters.shape[0])], axis=1).mean(axis=1))
            else:
                self.betas = sub_fitters.iloc[0][0].betas

            run_r2 = []
            for run in trg:

                # find design matrix of unfitted run
                expr = (f"subject = {sub}", "&", f"run = {run}")
                X = utils.select_from_df(
                    self.full_fitters, expression=expr).iloc[0][0].X
                prediction = X.dot(self.betas)

                # get r2
                run_f = utils.select_from_df(
                    self.full_model.func, expression=expr)

                r2 = []
                for i in range(run_f.shape[-1]):
                    r2.append(metrics.r2_score(
                        run_f.values[:, i], prediction.values[:, i]))

                run_r2.append(np.array(r2))

            # average over unfitted runs
            run_r2 = np.array(run_r2).mean(axis=0)
            sub_df.append(np.array(run_r2))

        sub_r2 = np.array(sub_df)
        cv_r2 = pd.DataFrame(sub_r2, columns=list(self.func.columns))
        cv_r2["subject"] = sub_ids

        return cv_r2

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
        self.full_fitters = self.full_model.model._get_response_fitters()

        self.combos = self.return_combinations(split=split)

        # loop through them
        self.fitters = []
        self.r2_ = []
        for ix, combo in enumerate(self.combos):
            unfitted_runs = [i for i in self.run_ids if i not in combo]

            self.fold_f = pd.concat([utils.select_from_df(
                self.func, expression=f"run = {i}") for i in combo])
            self.fold_o = pd.concat([utils.select_from_df(
                self.onsets, expression=f"run = {i}") for i in combo])

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
        self.r2_.set_index(["subject", "fold"], inplace=True)


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
            If `True`, sets all extracted parameters to zero for flat (zero-variance) timecourses. Default is `True`.
        debug : bool, optional
            If `True`, prints additional debugging information during parameter extraction. Default is `False`.
        progress : bool, optional
            If `True`, displays a progress bar using `alive_progress`. Default is `False`.
        thr_var : int or float, optional
            Variance threshold for input profiles. Time-series with variance below this threshold are ignored. Default is `0`.
        shift : int or float, optional
            A constant shift applied to all profiles before extraction to avoid negative baselines. Default is `0`.
        vox_as_index : bool, optional
            If `True`, uses DataFrame column indices as voxel identifiers. Otherwise, original column names are retained. Default is `False`.
        col_name : str, optional
            Custom name for the added index when processing 2D DataFrames. Example: `"roi"` for region-based processing. Default is `"vox"`.
        peak : int, optional
            Specifies which peak to use when multiple peaks exist. Uses `scipy.signal.find_peaks` to identify peaks.
            A value of `1` selects the largest peak (default).

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

        """_get_metrics

        Extracts key HRF metrics, including peak amplitude, time-to-peak, full-width half-maximum (FWHM), and area under the curve (AUC).

        Parameters
        ----------
        hrf : pd.DataFrame or np.ndarray
            The input HRF time-series.
        TR : float, optional
            Repetition time in seconds, required if input is a NumPy array.
        force_pos : bool or list, optional
            Forces extraction from the **positive** response peak, by default `False`.
        force_neg : bool or list, optional
            Forces extraction from the **negative** response peak, by default `False`.
        plot : bool, optional
            If `True`, plots the HRF profile for debugging, by default `False`.
        nan_policy : bool, optional
            If `True`, sets NaN values when HRF extraction fails, by default `True`.
        debug : bool, optional
            If `True`, prints debugging information, by default `False`.
        progress : bool, optional
            Displays a progress bar, by default `False`.
        thr_var : int or float, optional
            Variance threshold for filtering time-series, by default `0`.
        shift : int or float, optional
            Shifts the HRF profile to avoid negative baselines, by default `0`.
        vox_as_index : bool, optional
            If `True`, uses DataFrame column indices as voxel identifiers, by default `False`.
        col_name : str, optional
            Name for index column in the output DataFrame, by default `"vox"`.
        incl : list, optional
            List of HRF parameters to extract, by default includes `magnitude`, `fwhm`, `time_to_peak`, etc.
        peak : int, optional
            Specifies which peak to use in case of multiple peaks, by default `None`.

        Returns
        -------
        pd.DataFrame
            DataFrame containing extracted HRF parameters.
        np.ndarray
            Final processed HRF.

        Example
        ----------
        .. code-block:: python

            metrics, final_hrf = HRFMetrics._get_metrics(some_profile, TR=1.5)
        """

        # force into list for iterability
        if isinstance(incl, str):
            incl = [incl]

        # set shift to None because this doesn't deal with 2D arrays; just get
        # the format right
        hrf = self._verify_input(
            hrf,
            TR=TR,
            shift=None
        )

        orig_cols = list(hrf.columns)

        # find columns abs(var)>0
        filtered_variance = (hrf.var(axis=0) > thr_var).values
        filtered_df = hrf.iloc[:, filtered_variance]
        cols = list(filtered_df.columns)
        utils.verbose(
            f" {filtered_df.shape[1]}/{hrf.shape[1]}\tvertices survived variance threshold of {thr_var}",
            debug
        )

        if len(cols) == 0:
            raise ValueError(
                f"Variance threshold of {thr_var} is too strict; no vertices survived")

        if not isinstance(force_neg, list):
            force_neg = [force_neg for _ in cols]

        if not isinstance(force_pos, list):
            force_pos = [force_pos for _ in cols]

        # print(force_pos)
        col_metrics = []
        col_fwhm = []

        # initialize empty dataframe
        metrics = pd.DataFrame(
            np.zeros(
                (len(orig_cols), len(incl))), columns=incl)

        # separate loop for fancy progress bar
        if progress:
            with alive_bar(filtered_df.shape[1], force_tty=True) as bar:
                for ix, col in enumerate(filtered_df):
                    pars, fwhm_, final_hrf = self._get_single_hrf_metrics(
                        filtered_df[col],
                        TR=TR,
                        force_pos=force_pos[ix],
                        force_neg=force_neg[ix],
                        **kwargs
                    )

                    # progress
                    bar()

                    col_metrics.append(pars)
                    col_fwhm.append(fwhm_)
        else:
            for ix, col in enumerate(filtered_df):

                pars, fwhm_, final_hrf = self._get_single_hrf_metrics(
                    filtered_df[col],
                    TR=TR,
                    force_pos=force_pos[ix],
                    force_neg=force_neg[ix],
                    **kwargs
                )

                col_metrics.append(pars)
                col_fwhm.append(fwhm_)

        if len(col_metrics) > 0:
            col_metrics = pd.concat(col_metrics)

        # insert into empty dataframe
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            metrics.iloc[filtered_variance, :] = col_metrics.values.copy()

        if len(orig_cols) > 1:
            if vox_as_index:
                set_as = np.arange(0, len(orig_cols), dtype=int)
            else:
                set_as = orig_cols

            metrics[col_name] = set_as

        # set to integer
        if "magnitude_ix" in list(metrics.columns):
            metrics["magnitude_ix"] = metrics["magnitude_ix"].astype(int)

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

        # shift to zero
        if isinstance(shift, int):
            if shift > 1:
                bsl = hrf.values[:shift].mean()
            else:
                bsl = hrf.values[0]

            if bsl > 0:
                hrf -= bsl
            elif bsl < 0:
                hrf += abs(bsl)
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
    def _resample(
        self,
        hrf,
        resample_to_shape=500
        ):

        time, time_col = self._get_time(hrf)
        time_rs = np.linspace(
            time[0],
            time[-1],
            resample_to_shape
        )

        hrf_rs = pd.DataFrame(
            glm.resample_stim_vector(
                hrf.values,
                resample_to_shape,
                interpolate="linear"
            ).squeeze(),
            columns=hrf.columns
        )
        hrf_rs[time_col] = time_rs
        hrf_rs.set_index([time_col], inplace=True)

        return time_rs, hrf_rs
    
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

        if intsc == np.nan or verify_slope:
            
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

            if intsc == np.nan and not nan_policy:
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
    def _get_amplitude(
        self,
        hrf,
        force_pos=False,
        force_neg=False,
        peak=None,
        t_start=0,
        resample_to_shape=500,
        max_scaler=0.05,
        noise_scaler=1,
        verify_peaks=False,
        **kwargs
        ):

        """_get_amplitude

        Extracts the peak amplitude of the HRF.

        Parameters
        ----------
        hrf : pd.DataFrame
            The input HRF time-series.
        force_pos : bool, optional
            If `True`, forces extraction from a positive response, by default `False`.
        force_neg : bool, optional
            If `True`, forces extraction from a negative response, by default `False`.
        peak : int, optional
            Specifies which peak to use, by default `None`.

        Returns
        -------
        dict
            Dictionary containing:
            - `"amplitude"`: The peak value.
            - `"t"`: Time at which the peak occurs.
            - `"t_ix"`: Index of the peak.

        Example
        ----------
        .. code-block:: python

            amplitude_info = HRFMetrics._get_amplitude(hrf_data)
        """

        # fetch time stamps
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
                    
        # use scipy.signal.find_peaks to find n'th peak
        if isinstance(peak, int):

            ref = hrf_corr.values.squeeze()

            prominence = max(max_scaler*ref.max(), noise_scaler*np.std(ref))
            pos = list(
                signal.find_peaks(
                    ref,
                    prominence=prominence
                )[0]
            )

            # print(f"Before: {pos}")
            pos = [i for i in pos if hrf_corr.values.squeeze()[int(i)]>0]
            # print(f"After: {pos}")
            
            neg = list(
                signal.find_peaks(
                    -ref,
                    prominence=prominence
                )[0]
            )

            neg = [i for i in neg if hrf_corr.values.squeeze()[int(i)]<0]

            if force_pos or force_neg:
                if force_pos:
                    ppeaks = pos
            
                if force_neg:
                    ppeaks = neg
            else:
                ppeaks = sorted(pos + neg)



            if verify_peaks:
                pl = self.plot_profile_for_debugging(
                    hrf_rs,
                    peaks=[i+t0_idx for i in ppeaks],
                    **kwargs
                )

            if len(ppeaks) > peak - 1:
                mag_ix = ppeaks[peak - 1]
                mag = ref[mag_ix]

                t_ix = utils.find_nearest(
                    hrf_rs,
                    mag
                )[0]

                ddict = {
                    "amplitude": ref[mag_ix],
                    "t": time_rs[t_ix],
                    "t_ix": t_ix
                }
            else:
                # go for absolute max
                negative = self._check_negative(
                    hrf_corr,
                    force_neg=force_neg,
                    force_pos=force_pos
                )

                if not force_pos:
                    if negative:
                        mag_tmp = hrf_corr.min(axis=0).values
                    else:
                        mag_tmp = hrf_corr.max(axis=0).values
                else:
                    mag_tmp = hrf_corr.max(axis=0).values

                mag_ix = utils.find_nearest(
                    hrf_rs.values.squeeze(),
                    mag_tmp
                )[0]

                ddict = {
                    "amplitude": mag_tmp[0],
                    "t": time_rs[mag_ix],
                    "t_ix": mag_ix
                }         

        else:

            # check negative:
            negative = self._check_negative(
                hrf_corr,
                force_neg=force_neg,
                force_pos=force_pos
            )

            if not force_pos:
                if negative:
                    mag_tmp = hrf_corr.min(axis=0).values
                else:
                    mag_tmp = hrf_corr.max(axis=0).values
            else:
                mag_tmp = hrf_corr.max(axis=0).values

            mag_ix = utils.find_nearest(
                hrf_rs.values.squeeze(),
                mag_tmp
            )[0]

            ddict = {
                "amplitude": mag_tmp[0],
                "t": time_rs[mag_ix],
                "t_ix": mag_ix
            }

        return ddict

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

        hrf_corr_arr = hrf_corr.values.squeeze()

        if normalize:
            hrf_corr_arr /= hrf_corr_arr.max()

        # Separate positive and negative parts
        pos_mask = hrf_corr_arr > 0
        neg_mask = hrf_corr_arr < 0

        def compute_auc_safely(x, y):
            valid = np.isfinite(x) & np.isfinite(y)
            if np.any(valid) and x[valid].shape[0] > 1 and y[valid].shape[0] > 1:
                return abs(metrics.auc(x[valid], y[valid]))
            else:
                return 0.0  # No valid points → area is 0

        # Compute positive and negative AUCs safely
        pos_area = compute_auc_safely(time_corr[pos_mask], hrf_corr_arr[pos_mask])
        neg_area = compute_auc_safely(time_corr[neg_mask], hrf_corr_arr[neg_mask])

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
                amplitude=mag["amplitude"]
            )

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
    def __init__(
        self,
        x,
        hrf,
        amplitude=None,
        negative=False,
        resample=500,
        ):
        """FWHM

        Class to extract the full-width at half-max of a given profile. The profile can either have a positive or negative
        peak, in which case the `negative`-flag should be set accordingly. The final output is stored in the `fwhm` attribute
        of the class.

        Parameters
        ----------
        x: np.ndarray
            Generally this an array reflecting the time dimension
        hrf: np.ndarray
            Input profile (it says `hrf`, but it can be anything; also 1D pRF profiles)
        negative: bool, optional
            Assume the FWHM needs to be extracted from a negative bump or not
        resample: int, optional
            Resample the input to a higher grid so that the intersection between the profile and half-max can be drawn more
            accurately
        """

        self.x = x
        self.hrf = hrf
        self.resample = resample
        self.negative = negative
        self.amplitude = amplitude
        if x.shape[0] < self.resample:
            self.use_x = glm.resample_stim_vector(self.x, self.resample, interpolate="linear")
            self.use_rf = glm.resample_stim_vector(self.hrf, self.resample, interpolate="linear")
        else:
            self.use_x = x.copy()
            self.use_rf = hrf.copy()

        # Use amplitude if given
        if self.amplitude is not None:
            self.half_max = self.amplitude / 2.0
        else:
            self.half_max = min(self.use_rf) / 2.0 if self.negative else max(self.use_rf) / 2.0

        # Get intersection points with half-max
        try:
            self.hmx = utils.find_intersection(
                self.use_x,
                self.use_rf,
                np.full_like(self.use_rf, self.half_max)
            )
        except Exception:
            self.hmx = []

        # Try to select 2 points around the peak
        self.ts = []
        if isinstance(self.hmx, list) and len(self.hmx) >= 2:
            try:
                peak_ix = np.argmin(self.use_rf) if self.negative else np.argmax(self.use_rf)
                peak_time = self.use_x[peak_ix]
                hmx_times = [i[0][0] for i in self.hmx]

                # Split intersections into pre- and post-peak
                pre = [t for t in hmx_times if t < peak_time]
                post = [t for t in hmx_times if t > peak_time]

                if len(pre) > 0 and len(post) > 0:
                    # Take the closest before and after the peak
                    self.ts = [max(pre), min(post)]
                else:
                    raise ValueError("Could not find FWHM on both sides of peak.")
            except Exception:
                self.ts = []

        if len(self.hmx) < 2:
            if len(self.hmx) == 1:
                # Fallback: use single intersection and end of window
                self.ts = [self.hmx[0][0][0], self.use_x[-1]]
            else:
                raise ValueError("Could not find sufficient points close to half max..")

        # Fallback: just grab first 2 if above fails
        if len(self.ts) < 2 and isinstance(self.hmx, list) and len(self.hmx) >= 2:
            self.ts = sorted([i[0][0] for i in self.hmx[:2]])

        # If still insufficient points, use full window as fallback
        if len(self.ts) < 2:
            self.t0_ = self.use_x[0]
            self.t1_ = self.use_x[-1]
        else:
            self.t0_, self.t1_ = self.ts

        self.fwhm = abs(self.t1_ - self.t0_)

    def half_max_x(
        self,
        x,
        y,
        amplitude=None
        ):

        if max is None:
            if not self.negative:
                half = max(y) / 2.0
            else:
                half = min(y) / 2.0
        else:
            half = amplitude / 2.0

        return utils.find_intersection(x, y, np.full_like(y, half))

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

    # correct for negative onsets
    if x[0] < 0:
        x -= x[0]

    if not negative:
        y = (x / (d1))**a1 * np.exp(-(x - d1) / b1) - \
            c * (x / (d2))**a2 * np.exp(-(x - d2) / b2)
    else:
        y = (x / (d1))**a1 * -np.exp(-(x - d1) / b1) - \
            c * (x / (d2))**a2 * -np.exp(-(x - d2) / b2)

    y[x < 0] = 0

    return y


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

    time_points = np.linspace(*interval,
                              np.rint(float(interval[-1]) / dt).astype(int))
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
        self.interval = interval
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

        dt = 1 / self.osf
        time_points = np.linspace(
            *self.interval, np.rint(float(self.interval[-1]) / dt).astype(int))

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
            time_points = np.linspace(*self.interval, num=resample_to_shape)

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
        self.interval = interval
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
    def correct_baseline(
        d_,
        bsl=20,
        verbose=False
        ):

        """Correct the baseline of time-series data.

        Shifts the data to ensure the baseline is centered around zero. If the mean of the first `bsl` timepoints
        is negative, it shifts the entire time-series upwards. If positive, it shifts downwards.

        Parameters
        ----------
        d_ : pd.DataFrame or np.ndarray
            Input time-series data. If a dataframe is provided, the operation is applied per column.
        bsl : int, optional
            Number of initial timepoints to use for baseline correction. Default is `20`.
        verbose : bool, optional
            If `True`, prints shift information. Default is `False`.

        Returns
        -------
        pd.DataFrame or np.ndarray
            Baseline-corrected data with the same format as input.
        """

        if isinstance(d_, pd.DataFrame):
            d_vals = d_.values
            return_df = True
        else:
            d_vals = d_.copy()
            return_df = False

        if d_vals.ndim < 2:
            d_vals = d_vals[..., np.newaxis]

        shift_cols = []
        for col in range(d_vals.shape[-1]):

            col_vals = d_vals[:, col]
            m_ = col_vals[:bsl].mean()
            if m_ < 0:
                utils.verbose(
                    f"Shifting profile UP with {round(abs(m_),2)}", verbose)
                d_shift = col_vals + abs(m_)
            else:
                utils.verbose(
                    f"Shifting profile DOWN with {round(m_,2)}", verbose)
                d_shift = col_vals - m_

            if d_shift.ndim < 2:
                d_shift = d_shift[..., np.newaxis]

            shift_cols.append(d_shift)

        shift_cols = np.concatenate(shift_cols, axis=1)
        if return_df:
            return pd.DataFrame(shift_cols, index=d_.index)
        else:
            return d_shift

    @staticmethod
    def _get_epoch_timepoints(interval, TR):

        """Generate timepoints for the epoch window.

        Computes a time array covering the epoch interval at the given TR.

        Parameters
        ----------
        interval : list
            The time interval for epoch extraction, specified as `[start, end]`.
        TR : float
            Repetition time (TR) in seconds.

        Returns
        -------
        np.ndarray
            Array of timepoints spanning the given interval with a step size equal to `TR`.
        """

        total_length = interval[1] - interval[0]
        timepoints = np.linspace(
            interval[0],
            interval[1],
            int(total_length * (1 / TR) * 1),
            endpoint=False)

        return timepoints

    def epoch_events(
        self,
        df_func,
        df_onsets,
        index=False,
        ):

        """Extract event-specific epochs from functional data.

        Iterates over all events in the provided onset dataframe and extracts corresponding time windows from
        the functional data.

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
            Extracted epochs with time (`t`), event type, subject, and run information.
        """

        ev_epochs = []

        info = {}
        for el in ["event_type", "subject", "run"]:
            info[el] = utils.get_unique_ids(df_onsets, id=el)

        df_idx = ["subject", "run", "event_type", "epoch", "t"]
        for ev in info["event_type"]:

            epochs = []
            ev_onsets = utils.select_from_df(
                df_onsets, expression=f"event_type = {ev}")
            for i, t in enumerate(ev_onsets.onset.values):

                if ev not in ["response", "blink"]:
                    # find closest ID to interval
                    t_start = t - abs(self.interval[0])
                    ix_start = int(t_start / self.TR)

                    samples = self.interval[1] + abs(self.interval[0])
                    n_sampl = int(samples / self.TR)
                    ix_end = ix_start + n_sampl

                    # extract from data
                    # [...,np.newaxis]
                    stim_epoch = df_func.iloc[ix_start:ix_end, :].values
                    time = self._get_epoch_timepoints(self.interval, self.TR)

                    df_stim_epoch = pd.DataFrame(stim_epoch)

                    if len(time) != len(stim_epoch):
                        print(
                            f"""WARNING: could not extract full epoch around event; onset={round(t,2)}s,
t_start={round(t_start,2)}s with {n_sampl} samples ({samples}s).
Epoch has {len(stim_epoch)} samples""")

                    df_stim_epoch["t"], df_stim_epoch["epoch"], df_stim_epoch["event_type"] = time[:len(
                        stim_epoch)], i, ev
                    epochs.append(df_stim_epoch)

            # concatenate into 3d array (time,ev,stimulus_nr), average, and
            # store in dataframe
            ev_epoch = pd.concat(epochs)

            # format dataframe
            ev_epoch["subject"], ev_epoch["run"] = info["subject"][0], info["run"][0]

            ev_epochs.append(ev_epoch)

        # concatenate into dataframe
        df_epochs = pd.concat(ev_epochs)

        # set index or not
        if index:
            df_epochs.set_index(df_idx, inplace=True)

        return df_epochs

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

    def epoch_subjects(
        self,
        df_func,
        df_onsets
        ):

        """Extract epochs for all subjects.

        Loops over all subjects in the dataset, extracts subject-specific functional data, and applies :func:`lazyfmri.fitting.Epoch.epoch_tasks()` 
        or :func:`lazyfmri.fitting.Epoch.epoch_runs()` to process all tasks/runs within each subject.

        Parameters
        ----------
        df_func : pd.DataFrame
            Functional data indexed on `subject`, `task`, `run`, and `t`.
        df_onsets : pd.DataFrame
            Onset data indexed on `subject`, `task`, `run`, and `event_type`.

        Returns
        -------
        pd.DataFrame
            Extracted epochs for all subjects combined.
        """

        self.sub_ids = utils.get_unique_ids(self.func, id="subject")

        # loop through subject IDs
        sub_df = []
        for sub in self.sub_ids:

            # extract task-specific dataframes
            expr = f"subject = {sub}"
            self.sub_func = utils.select_from_df(df_func, expression=expr)
            self.sub_stims = utils.select_from_df(df_onsets, expression=expr)

            try:
                self.task_ids = utils.get_unique_ids(self.sub_func, id="task")
            except BaseException:
                self.task_ids = None

            if isinstance(self.task_ids, list):
                sub_epoch = self.epoch_tasks(self.sub_func, self.sub_stims)
                self.idx = [
                    "subject",
                    "task",
                    "run",
                    "event_type",
                    "epoch",
                    "t"]
            else:
                sub_epoch = self.epoch_runs(self.sub_func, self.sub_stims)
                self.idx = ["subject", "run", "event_type", "epoch", "t"]

            sub_epoch.set_index(self.idx, inplace=True)
            sub_df.append(sub_epoch)

        sub_df = pd.concat(sub_df)

        return sub_df

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
                arr[start:start + fc] = 1
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
        for i in ["subject", "task", "run"]:
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
        for i in ["subject", "task", "run"]:
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
        for i in ["subject", "task", "run"]:
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
    def glm_runs(
        self,
        df_func,
        df_onsets,
        **kwargs
        ):

        # loop through runs
        self.run_ids = utils.get_unique_ids(df_func, id="run")
        # print(f"task-{task}\t| runs = {run_ids}")
        run_df = []
        for run in self.run_ids:

            expr = f"run = {run}"
            run_func = utils.select_from_df(df_func, expression=expr)
            run_stims = utils.select_from_df(df_onsets, expression=expr)

            # get glms
            df = self.single_glm(
                run_func,
                run_stims,
                **kwargs
            )

            run_df.append(df)

        return run_df

    @classmethod
    def glm_tasks(
        self,
        df_func,
        df_onsets,
        **kwargs
        ):

        # read task IDs
        self.task_ids = utils.get_unique_ids(df_func, id="task")

        # loop through task IDs
        task_df = []
        for task in self.task_ids:

            # extract task-specific dataframes
            expr = f"task = {task}"
            task_func = utils.select_from_df(df_func, expression=expr)
            task_stims = utils.select_from_df(df_onsets, expression=expr)

            df = self.glm_runs(
                task_func,
                task_stims,
                **kwargs
            )

            task_df += df

        return task_df

    @classmethod
    def glm_subjects(
        self,
        df_func,
        df_onsets,
        **kwargs
        ):

        self.sub_ids = utils.get_unique_ids(df_func, id="subject")

        # loop through subject IDs
        sub_df = []
        for sub in self.sub_ids:

            # extract task-specific dataframes
            expr = f"subject = {sub}"
            self.sub_func = utils.select_from_df(df_func, expression=expr)
            self.sub_stims = utils.select_from_df(df_onsets, expression=expr)

            try:
                self.task_ids = utils.get_unique_ids(self.sub_func, id="task")
            except BaseException:
                self.task_ids = None

            if isinstance(self.task_ids, list):
                sub_glm = self.glm_tasks(
                    self.sub_func,
                    self.sub_stims,
                    **kwargs
                )
                self.idx = ["subject", "task", "run"]
            else:
                sub_glm = self.glm_runs(
                    self.sub_func,
                    self.sub_stims,
                    **kwargs
                )
                self.idx = ["subject", "run"]

            # sub_glm.set_index(self.idx, inplace=True)
            sub_df += sub_glm

        # sub_df = pd.concat(sub_df)
        return sub_df

    def glm_input(self, **kwargs):

        # run glm
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

        if not isinstance(df, pd.DataFrame):
            df = self.glm_output["r2"]

        r2 = df.iloc[:, vox_nr].values
        if len(r2) > 0:
            r2 = r2.mean()
        else:
            r2 = r2[0]

        return r2

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
