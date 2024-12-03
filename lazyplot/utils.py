import math
import numpy as np
import os
import operator
import pandas as pd

opj = os.path.join


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

    Select a subset of a dataframe based on an expression. Dataframe should be indexed by the variable you want to select on or have the variable specified in the expression argument as column name. If index is True, the dataframe will be indexed by the selected variable. If indices is specified, the dataframe will be indexed by the indices specified through a list (only select the elements in the list) or a `range`-object (select within range).

    Parameters
    ----------
    df: pandas.DataFrame
        input dataframe
    expression: str, optional
        what subject of the dataframe to select, by default None. The expression must consist of a variable name and an operator. The operator can be any of the following: '=', '>', '<', '>=', '<=', '!=', separated by spaces. You can also change 2 operations by specifying the `&`-operator between the two expressions. If you want to use `indices`, specify `expression="ribbon"`.
    index: bool, optional
        return output dataframe with the same indexing as `df`, by default True
    indices: list, range, numpy.ndarray, optional
        List, range, or numpy array of indices to select from `df`, by default None
    match_exact: bool, optional:
        When you insert a list of strings with `indices` to be filtered from the dataframe, you can either request that the items of `indices` should **match** exactly (`match_exact=True`, default) the column names of `df`, or whether the columns of `df` should **contain** the items of `indices` (`match_exact=False`).

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
    See https://linescanning.readthedocs.io/en/latest/examples/nideconv.html for an example of how to use this function (do ctrl+F and enter "select_from_df").
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

    if not isinstance(expression, (str, tuple, list)):
        raise ValueError(
            f"Please specify expressions to apply to the dataframe. Input is '{expression}' of type ({type(expression)})")

    if expression == "ribbon":

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
                f"Unknown type '{type(indices)}' for indices; must be a tuple of 2 values representing a range, or a list/array of indices to select")
    else:
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
        raise ValueError(f"List is empty")

    start_df = df.copy()
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
        raise ValueError(f"Please specify a identifier from the dataframe")

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

    except Exception as e:
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

    This function returns the file given a path and a substring. Avoids annoying stuff with glob. Now also allows multiple filters to be applied to the list of files in the directory. The idea here is to construct a binary matrix of shape (files_in_directory, nr_of_filters), and test for each filter if it exists in the filename. If all filters are present in a file, then the entire row should be 1. This is what we'll be looking for. If multiple files are found in this manner, a list of paths is returned. If only 1 file was found, the string representing the filepath will be returned.

    Parameters
    ----------
    filt: str, list
        tag for files we need to select. Now also support a list of multiple filters.
    path: str
        path to the directory from which we need to remove files
    return_msg: str, optional
        whether to raise an error (*return_msg='error') or return None (*return_msg=None*). Default = 'error'.
    exclude: str, optional:
        Specify string to exclude from options. This criteria will be ensued after finding files that conform to `filt` as final filter.

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

        # now we have a binary <number of files x number of filters> array. If all filters were available in a file, the entire row should be 1,
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
        if not el in list(kwargs.keys()):
            kwargs[el] = val
    else:
        kwargs[el] = val

    return kwargs
