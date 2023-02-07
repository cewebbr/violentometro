import numpy as np
import pandas as pd

#import xavy.explore as xe


def bold(text):
    """
    Takes a string and returns it bold.
    """
    return '\033[1m'+text+'\033[0m'


def print_string_series(series, max_rows=100):
    """
    Pretty print `series` of strings.
    """
    n = 1
    for i, v in zip(series.index, series.values):
        print('{}: {}'.format(bold(str(i)), v))
        n = n + 1
        if n > max_rows:
            break


def check_guarda_compartilhada(df, col_subordinada, col_raiz, drop_unique=True):
    """
    Return a Series whose index are the unique
    entries in `df` (DataFrame) column `col_subordinada`
    (str or list of str) and values are arrays of unique 
    entries in `col_raiz` (str) that appear in `df` 
    associated to each index.
    
    This function can be used to:
    - Double check that elements in `col_subordinada`
      are subdivisions of `col_raiz` completely contained
      in each element of `col_raiz`.
    - Find out which ramifications of `col_subordinada` there
      are listed in column `col_raiz`.
    """
    
    n_raiz_por_subordinado = df.groupby(col_subordinada)[col_raiz].unique()
    if drop_unique:
        return n_raiz_por_subordinado.loc[n_raiz_por_subordinado.str.len() > 1]
    else:
        return n_raiz_por_subordinado


def print_array_series(series):
    """
    Print a Series of arrays.
    """
    n = len(series)
        
    for i in range(n):
        print(bold(str(series.index[i]) + ': ') + ' / '.join(series.iloc[i]))


def crop_strings(series, str_range, ellipsis='…'):
    """
    Get slices of the strings, marking the removed ends 
    with ellipsis.
    
    Parameters
    ----------
    series : Pandas Series or Index object.
        The array-like containing the strings to slice.
    str_range : tuple of non-negative ints
        The positions (inclusive, exclusive) where to 
        slice the string.
    ellipsis : str
        The string representing an ellipsis, to be 
        added to cropped strings.
    
    Returns
    -------
    str_arr : array pr Series
        An array of sliced strings, with ellipsis marking
        the removal of string parts when required. Return 
        a Series with the same index as `series` if the 
        latter is a Series object.
    """
    
    if str_range[0] == 0:
        y = np.where((series.str.len() > str_range[1]), series.str.slice(*str_range) + ellipsis, series)
    else:
        y = np.where((series.str.len() > str_range[1] - str_range[0]), ellipsis + series.str.slice(*str_range) + ellipsis, ellipsis + series.str.slice(*str_range))
        
    if type(series) == pd.core.series.Series:
        return pd.Series(y, index=series.index)
    else:
        return y


def date_series_replace(series, year=None, month=None, day=None):
    """
    Replace date elements with the ones specified.
    
    Parameters
    ----------
    series : datetime Series
        The series with dates to be altered.
    year : int
        The fixed year used to replace the year in `series`.
    month : int
        The fixed month used to replace the year in `series`.
    day : int
        The fixed day used to replace the year in `series`.
    
    Returns
    -------
    date_series : datetime Series
        A new series like `series` but with the replacements
        made.
    """
    return pd.to_datetime(
        {'year':  series.dt.year if year is None else year,
         'month': series.dt.month if month is None else month,
         'day':   series.dt.day if day is None else day})


def unique_traits(df, agg_by, id_cols, agg_is_index=False):
    """
    Assert that columns `id_cols` (str or 
    list of str) only have one possible value 
    when `df` is grouped by columns `agg_by` 
    (str or list of str) and return these 
    unique values as a DataFrame. If 
    `agg_is_index` (bool) is True, its index 
    are `agg_by`, otherwise `agg_by` are 
    columns.
    """

    # Security check:
    assert type(agg_is_index)
    
    # Standardizing input:
    if type(agg_by) == str or type(agg_by) == int:
        agg_by = [agg_by]
    if type(id_cols) == str or type(id_cols) == int:
        id_cols = [id_cols]
    
    # Checking there is only one value in each column:
    grouped = df.groupby(agg_by)[agg_by + id_cols]
    assert (grouped.nunique() == 1).all().all(), 'Encontrado mais de um identificador para o agrupamento.'
    # Extract these values:
    id_df = grouped.head(1)
    # Set index:
    if agg_is_index:
        id_df = id_df.set_index(agg_by)
    
    return id_df


def add_column_suffix(df, suffix, inplace=False):
    """
    Rename a DataFrame's columns by adding a suffix
    to them.
    
    Parameters
    ----------
    df : DataFrame
        The DataFrame to change its columns' names.
    suffix : str
        The suffix to be appended to `df`'s column
        names.
    inplace : bool
        Whether to change the columns' names in place
        and return None or to keep `df` as it is and
        return the transformed DataFrame
        
    Returns
    -------
    
    new_df : DataFrame or None
        If `inplace` is False, return the DataFrame
        with the new column names. Otherwise, return
        None.
    """
    # Build translator:
    cols = df.columns
    new_cols = [str(col) + suffix for col in cols]
    translator = dict(zip(cols, new_cols))
    
    # Rename columns:
    if inplace == True:
        df.rename(translator, axis=1, inplace=True)
    else:
        return df.rename(translator, axis=1)
    
    return None


def rename_columns(df, pattern, replace, regex=True):
    """
    Rename DataFrame columns by replacing a pattern.
    
    Parameters
    ----------
    df : DataFrame
        DataFrame whose columns should be renamed.
    pattern : str
        Pattern (can be regex) to be replaced.
    replace : str
        Replacement string for `pattern`.
    regex : bool
        Whether `pattern` is a regular expression or not.
    
    Returns
    -------
    renamed_df : DataFrame
        A copy of `df` but with renamed columns.
    """
    
    # Build translation dict:
    x = df.columns
    y = x.str.replace(pattern, replace, regex=regex)
    m = {k:v for k,v in zip(x, y)}
    
    # Rename:
    new = df.rename(m, axis=1)
    
    return new


def build_onehot_df(cat_df, sort_by_cardinality=True, categories='auto', drop=None):
    """
    Given a DataFrame containing only categorical 
    columns, return a DataFrame of its one-hot 
    encodings.
    
    Parameters
    ----------
    cat_df : DataFrame
        DataFrame with categorical columns only.
    sort_by_cardinality : bool
        Wether to sort the columns of `cat_df` 
        by the number of different categories 
        they contain.
    categories : 'auto' or a list of array-like
        Categories in each column (see OneHotEncoder 
        docs).
    drop : {'first', 'if_binary'}, None, or a array-like 
        of shape (n_features,).

    Returns
    -------
    onehot_df : DataFrame
        DataFrame with one-hot encoding of 
        `cat_df`.
    """
    from sklearn.preprocessing import OneHotEncoder
    
    # Security check:
    assert type(sort_by_cardinality) == bool
    
    # Sort columns by number of distinct values:
    if sort_by_cardinality:
        sorted_cols = cat_df.nunique().sort_values().index
    # Or not:
    else:
        sorted_cols = cat_df.columns
    
    # One-hot encode features:
    cat_encoder = OneHotEncoder(sparse=False, categories=categories, drop=drop)
    encoded_data = cat_encoder.fit_transform(cat_df[sorted_cols])
    feature_names = cat_encoder.get_feature_names(sorted_cols)

    # Build DataFrame:
    onehot_df = pd.DataFrame(data=encoded_data, columns=feature_names, index=cat_df.index)
    onehot_df = onehot_df.astype(int)
    
    return onehot_df


def find_low_ncat_cols(df, max_ncats=None):
    """
    Return a Series with the number of 
    unique values per column.
    """
    n_cats = df.nunique().sort_values()
    
    if max_ncats is None:
        return n_cats
    else:
        low_ncat_cols = n_cats.loc[n_cats <= max_ncats]
        return low_ncat_cols


def prepare_series_for_cross_join(series, index=0):
    """
    Returns a Pandas Series containing the unique 
    values in `series` (Series), all associated 
    to the same `index` (int or str).
    """
    
    # Security checks:
    assert type(index) in (str, int), '`index` must be of type str or int.'
    
    # Get unique values:
    unique_values = series.drop_duplicates().values
    # Build Series:
    cj_series = pd.Series(unique_values, index=[index]*len(unique_values), name=series.name)
    
    return cj_series

    
def cross_join_columns(df, ascending=None):
    """
    Create a DataFrame with all unique combinations 
    of values from all input columns.
    
    Parameters
    ----------
    df : DataFrame
        Data containing the columns to be combined 
        so all possible combinations of their values
        are given in the output. Repeated values 
        are ignored.
    ascending : bool or None or list of (bool, None).
        If None, do not sort the columns (use the 
        order given by the first appearance of 
        each value). If True, sort the values in 
        the columns in ascending order. If False, 
        sort in reverse order.
        If list, use one of the criteria above for 
        each column in `df`.
    
    Returns
    -------
    cross_joined_df : DataFrame
        DataFrame with all distinct combinations of 
        values in `df` columns, with standard 
        indexing (0, 1, 2, ...).
    """
    
    # Standardize input:
    if type(ascending) in (type(None), bool):
        ascending = [ascending] * len(df.columns)
    else:
        assert len(ascending) == len(df.columns), '`ascending` must be the same length as the number of columns in `df`.'

    # Security check:
    for a in ascending:
        assert a in (None, True, False), '`ascending` can only contain True, False or None.'

    # Loop over columns, cross-joining on the way:
    cross_joined_df = pd.DataFrame(index=[0])    
    for c in range(len(df.columns)):

        # Create unique Series:
        cj_series = prepare_series_for_cross_join(df.iloc[:, c])

        # Sort, if requested:
        if ascending[c] is not None:
            cj_series = cj_series.sort_values(ascending=ascending[c])

        # Cross join:
        cross_joined_df = cross_joined_df.join(cj_series)

    # Reset index:
    cross_joined_df.reset_index(inplace=True, drop=True)
    
    return cross_joined_df


def cross_join_dfs(df1, df2):
    """
    Cross join two DataFrame, i.e. combine all rows from
    the first to all rows from the second. The final index
    is reset.
    """
    
    # Set all indexes to the same values:
    df1.index = np.zeros_like(df1.index)
    df2.index = np.zeros_like(df2.index)
    # Cross-join and reset index:
    dfx = df1.join(df2).reset_index(drop=True)
    
    return dfx


def generate_label_df(df, agg_cols, count_cols):
    """
    Create a DataFrame in which, for each combination of values present 
    in `df` columns `agg_cols`, create one row for each of the possible 
    combination of all values in columns `count_cols`. That is: the 
    combination of values in `agg_cols` are only those seen in `df`, while 
    any possible combination of them and values in `count_cols` gain one 
    row.
    
    Parameters
    ----------
    df : DataFrame
        The DataFrame with columns `agg_cols` and `count_cols` 
        from which the possible values and combinations will be 
        built.
        
    agg_cols : list of str
        Names of the columns to be used to group the data.
        Only the observed combination of their values will 
        be kept in the output.
        
    count_cols : list of str
        Names of the columns for which all the combinations 
        of all available values will be shown in the output, 
        even if such combination is never seen in `df`. This 
        is repeated for each combination of the `agg_cols` 
        values.
        
    Returns
    -------
    label_df : DataFrame
        A dataframe with columns `agg_cols` and `count_cols`, 
        where the values of the first are combined to reproduced 
        the observed combinations in `df`, and the values of 
        the latter are combined (appear in the same row) in every 
        possible combination between them and with the `agg_cols`.
    """

    # Create DataFrame with columns representing the groups:
    label_df = df.groupby(agg_cols).size().reset_index()[agg_cols]
    label_df.index = pd.Index([0] * len(label_df))

    # Loop over categories to count:
    for col in count_cols:

        # Build series of unique values:
        col_values = df[col].unique()
        count_series = pd.Series(col_values, index=[0] * len(col_values))
        count_series.name = col

        # Join to table of labels:
        label_df = label_df.join(count_series)

    # Reset index:
    label_df = label_df.reset_index(drop=True)
    
    return label_df


def iskeyQ(df):
    """
    Return True if columns in `df` (DataFrame or Series)
    uniquely identifies a row, and False otherwise.
    """
    Q = len(df) == len(df.drop_duplicates())
    return Q


def str_join(df, delimiter=', '):
    """
    Join all columns in `df` (DataFrame) separating 
    them with a `delimiter` (str).
    """
    
    series = df[df.columns[0]].copy()
    for col in df.columns[1:]:
        series = series + delimiter + df[col]
    
    return series


def replicate_rows(df, n_replicas):
    """
    Replicate rows in a DataFrame.
    
    Parameters
    ----------
    df : DataFrame
        DataFrame containing the rows to be 
        replicated.
    n_replicas : Series, str or int.
        A Series with the same index as `df`
        that specifies how many times each 
        row in `df` is to be replicated. If 
        a str or int, the name of the column 
        in `df` specifying it.
    
    Returns
    -------
    replicated_df : DataFrame
        Same contents `df` but with replicated
        rows, i.e., multiple rows with the 
        same content. Rows whose `n_replicas` 
        are zero are not present.
    """
    
    # Standardize input:
    if type(n_replicas) in (str, int):
        n_replicas = df[n_replicas]
    assert (n_replicas.index == df.index).all()
    assert n_replicas.dtype == int
    n_name = n_replicas.name
    
    # Prepare index for joining 
    indices_df = n_replicas.loc[n_replicas > 0].reset_index()
    replicated_index = indices_df.apply(lambda row: [row['index']] * row[n_name], axis=1).explode(ignore_index=True)
    
    # Replicate rows:
    replicated_df = pd.DataFrame(index=replicated_index).join(df)
    
    return replicated_df


def std_string_series(series, case=None):
    """
    Remove unecessary whitespace and normalize characters in the strings in `series`.
    If `case` is 'upper' or 'lower', transform strings to upper or lower case as well.
    Returns a new Series.
    """

    assert case in (None, 'upper', 'lower'), "Parameter `case` accepts None, 'upper' or 'lower' only, got '{}'.".format(case)
    
    new_series = series.str.strip().str.normalize('NFKC').str.split().str.join(' ')
    if case == 'upper':
        new_series = new_series.str.upper()
    if case == 'lower':
        new_series = new_series.str.lower()

    return new_series
