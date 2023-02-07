#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Tue Oct  5 17:37:29 2021

@author: skems
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
import re

from .dataframes import crop_strings

#############################
### From old utils module ###
#############################

def bold(text):
    """
    Takes a string and returns it bold.
    """
    return '\033[1m'+text+'\033[0m'


def unique(series):
    """
    Takes a pandas series as input and print all unique values, separated by 
    a blue bar.
    """
    u = series.unique()
    try:
        print(bold(str(len(u)))+': '+'\033[1;34m | \033[0m'.join(sorted(u.astype(str))))
    except:
        print(bold(str(len(u)))+': '+'\033[1;34m | \033[0m'.join(sorted(u)))


def columns(df):
    """
    Print the number of columns and their names, separated by a blue bar.
    """
    unique(df.columns)

    
def mapUnique(df):
    """
    Takes a pandas dataframe and prints the unique values of all columns and 
    their numbers. If the number of unique values is greater than maxItems, 
    only print out a sample.  
    """
    python_version = int(sys.version.split('.')[0])
    maxItems = 20
    
    for c in df.columns.values:
        try:
            u = df[c].unique()
            n = len(u)
        except TypeError:
            u = np.array(['ERROR (probably unhashable type)'])
            n = 'Unknown'
        if python_version == 2:
            isStr = all([isinstance(ui, basestring) for ui in u])
        else:
            isStr = all([isinstance(ui, str) for ui in u])
        print('')
        print(bold(c+': ')+str(n)+' unique values.')
        
        if n == 'Unknown':
            n = 1
        if n <= maxItems:
            if isStr:
                try:
                    print(',  '.join(np.sort(u)))
                except:
                    print(',  '.join(np.sort(u.astype('unicode'))))
            else:
                try:
                    print(',  '.join(np.sort(u).astype('unicode')))
                except:
                    print(',  '.join(np.sort(u.astype('unicode'))))
        else:
            if isStr:
                try:
                    print(bold('(sample) ')+',  '.join(np.sort(np.random.choice(u,size=maxItems,replace=False))))
                except:
                    print(bold('(sample) ')+',  '.join(np.sort(np.random.choice(u.astype('unicode'),size=maxItems,replace=False))))
            else:
                try:
                    print(bold('(sample) ')+',  '.join(np.sort(np.random.choice(u,size=maxItems,replace=False)).astype('unicode')))
                except:
                    print(bold('(sample) ')+',  '.join(np.sort(np.random.choice(u.astype('unicode'),size=maxItems,replace=False))))


def checkMissing(df, only_missing=True, return_df=False):
    """
    Takes a pandas dataframe and prints out the columns that have 
    missing values.

    Parameters
    ----------
    df : DataFrame
        The data to check for missing values.
    only_missing : bool
        Whether to return only the columns with missing values
        or all the columns.
    return_df : bool
        Whether to return a DataFrame or to print the results.

    Returns
    -------
    final : DataFrame or None
        If `return_df` is True, return a DataFrame with missing 
        statistics for each column. Otherwise, return nothing. 
    """
    colNames = df.columns.values
    Ntotal = len(df)
    Nmiss  = np.array([float(len(df.loc[df[c].isnull()])) for c in colNames])
    df2    = pd.DataFrame(np.transpose([colNames,[df[c].isnull().any() for c in colNames], Nmiss, np.round(Nmiss/Ntotal*100,2)]),
                     columns=['coluna','missing','N','%'])

    if only_missing == True:
        final = df2.loc[df2['missing']==True][['coluna','N','%']]
    else:
        final = df2[['coluna','N','%']]

    if return_df == True:
        return final
    else:
        print(bold('Colunas com valores faltantes:'))
        print(final)


def one2oneQ(df, col1, col2):
    """
    Check if there is a one-to-one correspondence between two columns in a 
    dataframe.
    """
    n2in1 = df.groupby(col1)[col2].nunique()
    n1in2 = df.groupby(col2)[col1].nunique()
    if len(n2in1)==np.sum(n2in1) and len(n1in2)==np.sum(n1in2):
        return True
    else:
        return False


def one2oneViolations(df, colIndex, colMultiples):
    """
    Returns the unique values in colMultiples for a fixed value in colIndex 
    (only for when the number of unique values is >1).
    """
    return df.groupby(colIndex)[colMultiples].unique().loc[df.groupby(colIndex)[colMultiples].nunique()>1]


#################################
### New exploratory functions ###
#################################


def map_subcategories(df):
    """
    Print out the names of columns whose 
    values are sub-categories of other 
    columns (i.e. for each value considered
    a sub-category, there is only one value 
    associated to it in another column).    
    """
    cols   = df.columns
    print(bold('{:25} --> {}').format('Macro-categoria', 'Sub-categoria'))
    for col1 in cols:
        for col2 in cols:
            if col1 != col2:
                n_mixed_cats = len(one2oneViolations(df, col1, col2))
                if n_mixed_cats == 0:
                    print('{:25} --> {}'.format(col2, col1))


def drop_redundant_cols(df, prefix_drop='CD_', verbose=True):
    """
    Return a new DataFrame with columns that are biunivocal to 
    some other column dropped.
    
    Parameters
    ----------
    df : DataFrame
        DataFrame to be checked.
    prefix_drop : str or list of str
        When a pair of biunivocal columns
        is found, assign the preference 
        to be dropped to columns containing
        one of the provided prefixes.
    
    Returns
    -------
    dropped_df : DataFrame
        The same data as `df` but with the 
        selected biunivocal columns dropped.
    """

    drop_set = find_redundant_cols(df, prefix_drop, verbose)
    dropped_df = df.drop(drop_set, axis=1)
    return dropped_df


def find_redundant_cols(df, prefix_drop='CD_', verbose=True):
    """
    Find columns that are biunivocal to 
    some other column in a DataFrame.
    
    Parameters
    ----------
    df : DataFrame
        DataFrame to be checked.
    prefix_drop : str or list of str
        When a pair of biunivocal columns
        is found, assign the preference 
        to be dropped to columns containing
        one of the provided prefixes.
    
    Returns
    -------
    drop_set : set
        Set of column names that can be dropped 
        since they are biunivocal to some other
        column.
    """

    # Standardize input:
    if type(prefix_drop) == str:
        prefix_drop = [prefix_drop]
    # Turn prefixes into regex:
    prefix_drop = '|'.join(prefix_drop)
    
    # Pega parâmetros:
    cols = df.columns
    n_cols = len(cols)

    # Prepara conjuntos:
    drop_set = set()
    # Loop sobre pares de colunas:
    for i in range(n_cols):
        for j in range(i + 1, n_cols):
            col1 = cols[i]
            col2 = cols[j]

            # Verifica se existe correspondência:
            is_same = one2oneQ(df, col1, col2)
            if is_same:
                if verbose:
                    print('{} é outro nome para {}'.format(col1, col2), end=' >> ')
                # Se uma coluna tem o prefixo especificado, remove ela:
                regex_match = re.match(prefix_drop, col1)
                if regex_match is not None:
                    if verbose:
                        print('Encontrado prefixo {}: remover a coluna {}'.format(regex_match.group(), col1))
                        drop_set.add(col1)
                        #keep_set.add(col2)
                # Caso contrário, remove a outra:
                else:
                    if verbose:
                        print('Remover a coluna {}'.format(col2))                
                    drop_set.add(col2)
                    #keep_set.add(col1)
    
    return drop_set


def non_unique_cols(df, print_uniques=True, transpose=False):
    """
    Get the columns that are not single-valued.
    
    Parameters
    ----------
    df : DataFrame
        DataFrame to analyse the amount of unique
        values per column.
    print_uniques : bool
        Whether to display the single-valued 
        columns and their respective values.
    transpose : bool
        When displaying the information above as 
        a DataFrame, transpose it.

    Returns
    -------
    non_unique_cols : list
        List of names of columns that are not
        single-valued.
    """
    # Security checks:
    assert type(print_uniques) == bool
    assert type(transpose) == bool
    
    # Get number of unique values per column:
    nunique = df.nunique(dropna=False)
    
    # If requested, display the values of the unique columns:
    if print_uniques:
        print(bold('Colunas com um único valor:'))
        unique_cols = list(nunique.loc[nunique == 1].index)
        if transpose:
            display(df[unique_cols].drop_duplicates().transpose())
        else:
            display(df[unique_cols].drop_duplicates())

    # Return a list of columns whose values vary:
    non_unique_cols = list(nunique.loc[nunique > 1].index)
    return non_unique_cols


def counts_table(series, show=False, head=10):
    """
    Build a value counts DataFrame with counts and frequency,
    including missing values.
    
    Parameters
    ----------
    series : Series
        The data to count.
    show : bool
        Whether to display the counts table and return nothing 
        or to return the table and do not display it.
    head : int
        The amount of lines to display if `show` is True. If 
        `show` is False, this is ignored.
    
    Returns
    -------
    df : DataFrame or None
        If `show` is True, return None. Otherwise, return 
        the full counts table.
    """
    
    # Value counts:
    freq = series.value_counts(normalize=True, dropna=False) * 100
    counts = series.value_counts(normalize=False, dropna=False)
    counts_df = pd.DataFrame({'Quantidade': counts, 'Frequência (%)': freq})
    
    # Return or display:
    if show == True:
        display(counts_df.head(head))
    else:
        return counts_df

    
def map_counts(df, head=10):
    """
    Display one table per column containing the 
    `df` unique value counts and frequencies. 
    Only show the first `head` (int) most frequent 
    values, including missing values.
    
    """
    cols = df.columns
    for col in cols:
        print(bold('\n' + col))
        counts_table(df[col], show=True, head=head)    


def guess_data_type(series, max_mag=2, max_int=30, verbose=False):
    """
    Try to classify the series into categorical ('cat'), 
    linear numerical ('num') or log numerical ('log') 
    data types.
    
    Parameters
    ----------
    series : Series
        The data whose type should be classified.
    max_mag : float
        For numerical data types, the maximum 10-folds
        for positive data to be classified as 'num'
        (linear scale). Above that, the Series is 
        classified as 'log'.
    max_int : int
        Maximum number of unique integers allowed
        so the data is classified as categorical. 
        Above this value, integers are considered 
        numerical ('num' or 'log').
        
    Returns
    -------
    
    dtype : str
        Data type, one of 'cat', 'num' or 'log'.
    """
    
    test_series = series.astype(str)

    
    # Numerical: is date.
    if pd.api.types.is_datetime64_ns_dtype(series):
        if verbose:
            print("Identified dates: set as 'num'.")
        return 'num'
    
    # Categorical: contains a character that is not used by numbers.
    if test_series.str.contains('[^\d,.\-]').any():
        if verbose:
            print('Non-numerical characters found.')
        return 'cat'
    
    # Categorical: minus sign does not appear at the start:
    if (test_series.str.contains('-') & ~test_series.str.contains('^-[\d.,]+')).any():
        if verbose:
            print('Minus sign not at the start.')
        return 'cat'
    
    # From now on, series is likely a number.
    
    # Is float:
    non_null_floats = series.astype(float).loc[~series.isnull()]
    if not np.isclose(non_null_floats.astype(int).astype(float), non_null_floats).all():
        if verbose:
            print('Idenfified floats.')
        # log-scale number: it is positive and covers many magnitudes:
        if (non_null_floats > 0).all() and (np.log10(non_null_floats).max() - np.log10(non_null_floats).min() > max_mag):
            if verbose:
                print('It is positive and covers many magnitudes.')
            return 'log'
        # lin-scale number: negative OR few magnitudes:
        else:
            if verbose:
                print('It is negative or covers few magnitudes.')
            return 'num'
    
    # Is integer:
    else:

        # Categorical: some integer starts with 0.
        if test_series.str.contains('^0').any():
            if verbose:
                print('Some integers start with zero.')
            return 'cat'
        
        if verbose:
            print('Idenfified integer.')        
        # Numerical (at least for plotting purposes): a lot of different categories
        if series.nunique() > max_int:
            if verbose:
                print('A lot of different integers.')

            # log-scale number: it is positive and covers many magnitudes:
            if (non_null_floats > 0).all() and (np.log10(non_null_floats).max() - np.log10(non_null_floats).min() > max_mag):
                if verbose:
                    print('It is positive and covers many magnitudes.')
                return 'log'
            # lin-scale number: negative OR few magnitudes:
            else:
                if verbose:
                    print('It is negative or covers few magnitudes.')
                return 'num'
        
        # Assume categorical:
        else:
            if verbose:
                print('Few different integers.')
            return 'cat'


def guess_all_data_types(df, max_mag=2, max_int=30, return_dict=False, verbose=False):
    """
    Try to classify the each column in `df` (DataFrame) 
    into categorical ('cat'), linear numerical ('num') or log 
    numerical ('log') data types.
    
    Parameters
    ----------
    df : DataFrame
        The data whose types should be classified.
    max_mag : float
        For numerical data types, the maximum 10-folds
        for positive data to be classified as 'num'
        (linear scale). Above that, the column is 
        classified as 'log'.
    max_int : int
        Maximum number of unique integers allowed
        so the data is classified as categorical. 
        Above this value, integers are considered 
        numerical ('num' or 'log').
    return_dict : bool
        Whether to return a dict from column name
        to data type ot to return a list of data 
        types.

    Returns
    -------
    
    dtypes : list of str or dict
        Data types, each one of 'cat', 'num' or 'log'.
    """

    # Security checks:
    assert type(return_dict) == bool

    # Guess data types:
    dtype_list = [guess_data_type(df[col], max_mag, max_int, verbose) for col in df.columns]

    # Return list of data types if requested:
    if not return_dict:
        return dtype_list

    # Return dict of data types if requested:
    dtype_dict = dict(zip(df.columns, dtype_list))
    return dtype_dict


def find_double_entries(df, key_col, drop_counts=1, dropna=False):
    """
    Return the rows in `df` whose key values appear 
    multiple times.
    
    Parameters
    ----------
    df : DataFrame
        Table to check for repeated keys.
    key_col : str or int
        Column in `df` containing the key.
    drop_counts : int
        Number of times a key appears in 
        `df` column `key_col` above which 
        the correspondent rows are returned.
    dropna : bool
        Whether to ignore NA values in the 
        keys when counting them.
    
    Returns
    -------
    repetitions : DataFrame
        Rows from `df` whose keys are repeated
        more than `drop_counts` times.
    """
    
    # Count number of times each key appears:
    counts = df[key_col].value_counts(dropna=dropna)
    # Select keys that appear multiple times:
    sel_keys = counts.loc[counts > drop_counts].index
    # Select data given those keys:
    sel = df.loc[df[key_col].isin(sel_keys)].sort_values(key_col, ascending=False)
    
    return sel


##########################
### Plotting functions ###
##########################


def exist_plot_Q(no_plot=['0.0', '0.2', '0.4', '0.6', '0.8', '1.0']):
    """
    Check if a plot is already being made.
    
    Parameters
    ----------
    no_plot : list of str
        Default axes labels, returned by pl.[xy]ticks() when 
        there is no plot.
    
    Returns
    -------
    is_plot : bool
        Return True if a plot is already being made, and False
        otherwise.
    """ 
    
    # Get X tick labels:
    loc, texts = pl.xticks()
    labels = [t.get_text() for t in texts]
    # If label is not default, plot exists:
    if labels != no_plot:
        return True

    # Get Y tick labels:
    loc, texts = pl.yticks()
    labels = [t.get_text() for t in texts]
    # If label is not default, plot exists:
    if labels != no_plot:
        return True
    
    # All labels are default, so assume plot does not exist:
    return False


def plot_categorical_dist(series, max_cat=30, cat_slice=(0, 30), normalize=False, horizontal='auto', use_ticks=False, **kwargs):        
    """
    Plot a histogram (actually a bar plot) for instaces 
    of a categorical variable. 
    
    Parameters
    ----------
    series : Pandas Series
        The instances of a categorical variable. They can 
        be strings or numbers.
    max_cat : int
        Maximum number of categories to plot. Only the 
        most frequent categories will appear.
    cat_slice : tuple of ints
        Cropping range (inclusive, exclusive) for string 
        instances. Parts beyond these limits are removed
        when labeling the plot bars.
    normalize : bool
        Whether to plot should be normalized to the total
        number of instances.
    horizontal : bool or 'auto'
        Whether the bars should be horizontal or vertical.
        If 'auto', guess based on the length the the string
        categories, otherwise use vertical.
    kwargs : other
        Arguments for the plot.
    """
    
    counts = series.value_counts(ascending=False, normalize=normalize)
    
    # If categories are strings:
    try:
        counts.index = crop_strings(counts.index, cat_slice)
        # Decide bar direction:
        if horizontal == 'auto':
            if counts.index.str.len().max() > 10:
                horizontal = True
            else:
                horizontal = False
    # If categories are not strings:
    except AttributeError:
        if horizontal == 'auto':
            horizontal = False
        counts.index = counts.index.astype(str)
    
    # If there is a plot alread made:
    if exist_plot_Q():
        # Get the order of the categories in the plot:
        if horizontal is True:
            cat_pos, cat_text = pl.yticks()
            cat_text = cat_text[::-1] # If horizontal, flip to unflip afterwards.          
        else:
            cat_pos, cat_text = pl.xticks()
        # Extract labels:
        cat_labels = np.array([t.get_text() for t in cat_text])
        # Select counts to plot, in the defined oder (missing values are set to zero):
        counts = pd.DataFrame(index=cat_labels).join(counts).fillna(0).iloc[:,0]
    # If creating a new plot:
    else:
        # Limit the number of bars by removing least frequent categories:
        counts = counts.iloc[:max_cat]
        
    # Plot:
    if horizontal == True:
        counts.iloc[::-1].plot(kind='barh', **kwargs)
    else:
        counts.plot(kind='bar', **kwargs)
        
        
def multiple_dist_plots(df, dtypes=None, n_cols=5, new_fig=True, fig_width=25, subplot_height=5, normalize=False, max_cat=30, cat_slice=(0, 30), n_bins='auto', **kwargs):
    """
    Create one plot of the distribution of values for each column 
    in a DataFrame.
    
    Parameters
    ----------
    df : DataFrame
        Data for which to plot the distribution.
    dtypes : None or list of Union[str, None]
        Types of the data for each column in `df`. It can be
        'cat' (for categorical distribution), 'num' (for numerical
        distribution, a.k.a a histogram), 'log'(same as 'num' but 
        for the base-10 log of the values) or None (no plot for this 
        column). If `dtypes` itself is None, guess the data types.
    n_cols : int
        Number of columns in the grid of subplots.
    new_fig : bool
        Whether to create a new figure or not.
    fig_width : float
        Width of the figure containing all subplots.
    subplot_height : float
        Approximate height of each subplot.
    normalize : bool
        Whether to normalize the plots by the number of instances 
        in `df`. For histograms (i.e. columns with 'num' or 'log'
        dtypes), plot the density.
    max_cat : int
        Maximum number of categories to include in the categorical
        plots (the most frequent ones are shown).
    cat_slice : tuple of ints
        Crop window applied to the categories' labels when drawing 
        a categorical plot. The categories names are cropped to this
        range.
    n_bins : int, array-like of floats or 'auto'
        Number of bins to use in histograms (plots for 'num' or 'log'
        columns), or edges of the bins (if array). If 'auto', decide
        the binning internally in an independent way for each plot.
    kwargs : other
        Parameters passed to the plots.
    """

    if type(dtypes) == type(None):
        dtypes = guess_all_data_types(df)
        
    # Security checks:
    assert len(dtypes) == len(df.columns), '`dtypes` must contain one entry per `df` column.'
    assert set(dtypes) - {'cat', 'num', 'log', None} == set(), "`dtypes` can only contain the elements: 'cat', 'num', 'log' and None."
    
    # Count plots:
    n_plots = len(list(filter(lambda x: x != None, dtypes)))
    n_rows  = int(n_plots / n_cols) + 1

    if new_fig == True:
        pl.figure(figsize=(fig_width, subplot_height * n_rows))
    panel = 0
    for i in range(len(dtypes)): 
        # One column:
        dtype  = dtypes[i]
        col    = df.columns[i]
        series = df[col]
        
        if dtype != None:
            # One plot:
            panel += 1
            pl.subplot(n_rows, n_cols, panel)
            pl.title(col)
            
            # Categorical:
            if dtype == 'cat':
                plot_categorical_dist(series, max_cat, cat_slice, normalize, **kwargs)
            # Numerical:
            if dtype in ('num', 'log'):
                if n_bins == 'auto':
                    bins = min(min(200, series.nunique()), int(np.round(len(series) / 10)))
                else:
                    bins = n_bins
                if dtype == 'log':
                    np.log10(series).hist(bins=bins, density=normalize, **kwargs)
                    pl.xlabel('log')
                else:
                    series.hist(bins=bins, density=normalize, **kwargs)
            
    pl.tight_layout()
