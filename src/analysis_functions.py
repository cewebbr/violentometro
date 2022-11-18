#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Funções utilizadas nos notebooks de análise do projeto Violentômetro
Copyright (C) 2022  Henrique S. Xavier
Contact: hsxavier@gmail.com

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as pl
from glob import glob

import xavy.dataframes as xd
import xavy.tse as tse
import xavy.stats as xx
import xavy.utils as xu
import xavy.plots as xp


def build_bem_features(df, index_cols):
    """
    Aggregate candidate's wealth described in `df` (DataFrame, with TSE information)
    and returns a DataFrame with features as columns and
    'ANO_ELEICAO', 'SG_UF', 'SQ_CANDIDATO' as indices.
    """
    # Group data by candidate:
    grouped_by_cand = df.groupby(index_cols)

    # Build features DataFrame:
    bem_features = pd.DataFrame()
    bem_features['NUMERO_BENS']      = grouped_by_cand.size()
    bem_features['VALOR_TOTAL_BENS'] = grouped_by_cand['VR_BEM_CANDIDATO'].sum()
    
    return bem_features


def load_bem_build_features(path_pattern, index_cols=['ANO_ELEICAO', 'SG_UF', 'SQ_CANDIDATO']):
    """
    Loads all 'bem_candidato' files that match the `path_pattern` (str),
    concatenate them and build aggregated features for each candidate. 
    Returns a DataFrame with indices 'ANO_ELEICAO', 'SG_UF', 'SQ_CANDIDATO'
    and features as columns.
    """
    file_list = sorted(glob(path_pattern))
    df = pd.concat([build_bem_features(pd.read_csv(f), index_cols) for f in file_list])
    #df = pd.concat([pd.read_csv(f) for f in file_list])
    
    return df


def load_n_eleitores_by_ue(path, ue_col, qt_col='QT_ELEITORES_PERFIL'):
    """
    Load cleaned eleitorado data and aggregate it 
    to the unidade eleitoral level.
    
    Parameters
    ----------
    path : str
        Path to the already cleaned CSV file with
        eleitorado counts under each category.
    ue_col : str
        Name of the column that specifies the 
        unidade eleitoral (e.g. 'CD_MUNICIPIO' or
        'SG_UF').
    qt_col : str
        Name of the column with the number of 
        eleitores, to sum over.
    
    Returns
    -------
    series : Total number of eleitores in each 
    unidade eleitoral described in `ue_col`.
    """
    
    df = pd.read_csv(path, low_memory=False, usecols=[ue_col, qt_col])
    series = df.groupby(ue_col)[qt_col].sum()
    
    return series


def etl_votos_nominais(filename, turno=1, 
                       usecols=['CD_TIPO_ELEICAO', 'NR_TURNO', 'SQ_CANDIDATO', 'QT_VOTOS_NOMINAIS'], 
                       votos_col='QT_VOTOS_NOMINAIS'):
    """
    Load raw TSE data on votes on candidates from a CSV file and 
    aggregate them by candidates.
    
    Parameters
    ----------
    filename : str or Path
        Path to the TSE raw CSV file with prefix 
        'votacao_candidato_munzona'.
    turno : int
        Which round to select, either 1 or 2.
    usecols : list of str
        Columns to load from the file. It is required to include
        'SQ_CANDIDATO', 'NR_TURNO' and the column with the vote 
        counts. Other columns should be included if there are 
        security and consistency checks inside the function. 
        These are not output.
    votos_col : str
        Name of the column containing the vote counts, such as 
        'QT_VOTOS_NOMINAIS' or 'QT_VOTOS_NOMINAIS_VALIDOS'.
    
    Returns
    -------
    total_votos : Series
        Total number of votes of class given by `votos_col`
        on round `turno` per candidate, identified by 
        'SQ_CANDIDATO'.
    """
    
    # Load data:
    df = tse.load_tse_votacao_data(filename, usecols=usecols)

    # Security checks:
    assert len(df['CD_TIPO_ELEICAO'].unique()) == 1, 'Esperamos apenas um tipo de eleição, mas encontramos mais. Verifique!'

    # Select just one round:
    df = df.loc[df['NR_TURNO'] == turno]

    # Aggregate votes by candidate:
    total_votos = df.groupby('SQ_CANDIDATO')[votos_col].sum()
    
    return total_votos


def load_cand_eleitorado_bens_votos(cand_file, eleitorado_file, bens_file, votos_file,
                              cand_sel_cols=['SQ_CANDIDATO', 'NM_CANDIDATO', 'NM_URNA_CANDIDATO', 'NR_CPF_CANDIDATO', 'NR_TITULO_ELEITORAL_CANDIDATO', 
                                             'SG_PARTIDO', 'SG_UF', 'SG_UE', 'NM_UE', 'DS_CARGO', 'NM_SOCIAL_CANDIDATO', 'NR_IDADE_DATA_POSSE', 
                                             'DS_GENERO', 'DS_GRAU_INSTRUCAO', 'DS_COR_RACA'],
                              drop_duplicates=True):
    """
    Create a Table with the candidates' data, the number of voters in 
    their electoral unit, the number and value of their declared 
    wealth and the total number of votes received on the first round.
    
    Parameters
    ----------
    cand_file : str
        Path to the file 'consulta_cand', from TSE, already cleaned.
    eleitorado_file : str
        Path to the cleaned TSE file containing data about the 
        voters' profile in each electoral section.
    bens_file : str
        Path to the cleaned TSE file containing data about the candidates'
        declared wealth.
    votos_file : str
        Path to the raw TSE file containing the number of votes received by 
        each candidate at each municipality-electoral zone.
    cand_sel_Cols : list of str
        Columns in `cand_file` to keep.
    drop_duplicates : bool
        Whether to remove duplicated rows in `cand_file` (with 
        regard to the selected columns `cand_sel_cols`).
    
    Returns
    -------
    df : DataFrame
        Table with the candidates' data, the number of voters in 
        their electoral unit and the number and value of their declared 
        wealth.
    """
    
    # Carrega perfil dos candidatos:
    cand_df = pd.read_csv(cand_file, low_memory=False)
    # Remove registros aparentemente duplicados:
    if drop_duplicates is True:
        cand_df = cand_df[cand_sel_cols].drop_duplicates()
    assert xd.iskeyQ(cand_df[['SQ_CANDIDATO']])

    # Descobre se as eleições em questão são municipais ou gerais:
    if cand_df['SG_UE'].dtype == np.dtype('int64'):
        ue_col = 'CD_MUNICIPIO'
    else:
        ue_col = 'SG_UF'
        
    # Carrega número de eleitores na unidade eleitoral e junta nos dados dos candidatos:
    eleitores = load_n_eleitores_by_ue(eleitorado_file, ue_col)
    cand_df = cand_df.join(eleitores, on='SG_UE')
    assert xd.iskeyQ(cand_df[['SQ_CANDIDATO']])
    # Preenche núm. eleitores do Brasil:
    cand_df.loc[cand_df['SG_UE'] == 'BR', 'QT_ELEITORES_PERFIL'] = eleitores.sum()

    # Carrega bens dos candidatos e junta nos dados:
    bens_df = load_bem_build_features(bens_file, index_cols=['SQ_CANDIDATO'])
    cand_df = cand_df.join(bens_df, on='SQ_CANDIDATO')
    cand_df['NUMERO_BENS'].fillna(0, inplace=True)
    cand_df['VALOR_TOTAL_BENS'].fillna(0, inplace=True)
    
    # Carrega votos totais recebidos:
    votos_df = etl_votos_nominais(votos_file)
    cand_df = cand_df.join(votos_df, on='SQ_CANDIDATO')
    
    return cand_df


# If running this code as a script:
if __name__ == '__main__':
    pass


def regex_selector(series, regex, exists=True, case=True, verbose=False):
    """
    Check is a regular expression is present in each
    element of a Series.
    
    Parameters
    ----------
    series : Series of str
        The texts to look for the regular expression.
    regex : str
        The regular expression to look for.
    exists : bool
        If False, look for texts that do NOT contain
        `regex`. Otherwise, look for texts that do 
        contain it.
    case : bool
        Whether to do a case-sensitive search or not.
    verbose : bool
        If True, print the selection criteria used.
        
    Returns
    -------
    selector : Series
        Boolean series specifying if each text in 
        `series` obeys the specified selection or 
        not.
    """
    
    # Print criteria if requested:
    if verbose is True:
        if exists is False:
            print('Not ', end='')
        print("'{}'".format(regex), end='')
        if case is False:
            print(' (any case)')
        else:
            print('')
    
    # Create selector and return:
    selector = series.str.contains(regex, case=case)
    if exists is False:
        return ~selector
    else:
        return selector
    

def strip_col_names(df):
    """
    Remove surrounding whitespaces from the column
    names of a DataFrame `df`, in place.
    """
    cols = df.columns
    assert len(cols) == len(set(cols)), 'Existem colunas duplicadas'
    col_std = dict(zip(cols, cols.str.strip()))
    df.rename(col_std, axis=1, inplace=True)
    cols = df.columns
    assert len(cols) == len(set(cols)), 'Existem colunas duplicadas após limpeza'
    

def check_columns(df, expected_cols, expected_dtypes):
    """
    Check if `df` DataFrame columns are the 
    expected ones, and print the differences, 
    if any. Return False if any problem is 
    found, True otherwise.
    
    Parameters
    ----------
    df : DataFrame
        DataFrame whose columns should be 
        checked.
    expected_cols : list
        Expected names of the columns.
    expected_dtypes : 
    """

    # Security check:
    assert len(expected_cols) == len(expected_dtypes)
    
    # Starting variables:
    ok = True
    cols = df.columns
    expected_set = set(expected_cols)
    input_set = set(cols)

    # Colunas faltando:
    missing_set = expected_set - input_set
    if len(missing_set) > 0:
        ok = False
        print('!! As seguintes colunas estão faltando: {}'.format(missing_set))

    # Colunas novas:
    extra_set = input_set - expected_set
    if len(extra_set) > 0:
        ok = False
        print('!! As seguintes colunas não eram esperadas: {}'.format(extra_set))
    
    # Ordem das colunas:
    if input_set == expected_set:
        if (np.array(cols) != np.array(expected_cols)).any():
            ok = False
            print('!! A ordem das colunas está diferente.')
    
    # Tipo das colunas:
    if ok is True:
        expected_series = pd.Series(expected_dtypes, index=expected_cols)
        diff_dtype = (expected_series != df.dtypes)
        if diff_dtype.any() > 0:
            ok = False
            diff_cols = expected_series.loc[diff_dtype]
            print('!! Data type das colunas {} deveriam ser {} mas são {}.'.format(list(diff_cols.index), list(diff_cols.values), list(df.dtypes.loc[diff_dtype].values)))
            
    return ok


def add_platform_onehot(redes_df, redes_regex, website_regex, arroba_regex, url_col='DS_URL'):
    """
    Add dummy columns to DataFrame that 
    specify if a platform is mentioned,
    in place.
    
    Parameters
    ----------
    redes_df : DataFrame
        DataFrame containing a column named
        `url_col` that may contain a 
        reference to a platform present in 
        `redes_regex` keys.
    redes_regex : dict
        A dict from the platform name (str)
        to a regex (str) used to identify 
        the platform in the strings under 
        `url_col` column.
    website_regex : str
        Regex that identifies a website. Only
        rows that do not match any regexes from
        `redes_regex` are tested for a website.
    arroba_regex : str
        Regex that identifies a 'user-like' 
        entry that starts with @. Only rows 
        that do not match any regexes above are 
        tested for a website.        
    url_col : str
        Columns in `redes_df` that may 
        mention a platform.
    
    Returns
    -------
    redes_df : DataFrame
        The input DataFrame with extra
        columns (`redes_df` is modified).
    """
    
    # Prepare data:
    series = redes_df[url_col]
    series = series.str.strip()
    
    # Add dummy columns:
    for key in redes_regex.keys():
        redes_df[key] = regex_selector(series, redes_regex[key], case=False).astype(int)
    
    # Add column for website: 
    redes_df['website'] = 0
    is_website = regex_selector(series, website_regex, case=False)
    no_redes   = (redes_df[redes_regex.keys()] == 0).all(axis=1)
    redes_df.loc[is_website & no_redes, 'website'] = 1
    
    # Add column for @:
    redes_df['arroba'] = 0
    is_arroba  = regex_selector(series, arroba_regex, case=False)
    no_redes   = (redes_df[list(redes_regex.keys()) + ['website']] == 0).all(axis=1)
    redes_df.loc[is_arroba & no_redes, 'arroba'] = 1    
    
    # Remaining are users:
    redes_df['usuario'] = 0
    no_redes   = (redes_df[list(redes_regex.keys()) + ['website', 'arroba']] == 0).all(axis=1)
    redes_df.loc[no_redes, 'usuario'] = 1    

    
    return redes_df


def load_tse_social_data(path, expected_cols, expected_dtypes):
    """
    Load, clean and check TSE CSV file on 
    candidates' social media.
    
    Parameters
    ----------
    path : str
        Path to the TSE CSV file.
    expected_cols : list of str
        Names of the columns in the CSV file,
        to double check.
    expected_dtypes : list of numpy dtypes
        Data types of the columns in the CSV 
        file, to double check.
    
    Returns
    -------
    redes_df : DataFrame
        The input data with the column 
        names stripped from whitespaces.
    """
    
    # Load:
    redes_df = tse.load_tse_raw_data(path)
    # Clean:
    strip_col_names(redes_df)
    # Checks:
    assert check_columns(redes_df, expected_cols, expected_dtypes)
    return redes_df


def return_redes_regex():
    """
    Returns a dict (str -> str) from a web platform to 
    a regular expression used to identify this platform
    on the URL field on a TSE CSV file.
    """
    
    # Regexes for identifying web platforms:
    redes_regex = {'facebook': r'f[aeo]i?z?c+e{1,2}o? ?[cb]{1,2}[ou]{1,}c?k?|\bfc?bk?\b|\bface\b|fa[sc]?bo+k+|faceook|fca?ebook',
                   'twitter': 't[uw]{1,}[ei]{1,}t{1,}w?e{1,}r?|tweet',
                   'instagram': r'[ei][mn]?a?sa?t[sr]?a?[nr]?[qg][rl]e?[am][mn]?|\binsta?\b|(?!instala[cç][aã]o|instala[çc][õo]es|intant[âa]ne[ao]|instabio\.cc)insta|in?s?tragr?[ea][nm]?|is?na?tagra[nm]?',
                   'tiktok': 'tit?[ck][\- ]?to[kc]',
                   'youtube': r'y[ou]{1,} ?t[uo]be?|voutube|\byoutu\.be\b',
                   'linkedin': 'lin?k[ie]?n?d[ \-]?ie?n|linked',
                   'soundcloud': 'soundcloud',
                   'snapchat': 'snape?[ \-]?chat',
                   'spotify': 'spoti?fy',
                   'whatsapp': r'wh?as?t+h?[ \-]?[sz]+[ \-]?a+p+|whats|watshapp|zap[ \-:\(\)]+\d+|\d+[ \-:\(\)]+zap|\bwa+p*\.me\b|contate\.me',
                   'telegram': r'tele?gra[nm]|\bt\.me\b',
                   'signal': 'signal',
                   'parler': 'par\\.pw|parler?',
                   'badoo': 'badoo',
                   'tinder': 'tinder',
                   'flickr': 'flicke?r',
                   'grindr': 'grindr',
                   'kwai': r'\bkw\.?ai\b',
                   'twitch': 'twitch',
                   'skoob': r'\bskoob\b', 
                   'blogspot': r'blog?spot\b',
                   'wordpress':'wordpress', # Website
                   'wix':'wix', # Website
                   'democratize': r'financie\.de', # Agregador de links
                   'votolegal': 'votolegal',
                   'drive': r'drive\.go',
                   'anchor': 'anchor',
                   'linktree': r'link(?:tr)?\.ee[/.@?]',
                   'essentjus': 'essentjus',
                   'pinterest':r'pinterest|pin\.it',
                   'messenger': 'mes+enger|m\.me/',
                   'lkt': 'lkt(?:3\.com|\.bio)', # Agregador de links
                   'tse': 'tse\.jus\.br',
                   'castbox': 'castbox\.fm',
                   'issuu': 'issuu', # Website
                   'campanhadobem':'campanhadobem', # Financiamento
                   'campsite': 'campsite', # Agregador de links
                   'biolinky': 'biolinky', # Agregador de links
                   'skype': r'\bsk[iy]pe\b',
                   'lattes': r'\blattes\b',
                   'flow.page': 'flow\.page', # Agregador de links
                   'medium':'medium\.com',
                   'apoia.org':'apoia\.org', # Financiamento
                   'deezer': 'deezer',
                   'bit.ly': 'bit\.ly', # Encurtador
                   'tumblr': 'tumblr',
                   'periscope': 'pscp\.tv',
                   'queroapoiar': 'queroapoiar', # Financiamento
                   'cutt.ly': 'cutt\.ly',        # Encurtador
                   'telefone': '^[+\d\(\)\- ]{8,}$',
                   'patriabook': 'patriabook',
                   'abre.bio': r'abre\.bio', # Agregador de links
                   'gettr': 'gettr',
                   'beacons': r'beacons.ai', # Agregador de links
                   'vaquinhaeleitoral': 'vaquinhaeleitoral.com.br',
                   'linklist': 'linklist.bio',
                   'nenhuma': 'n[aã]o (?:tem|possui)|nenhum',
                   'email': '[\w\-\.]+@+(?:[\w\-]+[\.,])+[\w\-]{2,6}|^email|e[ -]mail'}
    
    return redes_regex


def etl_tse_redes_sociais(path, redes_regex, 
                          website_regex=r'ht{1,2}p{1,2}s?:/{0,3}(?:www\.)?(?:[\w\-]+\.)+[\w\-]{2,6}|www\.(?:[\w\-]+\.)+[\w\-]{2,6}|\.(?:com|org|net)$|\.(?:com|art)\.br', 
                          arroba_regex=r'^@|[ \-:/]@'):
    """
    Load TSE data on the candidates' social networks from a CSV file
    and parse the social networks into a one-hot encoding. Websites, 
    usernames and others are parsed into their own column.
    
    Parameters
    ----------
    path : str
        Path to TSE's raw CSV file, containing the social networks
        declared by the candidates.
    redes_regex : dict (str -> str)
        A mapping from the name of the web platform to the regular 
        expression used to identify it in the data.
    website_regex : str
        Regular expression used to identify personal websites. Other
        URLs identified by any expression in `redes_regex` are not 
        marked as a personal website.
    arroba_regex : str
        Regular expression representing usernames starting with '@', 
        but that do not specify the platform. Only rows that were not
        identified by the regexes above are classified with this one.
    
    Returns
    -------
    df : DataFrame
        The data from `path`, joined to binary columns specifying 
        whether the URL in each row contained a reference to each 
        platform specified in `redes_regex`. Personal websites, 
        usernames starting with '@' and others that do not match 
        any regex are mapped to their own columns.
    """
    
    # Hard-coded security data check:
    expected_cols   = ['DT_GERACAO', 'HH_GERACAO', 'ANO_ELEICAO', 'CD_TIPO_ELEICAO', 'NM_TIPO_ELEICAO', 'CD_ELEICAO', 'DS_ELEICAO', 'SQ_CANDIDATO', 'NR_ORDEM', 'DS_URL']
    expected_dtypes = [np.dtype('O'), np.dtype('O'), np.dtype('int64'), np.dtype('int64'), np.dtype('O'), np.dtype('int64'), np.dtype('O'), np.dtype('int64'), np.dtype('int64'), np.dtype('O')] 

    redes_df = load_tse_social_data(path, expected_cols, expected_dtypes)
    # Elimina redes duplicidade de redes:
    redes_df = redes_df.drop_duplicates(subset=['SQ_CANDIDATO', 'DS_URL'])

    # Classificando URLs:
    add_platform_onehot(redes_df, redes_regex, website_regex, arroba_regex)
    
    return redes_df


def platform_use_by_cand(cand_df, redes_df, redes_cols):
    """
    Build DataFrame about whether a candidate uses 
    a platform or not.
    
    Parameters
    ----------
    cand_df : DataFrame
        Information about each candidate (one 
        candidate per row), identified by 
        'SQ_CANDIDATO' column.
    redes_df : DataFrame
        Platforms used by the candidates, each
        row is identified by 'SQ_CANDIDATO' and
        'NR_ORDEM' (platform number).
    redes_cols : list of str
        Names of columns in `redes_df` that 
        contains the dummy variables specifying 
        if the platform in 'DS_URL' column is 
        the one in question.
        
    Returns
    -------
    use_df : DataFrame
        A table informing which platforms are 
        used by each candidate, one row per 
        candidate identified by 'SQ_CANDIDATO'.
    """
    
    # Junta info dos candidatos com redes declaradas (vários candidatos não declararam rede nenhuma):
    use_df = cand_df.join(redes_df.groupby('SQ_CANDIDATO')[redes_cols].sum().clip(upper=1), on='SQ_CANDIDATO', how='left')
    assert len(use_df) == len(cand_df)

    # Preenche quem não declarou com "nenhuma":
    use_df['nenhuma'].fillna(1, inplace=True)
    for col in filter(lambda s: s != 'nenhuma', redes_cols):
        use_df[col].fillna(0, inplace=True)

    # Usa números inteiros:
    use_df[redes_cols] = use_df[redes_cols].astype(int)
    
    return use_df


def plot_platform_counts(cand_per_platform, n_cand, election_name=None, fig=None, 
                         labelsize=12, barwidth=0.8, drop_zero=True, **kwargs):
    """
    Create a plot showing the fraction and number of candidates using each platform.
    
    Parameters
    ----------
    cand_per_platform : Series
        The number of candidates (values) that use each platform (index).
    n_cand : int
        The number of candidates, used to compute the fraction that uses each 
        platform.
    election_name : str
        Name of the election (e.g. 'Eleições gerais 2022'), used as part of 
        the plot title.
    fig : Figure or None
        Figure on which to add the plots. If None, create a new Figure.
    labelsize : float
        Font size of the tick labels.
    barwidth : float
        A value between 0 and 1 specifying the width of the bars.
    drop_zero : bool
        Whether to remove from the plot the platforms that have no usage at all.
    kwargs : Dict
        Keyword arguments for the plots.    
    
    Returns
    -------
    fig : figure
        The plot.
    """
    
    # Select only mentioned platforms, if requested:
    if drop_zero is True:
        toplot_series = cand_per_platform.loc[cand_per_platform > 0]
    else:
        toplot_series = cand_per_platform
    
    # Create figure is None provided:
    if fig is None:
        fig = pl.figure(figsize=(10,10))
    
    if election_name is not None:
        pl.suptitle('Redes sociais das candidaturas - {}'.format(election_name), fontsize=14)

    pl.subplot(1,2,1)
    (toplot_series / n_cand * 100).plot(kind='barh', width=barwidth, **kwargs)
    pl.xlabel('% das candidaturas\n(podem indicar mais de uma rede)', fontsize=labelsize)
    # Format:
    pl.grid(axis='x', color='lightgray', linewidth=1)
    pl.tick_params(labelsize=labelsize)
    ax = pl.gca()
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    pl.subplot(1,2,2)
    toplot_series.plot(kind='barh', width=barwidth, **kwargs)
    pl.xlabel('# de candidaturas\n(podem indicar mais de uma rede)', fontsize=labelsize)
    pl.xscale('log')
    # Format:
    pl.grid(axis='x', color='lightgray', linewidth=1)
    pl.tick_params(labelsize=labelsize)
    ax = pl.gca()
    ax.set_axisbelow(True)
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    pl.subplots_adjust(wspace=0.04, top=0.95)
    
    return fig


def values_to_bins(series, n_bins, lower_clip=None, upper_clip=None, logscale=False, pad_value=1e-6):
    """
    Build bin labels for each value in the provided Series.
    
    Parameters
    ----------
    series : Series
        Values to be binned.
    n_bins : int
        Number of bins (not edges).
    lower_clip : float or None
        Lower clipping bound for `series` when binning.
    upper_clip : float or None
        Upper clipping bound for `series` when binning.
    logscale : bool
        Whether to build equally-spaced bins in a linear or log scale.
    pad_value : float
        Positive value subtracted from the lower bound of the bins
        and added to the upper bound of the bins.
    
    Returns
    -------
    values_labels : Series
        The bin labels for each value in `series`. The index is the 
        same as in `series`.
    """
    
    # Clip series:
    series = series.loc[~series.isnull()]
    values = series.clip(lower=lower_clip, upper=upper_clip)

    # Build bin edges:
    max_values = values.max()
    min_values = values.min()   
    if logscale is True:
        values_bins = np.logspace(np.log10(min_values - pad_value), np.log10(max_values + pad_value), n_bins + 1)
    else:
        values_bins = np.linspace(min_values - pad_value, max_values + pad_value, n_bins + 1)

    # Retorna rótulo de bin para cada valor da série:
    #assert values_bins[1] > 10, 'Bin width are smaller than 10, this is bad for the rounding we perform.'
    values_labels = ((values_bins[1:] + values_bins[:-1]) / 2).astype(int)
    values_digit  = pd.cut(values, values_bins, labels=values_labels)
    # Refaz os rótulos com o valor médio em cada bin:
    #bins_means    = values.groupby(values_digit).mean().values
    #values_digit  = pd.cut(values, values_bins, labels=bins_means)

    assert values_digit.isnull().sum() == 0, 'Some value was not assign a bin.'
    
    return values_digit


def plot_bin_counts(series, n_bins, lower_clip=None, upper_clip=None, logscale=False, pad_value=1e-06):
    """
    Plot the number of instances per bin, given a binning strategy.
    
    Parameters
    ----------
    series : Series
        Values to be binned.
    n_bins : int
        Number of bins (not edges).
    lower_clip : float or None
        Lower clipping bound for `series` when binning.
    upper_clip : float or None
        Upper clipping bound for `series` when binning.
    logscale : bool
        Whether to build equally-spaced bins in a linear or log scale.
    pad_value : float
        Positive value subtracted from the lower bound of the bins
        and added to the upper bound of the bins.
    """
    
    # Build bins:
    values_digit = values_to_bins(series, n_bins, lower_clip, upper_clip, logscale, pad_value)
    s = values_digit.value_counts().sort_index()
    
    # Plot:
    pl.plot(s.index, s.values, marker='.')
    pl.xlabel(series.name)
    pl.ylabel('# candidaturas')
    pl.yscale('log')
    if logscale is True:
        pl.xscale('log')


def confidence_interval(prior_mode, n_trials, n_success, min_percentile=0.05, max_percentile=0.95, dp=0.0005): 
    """
    Return a confidence interval for the success probability of a 
    Bernoully trial given an observed number of successes in a 
    certain number of trials.
    
    Parameters
    ----------
    prior_mode : float
        The mode of the triangular distribution that describes the 
        prior probability for the success probability of a single trial.
    n_trials : int
        Number of trials of the independent binary experiment.
    n_success : int
        Number of times the independent experiment returned a positive 
        (success) result.
    min_percentile : float
        The cumulative posterior value of the left side of the confidence
        interval.
    max_percentile : float
        The cumulative posterior value of the right side of the confidence
        interval.        
    dp : float
        Interval between point used to compute the posterior.
    
    Returns
    -------
    p_min : float
        
    """
    
    # Compute posterior:
    p, post = xx.triang_binom_posterior(prior_mode, n_trials, n_success, dp)

    p_min = p[np.argmin((post.cumsum() * dp - min_percentile)**2)]
    p_max = p[np.argmin((post.cumsum() * dp - max_percentile)**2)]

    return p_min, p_max


def build_binomial_stats_df(use_df, grouper, platforms, min_percentile=0.05, max_percentile=0.95):
    """
    Build a DataFrame with the total number of candidates, the number
    and fraction using each platform and the confidence intervals 
    for the fractions.
    
    Parameters
    ----------
    use_df : DataFrame
        Table where each row is a candidate. Some columns are candidates' 
        characteristics, and others are binary variables that tell if the
        candidate uses the platform that names the column or not.
    grouper : str or Series
        How to group the candidates in order to compute the statistics in 
        each group. This can be the name of a `use_df` column or a Series 
        with a group label for each candidate.
    platforms : list of str
        Names of columns in `use_df` that show whether the candidates use 
        or not the respective platform.
    min_percentile : float
        The cumulative posterior value of the left side of the confidence
        interval computed for the fraction of candidates that uses the 
        platform.
    max_percentile : float
        The cumulative posterior value of the right side of the confidence
        interval computed for the fraction of candidates that uses the 
        platform.        

    Returns
    -------
    df : DataFrame
        The counts and fractions statistics for the candidates in each 
        group (row) and each platform (columns).
    """
    
    # Count, in each bin, the number of candidates that declared each platform:
    n_success = use_df.groupby(grouper)[platforms].sum()

    # Count the total number of candidates in each bin:
    n_trials = use_df.groupby(grouper).size()
    n_success['trials'] = n_trials

    # Compute the fraction of candidates that declared each platform:
    for col in platforms:
        n_success['freq_{}'.format(col)] = n_success[col] / n_success['trials']

    # Compute the confidence interval for the fraction:
    priors  = use_df[platforms].mean()
    for i, lim in enumerate(['min_', 'max_']):

        # Compute confidence interval limit:
        conf_df = pd.DataFrame(index=n_trials.index, columns=[lim + x for x in priors.index], dtype=float)
        for col in priors.index:
            for row in conf_df.index:
                conf_df.loc[row, lim + col] = confidence_interval(priors[col], n_trials[row], n_success.loc[row, col], min_percentile, max_percentile, 1e-4)[i]

        # Join limit to result:
        n_success = n_success.join(conf_df)
    
    return n_success


def plot_usage_frac_by_value(count_stats, xlabel, platforms=None, cmap_name='tab10', coffset=0, alpha=0.2, labelsize=14, logscale=False):
    """
    Plot the fraction of candidates that use each platform as a function 
    of a continuous value.
    
    Parameters
    ----------
    count_stats : DataFrame
        The counts and fractions statistics for the candidates in each 
        group (row) and each platform (columns).
    xlabel : str
        The name of the variable that groups the candidates, plotted on
        the x axis.
    platforms : list of str or None
        Names of the platforms to plot the fraction for. If not specified,
        plot for all platforms present in `count_stats`.
    cmap_name : str
        Color map name used to select the columns.
    coffset : int
        Skip the first `coffset` colors in `cmap_name`.
    alpha : float
        Transparency of the bands representing the confidence interval for the 
        fraction.
    labelsize : float
        Font size of the tick labels and axis names.
    logscale : bool
        Whether the x axis should use logarithmic scale or not.
    """
    
    # Select colors:
    cmap = pl.get_cmap(cmap_name)

    # Find out what platforms to plot, if not specified:
    if platforms is None:
        platform_end = np.argwhere(count_stats.columns == 'trials')[0][0]
        platforms = count_stats.columns[:platform_end]
    
    # Plot the platform usage fraction as a function of a variable:
    for i, plat in enumerate(platforms):
        pl.plot(count_stats.index, count_stats['freq_' + plat] * 100, label=plat, color=cmap(i + coffset))
        pl.fill_between(count_stats.index, count_stats['min_' + plat] * 100, count_stats['max_' + plat] * 100, alpha=alpha, color=cmap(i + coffset))
    # Set x scale:
    if logscale is True:
        pl.xscale('log')

    # Format:
    pl.ylabel('% dos candidatos', fontsize=labelsize)
    pl.xlabel(xlabel, fontsize=labelsize)
    pl.tick_params(labelsize=labelsize)
    ax = pl.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def plot_usage_frac_by_cat(count_stats, cmap_name='tab10', coffset=0, alpha=0.5, 
                           labelsize=14, horizontal=False):
    """
    Plot platform usage statistics for candidates split into classes.
    
    Parameters
    ----------
    count_stats : DataFrame
        Table containing the usage statistics for different groups 
        (rows) and platforms (columns).
    cmap_name : str
        Color map name used to select the columns.
    coffset : int
        Skip the first `coffset` colors in `cmap_name`.
    alpha : float
        Transparency of columns representing the observed used 
        fraction.
    labelsize : float
        Font size of the tick labels and axis names.
    horizontal : bool
        Whether the bars are horizontal or vertical.
    """
    
    # Preprocess for plot:
    y, err = stats_to_plot_input(count_stats)

    # Plot:
    cmap = pl.cm.get_cmap(cmap_name)
    colors = [cmap(i + coffset) for i in range(len(y.columns))]
    xp.multiple_bars_plot(y * 100, err=(err[0] * 100, err[1] * 100), alpha=alpha, 
                          colors=colors, horizontal=horizontal)
    # Format:
    pl.tick_params(labelsize=labelsize)
    pl.ylabel('% dos candidatos', fontsize=labelsize)
    ax = pl.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    
def plot_usage_frac_by_dimension(use_df, platforms, dimension, bins, 
                                 lower_clip=None, upper_clip=None, logscale=False, 
                                 pad_value=1e-06, coffset=0, xlabel=None, 
                                 labelsize=14,
                                 label_rot=0, legend_loc=None, legend_size=10,
                                 horizontal=False):
    """
    Plot the fraction of candidates that use each platform as a function 
    of a dimension.
    
    Parameters
    ----------
    use_df : DataFrame
        Table where each row is a candidate. Some columns are candidates' 
        characteristics, and others are binary variables that tell if the
        candidate uses the platform that names the column or not.
    platforms : list of str
        Names of columns in `use_df` that show whether the candidates use 
        or not the respective platform.
    dimension : str
        Name of a column in `use_df` to be used to split the candidates.
    bins : int, None or list of str
        Number of bins (not edges) in case of a numerical `dimension`. 
        If None or list of str, use the `dimension` values as is 
        (e.g. categorical). The list fo str is used to filter and order 
        the categories to plot.
    lower_clip : float or None
        Lower clipping bound for `series` when binning.
    upper_clip : float or None
        Upper clipping bound for `series` when binning.
    logscale : bool
        Whether to build equally-spaced bins in a linear or log scale.
        Only used if  `bins` is not None.
    pad_value : float
        Positive value subtracted from the lower bound of the bins
        and added to the upper bound of the bins.
    coffset : int
        Skip the first `coffset` colors in `cmap_name`.
    xlabel : str
        Label for the `dimension`, in the x-axis. If None, use 
        `dimension`.
    labelsize : float
        Font size of the axis labels.
    label_rot : float
        Rotation of the x-axis labels.
    legend_loc : str or None
        Where to place the plot legend.
    legend_size : float
        Legend font size.
    horizontal : bool
        Only used when plotting usage for categorical groups; whether the 
        bar plot is horizontal or vertical.
    """
    
    # NUMERICAL:
    if type(bins) == int:
        
        # Compute binning:
        series = use_df[dimension]
        values_digit = values_to_bins(series, bins, lower_clip, upper_clip, logscale, pad_value)

        # Compute fraction and confidence intervals per bin:
        count_stats = build_binomial_stats_df(use_df, values_digit, platforms)

        # Plot:
        if xlabel is None:
            xlabel = dimension
        plot_usage_frac_by_value(count_stats, xlabel, logscale=logscale, coffset=coffset,
                                 labelsize=labelsize)
    
    # CATEGORICAL:
    else:
        
        # Compute fraction and confidence intervals per class:
        count_stats = build_binomial_stats_df(use_df, dimension, platforms)
        
        # Select and order categories (if provided):
        if bins is not None:
            count_stats = count_stats.loc[bins]

        # Plot:
        plot_usage_frac_by_cat(count_stats, coffset=coffset, horizontal=horizontal,
                               labelsize=labelsize)

    # Finally:
    if label_rot > 0:
        ha = 'right'
    elif label_rot < 0:
        ha = 'left'
    else:
        ha = 'center'
    pl.xticks(rotation=label_rot, ha=ha)    
    pl.legend(loc=legend_loc, fontsize=legend_size)


def stats_to_plot_input(count_stats, mean_prefix='freq_', min_prefix='min_', max_prefix='max_'):
    """
    Reorganize platform usage statistics for a categorical plot 
    (multi columns plot with error bars).
    
    Parameters
    ----------
    count_stats : DataFrame
        Table containing the usage statistics for different groups 
        (rows) and platforms (columns).
    mean_prefix : str
        Prefix for the columns containing the fraction of candidates
        that declared that use each platform.
    min_prefix : str
        Prefix for the columns containing the lower bound for 
        the confidence interval of the fraction above.
    max_prefix : str
        Prefix for the columns containing the upper bound for 
        the confidence interval of the fraction above.
        
    Returns
    -------
    exp_p : DataFrame
        Table with the observed fraction of candidates using each platform
        (columns) in each group (rows).
    err : tuple (DataFrame, DataFrame)
        Tables with the lower and upper bounds of the confidence interval 
        of the fraction of candidates using each platform (columns) 
        in each group (rows).
    """
    
    # Split statistics into different DataFrames and rename columns to the same names (platform names):
    min_p = xd.rename_columns(count_stats[xu.select_by_substring(count_stats.columns, min_prefix)], min_prefix, '')
    max_p = xd.rename_columns(count_stats[xu.select_by_substring(count_stats.columns, max_prefix)], max_prefix, '')
    exp_p = xd.rename_columns(count_stats[xu.select_by_substring(count_stats.columns, mean_prefix)], mean_prefix, '')
    
    # Compute the upper and lower errors:
    err_low = exp_p - min_p
    err_up  = max_p - exp_p
    err = (err_low, err_up)
    
    return exp_p, err