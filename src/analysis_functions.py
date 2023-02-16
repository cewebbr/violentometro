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

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as pl
from glob import glob
from sklearn.feature_extraction.text import CountVectorizer
import scipy.stats as stats
from itertools import repeat
from multiprocessing import Pool

from .xavy import dataframes as xd
from .xavy import tse as tse
from .xavy import stats as xx
from .xavy import utils as xu
from .xavy import plots as xp
from .xavy import drive as dr
from .xavy.text import text2tag


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


def load_cand_eleitorado_bens_votos(cand_file, eleitorado_file, bens_file, votos_file, lgbt_sq_cand=None,
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
    lgbt_sq_cand : iterable or None
        List-like or set of 'SQ_CANDIDATO's that
        are listed on the VoteLGBT platform. If 
        None, this is ignored.
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
    
    if lgbt_sq_cand is not None:
        # Cria série que identifica se há cadastro no VoteLGBT:    
        lgbt_series = pd.Series('Sem cadastro', index=cand_df.index)
        lgbt_series.loc[cand_df['SQ_CANDIDATO'].isin(lgbt_sq_cand)] = 'Cadastro no VoteLGBT'
        lgbt_series.name = 'VOTE_LGBT'
        # Adiciona info do VoteLGBT:
        cand_df = cand_df.join(lgbt_series)

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
    use_redes = redes_df.groupby('SQ_CANDIDATO')[redes_cols].sum().clip(upper=1)
    use_df = cand_df.join(use_redes, on='SQ_CANDIDATO', how='left')
    assert len(use_df) == len(cand_df)

    # Preenche quem não declarou com "nenhuma":
    use_df['nenhuma'].fillna(1, inplace=True)
    for col in filter(lambda s: s != 'nenhuma', redes_cols):
        use_df[col].fillna(0, inplace=True)

    # Usa números inteiros:
    use_df[redes_cols] = use_df[redes_cols].astype(int)
    
    return use_df


def plot_platform_counts(cand_per_platform, n_cand, election_name=None, fig=None, 
                         labelsize=12, barwidth=0.8, drop_zero=True, lims=None,
                         **kwargs):
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
    lims : list of tuples or None
        Limits to the plot range in the x axis, for the fraction plot (on the
        left) and the absolute counts (on the right). Each limit can be replaced
        by None, in which case the range for the respective plot is not set.        
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
    if type(lims) is list:
        if lims[0] is not None:
            pl.xlim(lims[0])

    pl.subplot(1,2,2)
    toplot_series.plot(kind='barh', width=barwidth, **kwargs)
    pl.xlabel('Número de candidaturas\n(podem indicar mais de uma rede)', fontsize=labelsize)
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
    if type(lims) is list:
        if lims[1] is not None:
            pl.xlim(lims[1])
            
    pl.subplots_adjust(wspace=0.04, top=0.95)
    
    return fig


def clipped_values_to_edges(values, n_bins, logscale=False, pad_value=1e-6):
    """
    Build bin edges given the data and parameters provided.
    
    Parameters
    ----------
    values : values
        Values to be binned, already clipped to avoid outliers if any).
    n_bins : int
        Number of bins (not edges).
    logscale : bool
        Whether to build equally-spaced bins in a linear or log scale.
    pad_value : float
        Positive value subtracted from the lower bound of the bins
        and added to the upper bound of the bins.
    
    Returns
    -------
    values_bins : array
        The bin edges. There are `n_bins` + 1 entries.
    """
    
    # Build bin edges:
    max_values = values.max()
    min_values = values.min()   
    if logscale is True:
        values_bins = np.logspace(np.log10(min_values - pad_value), np.log10(max_values + pad_value), n_bins + 1)
    else:
        values_bins = np.linspace(min_values - pad_value, max_values + pad_value, n_bins + 1)

    return values_bins


def values_to_bins(series, n_bins, lower_clip=None, upper_clip=None, logscale=False, pad_value=1e-6):
    """
    Build bin labels for each non-null value in the provided Series.
    
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
        The bin labels for each non-null value in `series`. The index is the 
        same as in `series`.
    """
    
    # Clip series:
    series = series.loc[~series.isnull()]
    values = series.clip(lower=lower_clip, upper=upper_clip)

    # Build bin edges:
    values_bins = clipped_values_to_edges(values, n_bins, logscale, pad_value)
    
    # Retorna rótulo de bin para cada valor da série:
    #assert values_bins[1] > 10, 'Bin width are smaller than 10, this is bad for the rounding we perform.'
    values_labels = ((values_bins[1:] + values_bins[:-1]) / 2).astype(int)
    values_digit  = pd.cut(values, values_bins, labels=values_labels)
    # Refaz os rótulos com o valor médio em cada bin:
    #bins_means    = values.groupby(values_digit).mean().values
    #values_digit  = pd.cut(values, values_bins, labels=bins_means)

    assert values_digit.isnull().sum() == 0, 'Some value was not assign a bin.'
    
    return values_digit


def values_to_edges(series, n_bins, lower_clip=None, upper_clip=None, logscale=False, pad_value=1e-6):
    """
    Build bin edges given the data and parameters provided.
    
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
    values_bins : array
        The bin edges. There are `n_bins` + 1 entries.
    """
    
    # Clip series:
    series = series.loc[~series.isnull()]
    values = series.clip(lower=lower_clip, upper=upper_clip)

    # Build bin edges:
    values_bins = clipped_values_to_edges(values, n_bins, logscale, pad_value)
    
    return values_bins

    
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


def vocab_size(series):
    """
    Returns the vocabulary size of a corpus `series` (Series of str), 
    ignoring accents and cases.
    """
    
    vec = CountVectorizer(strip_accents='unicode', lowercase=True)
    vsize = len(vec.fit(series.fillna('')).get_feature_names_out())
    
    return vsize


def select_twitter_ids(ids_df, selector, verbose=True):
    """
    Return unique twitter IDs selected from the provided DataFrame.
    
    Parameters
    ----------
    ids_df : DataFrame
        A dataset containing the twitter IDs under the column 'id',
        among other columns.
    selector : Series
        Boolean series stating if each row in `ids_df` should be 
        selected.
    verbose : bool
        Whether to print the number of IDs selected.
    
    Returns
    -------
    twitter_ids : Series
        Unique (no duplicates) IDs from `ids_df`, selected according
        to `selector`.
    """
    
    twitter_ids = ids_df.loc[selector, 'id'].drop_duplicates()
    if verbose is True:
        print('Encontramos {} perfis nesse grupo.'.format(len(twitter_ids)))
    
    return twitter_ids


def mentions_to_pop(tweets_df, min_mentions, verbose=True):
    """
    Select tweets directed to popular users.
    
    Paramreters
    -----------
    tweets_df : DataFrame
        Tweets (one per line) mentioning some twitter user. The mentioned user ID 
        is given by the column 'batch_user'.
    min_mentions : int
        Minimum number of tweets in `tweets_df` mentioning a certain user required 
        for that user to be selected.
    verbose : bool
        Whether to print the number of mentioned users selected.
    
    Returns
    -------
    tweet_pop_df : DataFrame
        Slice of `tweets_df` containing all and only the tweets that mention the 
        users with the minimum number of mentions `min_mentions`.
    """
    
    # Count mentions per candidate:
    n_tweets_per_cand = tweets_df['batch_user'].value_counts()
    # Get candidates with minimum number of mentions:
    pop_cands = n_tweets_per_cand.loc[n_tweets_per_cand >= min_mentions].index.values
    
    if verbose is True:
        print('Encontramos {} perfis com no mínimo {} menções.'.format(len(pop_cands), min_mentions))
    
    # Select mentions to these candidates:
    tweet_pop_df = tweets_df.loc[tweets_df['batch_user'].isin(pop_cands)]
    
    return tweet_pop_df


def mentions_to_pop_in_group(tweets_df, ids_df, selector, min_mentions, verbose=True):
    """
    Select tweets that target popular candidates (i.e. with number of mentions above
    a threshold) belonging to the specified selection.
    
    Parameters
    ----------
    tweets_df : DataFrame
        Tweets (one per line) mentioning some twitter user. The mentioned user ID 
        is given by the column 'batch_user'.
    ids_df : DataFrame
        Table of candidate's twitter IDs (in column 'id') and their associated 
        social characteristics.
    selector : Series
        Boolean series with same length and index as `ids_df` stating if each row 
        in `ids_df` should be selected.
    min_mentions : int
        Minimum number of tweets in `tweets_df` mentioning a certain user required 
        for that user to be selected.
    verbose : bool
        Whether to print the number of IDs selected.
        
    Returns
    -------
    sel_tweets_df : DataFrame
        Slice of `tweets_df` containing all and only the tweets directed to candidates
        selected and with a mininum of mentions.
    """
    
    # Select tweets targeting the specified candidates:
    twitter_ids = select_twitter_ids(ids_df, selector, verbose)
    group_df = tweets_df.loc[tweets_df['batch_user'].isin(twitter_ids)]
    
    # Select only tweets mentioning popular candidates:
    sel_tweets_df = mentions_to_pop(group_df, min_mentions, verbose)
    
    return sel_tweets_df


def bin_prob(series, bin_width):
    """
    Returns the bin number for each entry in Series.
    
    Parameters
    ----------
    series : Series
        Values to be binned.
    bin_width : float
        Size of the bin.
    
    Returns
    -------
    A bin ID for each value in `series`. The edges of the bin `i` are given by:
    `i * bin_width` and `(i + 1) * bin_width`.
    """
    
    return (series / bin_width).astype(int)


def equal_prob_weights(series):
    """
    Return weights for sampling the given Series such that each unique 
    value is equally likely to get sampled.
    
    Parameters
    ----------
    series : Series
        Series of categorical variables that probably repeat.
    
    Returns
    -------
    df : DataFrame
        Table with the original categorical values, their counts and 
        respective weights.
    """
    
    counts = series.value_counts()
    df = counts.reset_index()
    df.rename({'index':series.name, series.name: 'n_' + series.name}, axis=1, inplace=True)
    df['w_' + series.name] = 1 / df['n_' + series.name]
    
    return df


def build_sampling_weight_df(tweets_df, dim1='batch_user', dim2='prob_bin', dim2_exp=1.0):
    """
    Create a DataFrame with weights for sampling the input so its marginal distributions 
    under the two specified dimensions are not extremely different from uniform.
    
    (The weight factoring anzatz we use leads to actually non-uniform distributions)
    
    Parameters
    ----------
    tweets_df : DataFrame
        Table whose rows are to be sampled.
    dim1 : str
        Name of column in `tweets_df` containing categorical values that are to be sampled 
        such that the likelyhood of each unique value is the same.
    dim2 : str
        Same as `dim2`, but for another column.
    dim2_exp : floatquanto
        Exponent applied to the inverse of `dim2` frequency. Values greater than 1.0 decrease
        the importance of `dim2` during sampling, and a value smaller than 1.0 increase it.

    Returns
    -------
    weight_df : DataFrame
        Table with a column 'w_eq_sampling' containing a weight for each combination of the unique 
        values in `dim1` and `dim2` columns in `tweets_df`.        
    """
    weight_df = xd.cross_join_dfs(equal_prob_weights(tweets_df[dim1]), equal_prob_weights(tweets_df[dim2]))
    weight_df['w_eq_sampling'] = weight_df['w_' + dim1] * weight_df['w_' + dim2] ** dim2_exp
    
    return weight_df


def sample_tweets_in_group(tweets_df, ids_df, selector, min_mentions, prob_bin, cand_exp, n_samples, random_state, verbose=True):
    """
    Sample tweets targeting popular candidates within a specified group, trying to 
    return similar frequencies for rates of directed violence and for different 
    candidates.
    
    Parameters
    ----------
    tweets_df : DataFrame
        Tweets (one per line) mentioning some twitter user. The mentioned user ID 
        is given by the column 'batch_user'.
    ids_df : DataFrame
        Table of candidate's twitter IDs (in column 'id') and their associated 
        social characteristics.
    selector : Series
        Boolean series with same length and index as `ids_df` stating if each row 
        in `ids_df` should be selected.
    min_mentions : int
        Minimum number of tweets in `tweets_df` mentioning a certain user required 
        for that user to be selected.
    prob_bin : float
        Size of the bin used for binning the directed violence score in order to 
        sample the tweets.
    cand_exp : float
        Exponent applied to the inverse of candidate frequency, when sampling. Values 
        greater than 1.0 decrease the importance of the candidate dimension during 
        sampling, and a value smaller than 1.0 increase it.        
    n_samples : int
        Number of tweet samples to return.
    random_state : int
        Seed for the sampling.
    verbose : bool
        Whether to print the number of IDs selected.
        
    Returns
    -------
    tweet_sample_df : DataFrame
        Sampled rows of `tweets_df` with added probability bin and weight columns.
        The sampling use weights to push the marginal distributions on the targeted
        candidates and on the directed violence rate close to uniform.
    """
    
    # Select popular candidates among the specified group:
    tweet_pool_df = mentions_to_pop_in_group(tweets_df, ids_df, selector, min_mentions, verbose)
    
    # Add sampling weights to the tweets:
    tweet_pool_df['prob_bin'] = bin_prob(tweet_pool_df['hate_score'], prob_bin)
    weights = build_sampling_weight_df(tweet_pool_df, dim2_exp=cand_exp).set_index(['batch_user', 'prob_bin'])[['n_prob_bin', 'n_batch_user', 'w_batch_user', 'w_prob_bin', 'w_eq_sampling']]
    tweet_pool_df = tweet_pool_df.join(weights, on=('batch_user', 'prob_bin'))

    # Sample the tweets:
    tweet_sample_df = tweet_pool_df.sample(n_samples, replace=False, weights='w_eq_sampling', random_state=random_state)
    
    return tweet_sample_df


def plot_tweet_sampling_diagnosis(tweet_sample_df):
    """
    Create diagnostic plots for the tweet sample.
    """
    
    pl.figure(figsize=(15,4))

    pl.subplot(1, 3, 1)
    tcounts = tweet_sample_df['prob_bin'].value_counts().sort_index()
    pl.bar(tcounts.index, tcounts.values)
    pl.xlabel('prob_bin')
    pl.ylabel('Núm. de tweets na amostra')

    pl.subplot(1, 3, 2)
    tcounts = tweet_sample_df['batch_user'].value_counts()
    tcounts.hist(bins=range(tcounts.max() + 2))
    pl.xlabel('Núm. de menções à candidatura')
    pl.ylabel('Núm. de candidatos')
    
    test_sampling = pd.DataFrame()
    test_sampling['counts']   = tweet_sample_df['batch_user'].value_counts()
    test_sampling['mentions'] = tweet_sample_df[['batch_user', 'n_batch_user']].drop_duplicates().set_index('batch_user')

    pl.subplot(1,3,3)
    pl.scatter(test_sampling['mentions'], test_sampling['counts'], alpha=0.2)
    pl.xscale('log')
    pl.xlabel('Núm. de tweets no pool')
    pl.ylabel('Núm. de tweets na amostra')

    pl.show()
    

def export_sample_for_annotation(tweet_sample_df, filename, specificity, verbose=True):
    """
    Create a CSV file with the input sample tweets, along with empty column 
    for manual annotation.
    
    Parameters
    ----------
    tweet_sample_df : DataFrame
        Sample tweets for annotation. Required columns are: 'id' (tweet ID),
        'text' (tweet content) and 'tweet_url' (link to the Tweet).
    filename : str
        Where to save the CSV file with the tweets. This might contain a '{}'
        that will be filled accordingly to `specificity`.
    specificity : str
        One on {'LGBTfóbico', 'controle', 'machista', 'racista'}. This names 
        the last column for annotation and changes the filename if '{}' is 
        in `filename`.
    verbose : bool
        Whether to print the destination file.
    """
    
    # Hard-coded:
    file_suffix = {'racista':'pessoas_negras', 'machista':'mulheres', 'LGBTfóbico':'lgbts', 'controle':'controle'}
    assert specificity in file_suffix.keys(), '`specificity` options are: {}'.format(set(file_suffix.keys()))
    sample_cols = ['id', 'text', 'tweet_url', 'n_prob_bin']

    # Create table:
    togo_df = tweet_sample_df[sample_cols].copy()
    togo_df['É violento?\n(1=sim, 0=não)'] = np.NaN
    togo_df['O candidato é o objeto do comentário?\n(1=sim, 0=não)'] = np.NaN
    togo_df['Comentário é racista/machista/LGBTfóbico?\n(1=sim, 0=não)'.format(specificity)] = np.NaN

    # Export
    outfile = filename.format(file_suffix[specificity])
    togo_df.to_csv(outfile, index=False)
    if verbose is True:
        print('Data saved to {}'.format(outfile))
        

def load_annotations_from_local_or_drive(row, path_template, force_drive=False, save_data=True):
    """
    Load tweet annotation table from a local file or from Google Sheets.
    
    Parameters
    ----------
    row : Series
        A list of informations about the annotation data to load, with keys:
        'Grupo', 'Anotador', 'Link'.
    path_template : str
        A string representing the path to the local file, with '{}' placeholders
        for the social group and the annotator, e.g.:
        '../dados/brutos/eletweet22/tweets_anotados_{}_{}.csv'
    force_drive : bool
        Whether to download data from Google Sheets even if the local file exists.    
    save_data : bool
        Wheter to save downloaded data to local file or not.]
    """
    
    # Define nome de arquivo onde salvar:
    filename = path_template.format(text2tag(row['Grupo']), row['Anotador'])
    annotation_df = dr.load_data_from_local_or_drive(row['Link'], filename, force_drive=force_drive, save_data=False)
    
    #annotation_df['id'] = annotation_df['id'].astype(int)
    annotation_df['id'] = annotation_df['tweet_url'].str.split('/').str.slice(-1).str.join('|').astype(int)
    
    if (os.path.isfile(filename) == False or force_drive == True) and save_data is True:
        print('Saving data to file...')
        annotation_df.to_csv(filename, index=False)
    
    return annotation_df


def prepare_one_group_annotation(df, annotator):
    """
    Get a raw tweet annotations from one annotator and prepare it for 
    joining with other annotations.
    """
    
    # Hard-coded:
    base = ['id', 'text', 'tweet_url']
    orig = ['É violento?\n(1=sim, 0=não)',
           'O candidato é o objeto do comentário?\n(1=sim, 0=não)',
           'Comentário é racista?\n(1=sim, 0=não)',
           'Comentário é machista?\n(1=sim, 0=não)',
           'Comentário é LGBTfóbico?\n(1=sim, 0=não)']
    to = ['violento', 'cand_objeto', 'racista', 'machista', 'lgbtfobico']
    
    # Verifica que todas as colunas esperadas existem:
    missing_cols = set(base + orig) - set(df.columns)
    assert missing_cols == set(), 'As seguintes colunas do anotador {} estão faltando: {}.'.format(annotator, missing_cols)
    
    # Renomeando colunas:
    to = [t + '_' + annotator for t in to]
    col_renamer = dict(zip(orig, to))
    df.rename(col_renamer, axis=1, inplace=True)
    
    # Seleciona colunas (evita colunas novas criadas pelos anotadores):
    df = df[base + to]
    
    # Sanity check:
    assert xd.iskeyQ(df[['id']]), "Existem tweets duplicados"

    # Coloca tweet ID como índice:
    df.set_index('id', inplace=True)
    
    return df


def etl_one_group_annotation(row, path_template, force_drive=False, save_data=True):
    """
    Load and clean tweet annotations given the metadata provided.
    
    Parameters
    ----------
    row : Series
        Metadata about the tweet annotations to load and clean. 
        Required fields are: 'Grupo', 'Anotador' e 'Link'.
    path_template : str
        A string representing the path to the local file, with '{}' placeholders
        for the social group and the annotator, e.g.:
        '../dados/brutos/eletweet22/tweets_anotados_{}_{}.csv'
    force_drive : bool
        Whether to download data from Google Sheets even if the local file exists.    
    save_data : bool
        Wheter to save downloaded data to local file or not.]

    Returns
    -------
    df : DataFrame
        Cleaned annotations loaded from the local CSV file or Google Sheets
        referred by the metadata provided at `row`.
    """
    
    # Load raw data from Sheets or from local file:
    df = load_annotations_from_local_or_drive(row, path_template, force_drive, save_data)
    # Prepare data:
    df = prepare_one_group_annotation(df, row['Anotador'])
    
    return df


def etl_group_annotations(dir_df, group, path_template, force_drive=False, save_data=True):
    """
    Load and join all annotations for the same tweets.
    
    Parameters
    ----------
    dir_df : DataFrame
        Table of Sheets that contain the annotations. Expected columns are:
        'Grupo', 'Anotador', 'id', 'Link', 'text', 'tweet_url'.
    group : str
        Nome do grupo para selecionar.
    path_template : str
        A string representing the path to the local file, with '{}' placeholders
        for the social group and the annotator, e.g.:
        '../dados/brutos/eletweet22/tweets_anotados_{}_{}.csv'
    force_drive : bool
        Whether to download data from Google Sheets even if the local file exists.    
    save_data : bool
        Wheter to save downloaded data to local file or not.
    
    Returns
    -------
    joined_df : DataFrame
        All annotations for `group` joined by tweet ID. 
    """
    
    # Hard-coded:
    check_cols = ['text', 'tweet_url']
    
    # Seleciona grupo social de candidatos:
    group_sheets_df = dir_df.query('Grupo == "{}"'.format(group))

    # Pega o primeiro conjunto de anotações:
    row = group_sheets_df.iloc[0]
    grupo = row['Grupo']
    joined_df = etl_one_group_annotation(row, path_template, force_drive=force_drive, save_data=save_data)

    # Loop sobre os demais conjuntos de anotações:
    for i in range(1, len(group_sheets_df)):

        # Pega o primeiro conjunto de anotações:
        row = group_sheets_df.iloc[i]
        assert row['Grupo'] == grupo, 'A lista de sheets fornecida não correpondem a um único grupo social.'
        extra_df = etl_one_group_annotation(row, path_template, force_drive=force_drive, save_data=save_data)

        # Verifica que os dados tratam dos mesmos tweet IDs:}
        assert set(joined_df.index) == set(extra_df.index), 'O conjunto de IDs dos tweets do anotador {} é diferente dos do anotador anterior'.format(row['Anotador'])

        # Cria uma tabela de teste de junção:
        test_df = joined_df[check_cols].join(extra_df[check_cols], how='outer', lsuffix='_A', rsuffix='_B')

        # Verifica se a junção das tabelas resulta nas propriedades esperadas:
        assert len(test_df) == len(joined_df), 'Junção das tabelas de anotações aumentou o número de linhas.'
        assert test_df.isnull().any().any() == False, 'Algum texto ou URL de tweet está faltando.'

        # Verifica se os conteúdos dos tweets são os mesmos nas duas tabelas:
        for col in check_cols:
            assert (test_df[col + '_A'] == test_df[col + '_B']).all(), 'As duas colunas de {} têm conteúdos diferentes.'.format(col)

        # Junta anotações numa tabela só:
        joined_df = joined_df.join(extra_df.drop(check_cols, axis=1))

        # Padroniza valores faltantes:
        joined_df.replace('', np.NaN, inplace=True)
        
    return joined_df.drop(check_cols, axis=1)


def find_cols(df, substr):
    """
    Return a list of `df` (DataFrame) columns that contain `substr` (str)
    as a substring.
    """
    return list(filter(lambda s: s.find(substr) != -1, df.columns))


def classification_mode(df, col, default, annotator_tag='_A'):
    """
    Do a majority summary of the annotations.
    
    Parameters
    ----------
    df : DataFrame
        Table of annotated tweets. The relevant columns should contain
        only 0s and 1s. Each annotator gives its classification in a 
        separate column.
    col : str
        Substring present in the columns to summarize. The classification
        given in these columns enter in a majority voting.
    default : 1 or 0
        What final classification to use in case the annotator's classes
        enter a tie.
    annotator_tag : str
        Suffix appended to `col` when looking for columns to summarize the
        annotations. This is to avoid columns containing already summarized
        information.
    Returns
    -------
    final_mode : Series
        Series containing the mode of the classifications given in the 
        columns with substring given by `col`. Ties are set to `default`.
    """
    
    # Security check:
    assert default in {0, 1}, "`default` pode ser 0 ou 1, mas encontrei o valor '{}'.".format(default)
    
    # Seleciona as colunas:
    cols = find_cols(df, col + annotator_tag)
    assert len(cols) >= 1, "Nenhuma coluna com a substring '{}' foi encontrada.".format(col) 
    # Data quality check:
    assert df[cols].isin([0, 1, np.NaN]).all().all(), "Encontrei algum valor nas colunas '{}' diferente de 0 ou 1.".format(col)
    
    # Pega os valores mais frequentes por linha (mais de um em caso de empate):
    modes = df[cols].mode(axis=1)
    assert len(modes.columns) <= 2

    # Para lidar com casos de empate (mais de uma moda):
    if default == 0:
        final_mode = modes.min(axis=1)
    else:
        final_mode = modes.max(axis=1)
    # Rename series:
    final_mode.name = col + '_final'

    return final_mode


def n_classifications(df, col, annotator_tag='_A'):
    """
    Count the number of classifications given by annotators.
    
    Parameters
    ----------
    df : DataFrame
        Table of annotated tweets. The relevant columns should contain
        only 0s and 1s. Each annotator gives its classification in a 
        separate column.
    col : str
        Substring present in the columns to check for classifications. 
    annotator_tag : str
        Suffix appended to `col` when looking for columns to summarize the
        annotations. This is to avoid columns containing already summarized
        information.
    
    Returns
    -------
    final_mode : Series
        Series containing the number of classifications given in the 
        columns with substring given by `col`.
    """
        
    # Seleciona as colunas:
    cols = find_cols(df, col + annotator_tag)
    assert len(cols) >= 1, "Nenhuma coluna com a substring '{}' foi encontrada.".format(col) 
    # Data quality check:
    assert df[cols].isin([0, 1, np.NaN]).all().all(), "Encontrei algum valor nas colunas '{}' diferente de 0 ou 1.".format(col)
    
    # Pega os valores mais frequentes por linha (mais de um em caso de empate):
    n = (~df[cols].isna()).sum(axis=1)
    n.name = 'n_' + col

    return n


def summarize_annotations(df, ignore_annotators=[], labels=['violento', 'cand_objeto', 'racista', 'machista', 'lgbtfobico'], defaults=[0, 1, 0, 0, 0]):
    """
    For each annotation label, add a column counting the number of annotations 
    made and their majority voting result.
    
    Parameters
    ----------
    df : DataFrame
        Table of annotated tweets. The annotation columns should contain
        only 0s and 1s. Each annotator gives its classification for a given 
        label in a separate column.
    ignore_annotators : list of str
        IDs of the annotators (e.g. 'A13') to be ignored when summarizing the
        data.
    labels : list of str
        The categorizations given to each tweet. Each category is binary and
        can be annotated by more than one annotator. All `df` columns 
        containing a given label in `labels` as a substring are considered 
        as annotations for the same category, given by different annotators.
    defaults : list of 0s and 1s.
        When getting the majority vote for a given category, these are the 
        values used in case of a tie. The values should be aligned with the 
        respective categories given in `labels`.
    
    Returns
    -------
    mod_df : DataFrame
        The input DataFrame (which is modified in place), with added columns,
        for each category in `labels`:
        - The final classification, obtained through majority voting;
        - The number of annotations made.
    """
    
    # Encontra colunas dos anotadores a serem ignorados:
    ignore_cols = []
    for a in ignore_annotators:
        ignore_cols += find_cols(df, a)
    
    # Loop sobre rótulos:
    for col, d in zip(labels, defaults):
        # Pega a classificação mais dada pelos anotadores:
        mode = classification_mode(df.drop(ignore_cols, axis=1), col, d)
        # Contabiliza número de classificações:
        n = n_classifications(df.drop(ignore_cols, axis=1), col)        
        # Add info to DataFrame:
        df[col + '_final'] = mode
        df['n_' + col] = n
        
    return df


def security_checks(df):
    """
    Make tests on the annotated tweets in `df` (DataFrame) and print 
    messages in case of problems found. Returns False if a problem 
    is found, and True otherwise.
    """
    
    status = True
    
    # Os tweets foram anotados de forma completa, sem informações faltando:
    annotation_counts = df[find_cols(df, 'n_')]
    same_n_annotations = annotation_counts.eq(annotation_counts.iloc[:, 0], axis=0).all(axis=1)
    diff_n_annotations = ~same_n_annotations
    if diff_n_annotations.sum() > 0:
        status = False
        print("!! Os seguintes tweets não possuem o mesmo número de anotações em todas as categorias: {}".format(set(diff_n_annotations.loc[diff_n_annotations].index)))
        
    return status


def load_group_data(filename, usecols=['id', 'n_prob_bin'], prefix='amostra_tweets_para_anotacao_'):
    """
    Load DataFrame from CSV file and add column with the social 
    group identified by the filename suffix.
    
    Parameters
    ----------
    filename : str
        Path to the CSV file to open.
    usecols : list of str
        Columns in the CSV file to load.
    prefix : str
        String that appears before the suffix that identifies the 
        social group.
        
    Returns
    -------
    df : DataFrame
        The columns `usecols` in the CSV file, joined with a constant
        column identifying the social group referenced by the 
        `filename`.
    """
    
    df = pd.read_csv(filename, usecols=usecols)
    df['grupo'] = filename.split(prefix)[-1].split('.')[0]
    
    return df


def count_annotations_made(tweets_annotated_df, annotation_regex=r'_A\d{1,2}$'):
    """
    Count the number of annotations made by each annotator.
    
    Parameters
    ----------
    tweets_annotated_df : DataFrame
        Tweets (one per row) with all annotations made (certain columns).
    annotation_regex : str
        Regular expression used to identify the columns in 
        `tweets_annotated_df` that contain the annotations.
    
    Returns
    -------
    n_annotations_df : DataFrame
        Columns are the labels (questions), index are the annotators.
        The values are the number of annotations made, which can be
        0 or 1.
    """
    
    # Encontra colunas com as anotações:
    annotation_cols  = tweets_annotated_df.columns[tweets_annotated_df.columns.str.contains(annotation_regex)]
    
    # Contabiliza as anotações:
    n_annotations = (~tweets_annotated_df[annotation_cols].isnull()).sum()
    n_annotations.name = 'n_anotacoes'
    n_annotations = n_annotations.reset_index()

    # Separa a questão do anotador:
    categories_df = pd.DataFrame(n_annotations['index'].str.split('_A').tolist(), index=n_annotations.index, columns=['label', 'anotador'])
    categories_df['anotador'] = 'A' + categories_df['anotador']
    
    # Junta a contagem:
    categories_df = categories_df.join(n_annotations['n_anotacoes'])
    
    # Coloca anotadores nas linhas e questões nas colunas:
    n_annotations_df = categories_df.pivot(index='anotador', columns='label', values='n_anotacoes')
    
    return n_annotations_df


def normal(x, m, s):
    """
    A normal distribution, not normalized (maximum is 1, always).
    """
    #n = 1.0 / s / np.sqrt(2 * np.pi)
    return np.exp(-(x - m)**2 / 2 / s**2)
    
def lognormal(x, m, s):
    """
    Lognormal distribution.
    """
    y = np.where(x <=0, 0, 1.0 / x / s / np.sqrt(2 * np.pi) * np.exp(-(np.log(x) - m)**2 / 2 / s**2))
    return y

def shifted_lognormal(x, m, s, p):
    mode = np.exp(m - s**2)
    y = lognormal(x - p + mode, m, s)
    return y

def alpha_limiter(x, y, f, e):
    m = (1 - x) / 2
    s = f * (1.2 - np.abs(x)) / 2
    return np.where(s < e, 0, normal(y, m, s))

def hate_score_bias_prior(a, b):
    amod = alpha_limiter(b, a, 1.0, 0.02)
    bmod = shifted_lognormal(-b, -0.7, 0.9, -1)
    return amod * bmod


def annotations_to_stats(tweets_df, annotation_col='violento', final_suffix='_final', machine_col='hate_score', bin_size=0.1):
    """
    Go from a table of annotated tweets (one per row) to a table of binomial statistics, 
    one row per bin of the provided score. 
    
    Parameters
    ----------
    tweets_df : DataFrame
        Table of tweets and their annotations, along with other information.
    annotation_col : str
        Substring that specifies the question that was answered by annotators.
    final_suffix : str
        Suffix that identifies, together with `annotation_col`, the colum containing 
        the tweet annotation {0,1}.
    machine_col : str
        Name of the column containing the scores used to bin the tweets.
    bin_size : float
        Width of the bin. They cover the range from 0 to 1.
    
    Returns
    -------
    Table of binomial statistics of the tweets.
    The output columns are:
    
    - Number of tweets annotated as 1, in a binary {0,1} classification;
    - Number of tweets in the bin;
    - Fraction of tweets marked as 1;
    - 05-percentile of the posterior distribution for the success rate;
    - 95-percentile of the posterior distribution for the success rate;
    - Average score value in the bin.
    """
    
    # Seleciona linhas com dados anotados:
    with_data_df = tweets_df.query('n_{} > 0'.format(annotation_col))

    # Bin data by machine score:
    bin_id = bin_prob(with_data_df[machine_col], bin_size)
    bin_id.name = machine_col + '_bin'

    # Compute avg. machine score in each bin:
    machine_mean = with_data_df.groupby(bin_id)[machine_col].mean()

    # Build binomial statistics DataFrame:
    stats_df = build_binomial_stats_df(with_data_df, bin_id, [annotation_col + final_suffix])
    stats_df['machine_avg_' + machine_col] = machine_mean
    
    return stats_df


def sample_fixed_proportion(df, label_col, frac_pos, replace=True, random_state=None):
    """
    Resample the positive class (label 1) so its fraction is the requested
    one, assuming the other class has label 0.
    
    Parameters
    ----------
    df : DataFrame
        Data to be sampled. It must contain the `label_col` column.
    label_col : str
        Name of the column containing the binary labels 0 or 1.
    frac_pos : float
        Fraction that the positive instances (label 1) must comprise
        in the final sample.
    replace : bool
        Whether to do sampling with replacement or not
    random_state : int or None
        Seed for the pseudo random number generator.

    Returns
    -------
    sample_df : DataFrame
        A table with all negative instances from `df` with a sample of 
        positive instances appended to its end.
     """
    
    # Separate classes:
    neg_df = df.loc[df[label_col] == 0]
    pos_df = df.loc[df[label_col] == 1]
    
    # Count the number of instances in each class: 
    n_neg = len(neg_df)
    n_pos = int(n_neg * frac_pos / (1 - frac_pos) + 0.5)
    
    # Build sample:
    sample_df = pd.concat([neg_df, pos_df.sample(n_pos, replace=replace, random_state=random_state)])
    
    return sample_df


def fixed_proportion_scores(df, true_col, prob_col, frac_pos, scorer, threshold=0.5, n_samples=200, replace=True, random_state=None):
    """
    Compute a metric score over multiple resampled datasets in which 
    the fraction of positive instances (in a binary classification)
    is the given one.
    
    Parameters
    ----------
    df : DataFrame
        Data to be sampled. It must contain the `true_col` and 
        `prob_col` columns.
    true_col : str
        Name of the column containing the true binary labels 0 or 1.
    prob_col : str
        Name of the column containing the probability that each 
        instance is positive.
    frac_pos : float
        Fraction that the positive instances (label 1) must comprise
        in the final samples used to compute the scores.
    scorer : callable
        Scorer like `accuracy_score` with parameters (y_true, y_pred).
    threshold : float
        Probability threshold for the values under the `prob_col` 
        columns above which the instance is considered as positive.
    n_samples : int
        Number of resampled datasets (and derived scores) to produce. 
    replace : bool
        Whether to do sampling with replacement or not
    random_state : int or None
        Seed for the pseudo random number generator.

    Returns
    -------
    scores : array
        Scores computed from the `n_samples` resampled datasets 
        containing all negative instances from `df` and the 
        sampled positive instances.
    """    
    
    scores = []
    seed = random_state
    
    for i in range(n_samples):
        
        if random_state is not None:
            seed = random_state + i
            
        sample_df = sample_fixed_proportion(df, true_col, frac_pos, replace=replace, random_state=seed)
        scores.append(scorer(sample_df[true_col], sample_df[prob_col] > threshold))
    
    return np.array(scores)


def multi_binomial_likelihood(intercept, coef, n_successes, n_trials, x):
    """
    Compute the likelihood (probability) of observing a given set of successes
    in multiple independent binomial experiments where the single trial success
    rate in each experiment is given by a linear function of an independent 
    variable.
    
    Parameters
    ----------
    intercept : float
        The intercept of the linear relation between the single trial success
        rate and the independent variable `x`.
    coef : float
        The factor that multiplies the independent variable `x` to return the 
        single trial success rate.
    n_successes : array, shape (m,)
        Number of successes in each of the `m` independent binomial experiments.
    n_trials : array, shape (m,)
        Number of trials in each of the `m` independent binomial experiments.
    x : array, shape (m,)
        Independent variable that specifies the success probability through a 
        linear function.
        
    Returns
    -------
    
    L : float
        Probability that the given result is observed, assuming all 
        experiments are independent.
    """
    
    ps = intercept + coef * x 
    return stats.binom.pmf(n_successes, n_trials, ps).prod()

def get_2d_max(xx, yy, zz):
    """
    Return the coordinates of the maximum value of a matrix.
    
    Parameters
    ----------
    zz : array, shape (n, m)
        Values where to look for the maximum.
    xx : array, shape (n, m)
        X coordinates of the values in `zz`.
    yy : array, shape (n, m)
        Y coordinates of the values in `zz`.
    
    Returns
    -------
    x_max : float
        Value from `xx` in the same position as the maximum value in `zz`.
    y_max : float
        Value from `yy` in the same position as the maximum value in `zz`.
    """
    
    i_max = zz.ravel().argmax()
    x_max = xx.ravel()[i_max]
    y_max = yy.ravel()[i_max]
    
    return x_max, y_max

def map_multi_binomial_posterior(stats_df, amin, amax, da, bmin, bmax, db, n_success_col='violento_final'):
    """
    Map the Posterior probability distrbution for linear coefficients that
    relate the IA hate score to the fraction of tweets annotated as violent
    by humans.
    
    Parameters
    ----------
    stats_df : DataFrame
        Table with statistical information about the number of tweets 
        considered violent by humans, in bins of IA hate scores (each
        bin is a row). The required columns are: 'violento_final' (number of 
        tweets considered violent), 'trials' (total number of tweets in the
        bin) and 'machine_avg_hate_score' (average of the hate scores in 
        the bin).
    amin : float
        Minimum value of the intercept in the linear relation between 
        human and IA evaluation.
    amax : float
        Maximum value of the intercept in the linear relation between 
        human and IA evaluation.
    da : float
        Interval between each point in the intercept range.
    bmin : float
        Minimum value of the angular coefficient in the linear relation 
        between human and IA evaluation.
    bmax : float
        Maximum value of the angular coefficient in the linear relation 
        between human and IA evaluation.
    db : float
        Interval between each point in the angular coefficient range.
    
    Returns
    -------
    aa : array, shape (n, m)
        Values of the intercept in each point of the 2D map of the 
        posterior.
    bb : array, shape (n, m)
        Values of the angular coefficient in each point of the 2D map 
        of the posterior.
    Post : array, shape (n, m)
        Values of the posterior.
    """
    
    # Build grid of coordinates:
    aa, bb = np.mgrid[amin:amax:da, bmin:bmax:db]
    
    # Map prior:
    P = hate_score_bias_prior(aa, bb)
    
    # Map likelihood:
    with Pool() as pool:
        L = np.nan_to_num(pool.starmap(multi_binomial_likelihood, zip(aa.ravel(), bb.ravel(), repeat(stats_df[n_success_col]), repeat(stats_df['trials']), repeat(stats_df['machine_avg_hate_score']))))
    L = L.reshape(aa.shape)
    
    # Map posterior:
    Post = P * L
    Post = Post / (Post.sum() * da * db)
    
    return aa, bb, Post

def find_pdf_levels(pdf, dx, dy, conf_intervals=[0.68, 0.95]):
    """
    Find the values of a 2D PDF that correspond to the given confidence intervals.
    
    Parameters
    ----------
    pdf : 2D array
        Map of PDF values on a 2D equally-spaced grid.
    dx : float
        Interval between two grid points in the one direction.
    dy : float
        Interval between two grid points in the other direction.    
    conf_intervals : list of floats
        Confidence intervals, from 0 to 1, to look for.
        
    Returns
    -------
    levels : array of floats
        PDF values that correspond to the given confidence intervals.
    """
    
    # Sort intervals:
    conf_intervals = sorted(conf_intervals)[::-1]
    
    # Get correspondence between levels and confidence intervals:
    z_values = np.unique(pdf.ravel())[::-1]
    z_prob = np.array([pdf[pdf >= z].sum() * dx * dy for z in z_values])
    
    # Get intervals for the requested levels:
    levels = np.array([z_values[np.argmin((z_prob - p)**2)] for p in conf_intervals])
    
    return levels

def plot_score_correspondence(stats_df, intercept, coef, label, color, annotation_col='violento', final_suffix='_final', machine_col='hate_score'):
    """
    Create a scatter plot, with error bars, of the estimated fraction of successes
    in a series of binomial experiments, along with a linear fit line.
    
    Parameters
    ----------
    stats_df : DataFrame
        Table with binomial statistics per bin (data point).
    intercept : float
        Intercept of the linear fit.
    coef : float
        Angular coefficient of the linear fit.
    label : str
        What to call the data points.
    color : str
        Color used for the data points and fit line.
    annotation_col : str
        Substring present in the `stats_df` columns, indicating the question
        answered by the annotators.
    final_suffix : str
        Suffix added to `annotation_col`, present in the `stats_df` columns, 
        indicating the aggregation or annotator used when computing the stats.
    machine_col : str
        Suffix indicating the score used to bin the data.
    """
    
    pl.plot([0,100], [intercept * 100, intercept * 100 + coef * 100], linewidth=1, color=color)
    
    pl.errorbar(stats_df['machine_avg_' + machine_col] * 100, stats_df['freq_' + annotation_col + final_suffix] * 100, 
                yerr=[(stats_df['freq_' + annotation_col + final_suffix] - stats_df['min_' + annotation_col + final_suffix]).clip(lower=0) * 100,
                      (stats_df['max_' + annotation_col + final_suffix] - stats_df['freq_' + annotation_col + final_suffix]).clip(lower=0) * 100],
                marker='o', linestyle='none', alpha=0.5, color=color, label=label)

def plot_tweets_correspondence(tweets_df, label, color, annotation_col='violento', final_suffix='_final'):
    """
    Creates two subplots:
    - A scatter plot + linear fit of the fraction of annotated tweets per bin of 
      IA hate score;
    - A 2D confidence contour plot for the parameters of the linear fit.
    
    Parameters
    ----------
    tweets_df : DataFrame
        Table of tweets and their annotations, along with other information.
    label : str
        What to call the data points.
    color : str
        Color used for the data points, contours and fit line.
    annotation_col : str
        Substring present in the `stats_df` columns, indicating the question
        answered by the annotators.
    final_suffix : str
        Suffix added to `annotation_col`, present in the `stats_df` columns, 
        indicating the aggregation or annotator used when computing the stats.    
    """
    
    # Hard-coded:
    amin = -0.05
    amax =  0.6
    da   =  0.002
    bmin =  0.1
    bmax =  1.1
    db   =  0.002

    # Count annotations:
    stats_df = annotations_to_stats(tweets_df, annotation_col=annotation_col, final_suffix=final_suffix)
    
    # Compute the posterior for the linear relation coefficients:
    aa, bb, Post = map_multi_binomial_posterior(stats_df, amin, amax, da, bmin, bmax, db, n_success_col= annotation_col + final_suffix)
    a_max, b_max = get_2d_max(aa, bb, Post)

    pl.subplot(1,2,1)
    plot_score_correspondence(stats_df, a_max, b_max, label, color, annotation_col=annotation_col, final_suffix=final_suffix)
    pl.subplot(1,2,2)
    pl.contour(bb, aa, Post, colors=color, alpha=0.5, levels=find_pdf_levels(Post, da, db))
    
    return a_max, b_max


def f_to_precision(threshold, density, fraction):
    """
    Compute precision as a function of the threshold from the density of
    instances per score and the fraction of positive instances per score.
    
    Parameters
    ----------
    threshold : array (N,)
        Probability threshold above which an instance is predicted as 
        positive. This should be generated by np.arange(0, 1, dx).
    density : array (N,)
        Density of instances per probability score.
    fraction : array (N,)
        Fraction of true positive instances poer probability score.
    
    Returns
    -------
    precision : array (N,)
        Precision metric for each threshold. It is computed for instances
        above the threshold.
    """
    
    # Compute data sampling interval:
    dt = threshold[1:] - threshold[:-1]
    dt = np.append(dt, dt[-1]) # Assume last data use the lase bin available.
    
    # Compute quantities:
    true_positives = (density * fraction * dt)[::-1].cumsum()[::-1]
    pred_positives = (density * dt)[::-1].cumsum()[::-1]
    
    # Compute precision:
    precision = true_positives / pred_positives
    
    return precision


def f_to_recall(threshold, density, fraction):
    """
    Compute recall as a function of the threshold from the density of
    instances per score and the fraction of positive instances per score.
    
    Parameters
    ----------
    threshold : array (N,)
        Probability threshold above which an instance is predicted as 
        positive. This should be generated by np.arange(0, 1, dx).
    density : array (N,)
        Density of instances per probability score.
    fraction : array (N,)
        Fraction of true positive instances poer probability score.
    
    Returns
    -------
    recall : array (N,)
        Recall metric for each threshold. It is computed for instances
        above the threshold.
    """
    
    # Compute data sampling interval:
    dt = threshold[1:] - threshold[:-1]
    dt = np.append(dt, dt[-1])
    
    # Compute quantities:
    true_positives = (density * fraction * dt)[::-1].cumsum()[::-1]
    all_positives  = true_positives[0]
    
    # Compute recall:
    recall = true_positives / all_positives
    
    return recall
