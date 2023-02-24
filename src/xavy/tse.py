#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Functions for dealing with data from Brazilian Supreme Electoral Court (TSE)
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
from . import dataframes as xd


def load_tse_raw_data(path, usecols=None, low_memory=False, sep=';', encoding='latin-1'):
    """
    Load raw TSE data from CSV file downloaded 
    directly from TSE.
    
    NOTE: This function works with newer data 
    that includes the header. Older files that
    do not have headers will not work properly.
    
    Parameters
    ----------
    path : str or Path
        Path to the CSV file.
    usecols : list of str or int, or None
        Columns to load. If None, load everything. 
    low_memory : bool
        Whether to set `pd.read_csv` parameter
        with the same name to True or False. 
        Defaults to False to avoid mixed data 
        types.
    sep : str
        Column separator used in the CSV file.
        Defaults to ';', the separator used 
        by TSE.
    encoding : str
        The encoding used by the CSV file. TSE
        usually uses 'latin-1'. Another encoding
        would be 'utf-8', for instace.
        
    Returns
    -------
    df : DataFrame
        A Pandas DataFrame containing the data.
    """
    df = pd.read_csv(path, low_memory=low_memory, sep=sep, encoding=encoding, usecols=usecols)
    return df


def load_tse_votacao_data(filename, cargos=None, usecols=None):
    """
    Load raw TSE CSV file containing the column DS_CARGO
    and return the rows and columns selected.
    
    Parameters
    ----------
    filename : str
        Path to the CSV file.
    cargos : str or list of str, or None
        Name of the cargos to select from the raw data.
        Only rows containing it will be returned. If 
        None, do not filter.
    usecols : str or list of str, or None
        Columns to be returned. If None, return all
        columns.
    
    Returns
    -------
    df : DataFrame
        Filtered data from the CSV file.
    """
    
    # Standardize input:
    if type(cargos) in (int, str):
        cargos = [cargos]
    
    # Load data:
    df = pd.read_csv(filename, encoding='latin-1', sep=';', usecols=usecols)
    
    # Filter data:
    if cargos is not None:
        df = df.loc[df['DS_CARGO'].isin(cargos)]
    
    return df


def contabiliza_votos_por_cand(df, cargos=['DEPUTADO FEDERAL'], resultados=['ELEITO', 'ELEITO POR MÉDIA', 'ELEITO POR QP', 'SUPLENTE'], turno=1,
                               agg_by=['CD_MUNICIPIO'], 
                               id_cols=['SG_UF', 'NM_MUNICIPIO', 'DS_CARGO', 'NR_CANDIDATO', 'NM_CANDIDATO', 'NM_URNA_CANDIDATO', 'DS_SITUACAO_CANDIDATURA', 'DS_DETALHE_SITUACAO_CAND', 'SG_PARTIDO', 'DS_COMPOSICAO_COLIGACAO', 'DS_SIT_TOT_TURNO'], 
                               verbose=False):
    """
    Sum the votes of each candidate under the specified 
    slices and groups.
    
    Parameters
    ----------
    df : DataFrame
        Cleaned 'votacao_candidato_munzona' TSE CSV file.
        Check the data in the 'laranjometro' project.
    cargos : list of str
        Select candidates from `df` that are running for
        these cargos (e.g. `[SENADOR]`)
    resultados : list of str
        Select candidates whose final result in the 
        elections ('DS_SIT_TOT_TURNO') is one of these.
    turno : int
        Select votes gained by the candidate in this 
        turn (for legislative positions this is 1).
    agg_by : list of str
        Besides the candidate themselves, the votes 
        will be counted for the columns specified here.
    id_cols : list of str
        Columns whose values are unique to the candidate 
        and `agg_by` columns and that should be added to 
        the final DataFrame.
    verbose : bool
        Print messages or not.
    
    Returns
    -------
    final : DataFrame
        A dataframe whose columns are 'SQ_CANDIDATO', 
        `agg_by`, `id_cols` and the total number of 
        votes for the selections specified in each line.
    """
    
    # Security checks:
    assert type(verbose) == bool
    assert agg_by is not None, 'Especifique o nível de agregação (colunas de agregação) dos votos.'

    # Standardize input:
    if type(agg_by) == str:
        agg_by = [agg_by]
    if 'SQ_CANDIDATO' not in agg_by:
        agg_by = ['SQ_CANDIDATO'] + agg_by

    # Selecionando dados de interesse:
    if verbose:
        print('Selecionando dados de interesse...')
    df = df.loc[df['DS_CARGO'].isin(cargos)]
    df = df.loc[df['DS_SIT_TOT_TURNO'].isin(resultados)]
    df = df.loc[df['NR_TURNO'] == turno]
    # Slice security checks:
    assert df['NM_TIPO_ELEICAO'].nunique() <= 1, 'Encontrado mais um tipo de eleição. Verificar.'

    # Agregando dados:
    if agg_by != None:
        # Extraindo informações únicas do agrupamento:
        if verbose:
            print('Extraindo dados de identificação...')
        extra_info = xd.unique_traits(df, agg_by, id_cols)
        # Totalizando votos nos agrupamentos:
        if verbose:
            print('Contabilizando os votos...')
        votos = df.groupby(agg_by)['QT_VOTOS_NOMINAIS'].sum()
        # Juntando votos às informações únicas:
        if verbose:
            print('Juntando dados de identificação...')
        final = extra_info.join(votos, on=agg_by, how='right')

    return final
