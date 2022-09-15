#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Sistema automático de ranqueamento por violência de tweets de candidatos capturados
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
import json
import pandas as pd
from pathlib import Path
import datetime as dt
import time
from datasets.utils.logging import set_verbosity_error

import speechwrapper as sw


def read_config(filename='../tweets/tweet_capture_config.json'):
    """
    Read JSON from `filename` (str).
    """
    with open(filename, 'r') as f:
        config = json.load(f)
    return config


def write_config(config, filename='../tweets/tweet_capture_config.json'):
    """
    Write `config` (dict) to `filename` (str) as JSON.
    """
    with open(filename, 'w') as f:
        json.dump(config, f, indent=1)

        
def make_necessary_dirs(filename):
    """
    Create directories in the path to `filename` (str), if necessary.
    """
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

                
def in_out_file_rename(input_file, input_term, output_term):
    """
    Transform an filename `input_file` (str or Path) by 
    replacing `input_term` (str) by `output_term`.
    """
    output_file = str(input_file).replace(input_term, output_term)
    return output_file


def add_hate_score(model, input_file, output_file):
    """
    Load `input_file` (str or path to a CSV file), add column
    'hate_score' and save it to `output_file` (str or path to 
    a CSV file).
    
    Returns the number of instances in `input_file`.
    """
    
    # Read data:
    try:
        # Leitura básica:
        df = pd.read_csv(input_file)
    except pd.errors.ParserError:
        # Erro pode ser causado por carriage return. Nesse caso, tenta:
        df = pd.read_csv(input_file, lineterminator='\n')        
    
    # Rate:
    y_pred = model.predict_proba(df['text'], verbose=0)
    # Add column:
    df['hate_score'] = pd.Series(y_pred, index=df.index)
    
    # Save result:
    make_necessary_dirs(output_file)
    df.to_csv(output_file, index=False)
    
    return len(df)


def parse_utc_time(time_in, time_fmt='%Y-%m-%dT%H:%M:%S', bsb2utc=True):
    """
    Parse a (possibly) local time into UTC time.
    
    Parameters
    ----------
    time_in : str or datetime
        Time to parse to UTC datetime.
    time_fmt : str
        If `time_in` is str, this is used to parse it to datetime.
    bsb2utc : bool
        Whether to assume `time_in` is Brasilia local time and 
        convert it to UTC.
        
    Returns
    -------
    time_utc : datetime
        Time in UTC (assuming `time_in` is UTC and `bsb2utc` is
        False; or `time_in` is UTC-3 and `bsb2utc` is True).
    """
    # Parse str to datetime:
    if type(time_in) is str:
        time_dt = dt.datetime.strptime(time_in, time_fmt)
    else:
        time_dt = time_in
        
    # Convert Brazilia (UTC-3) to UTC:
    if bsb2utc is True:
        time_dt = time_dt + dt.timedelta(hours=3)
    
    return time_dt


def next_batch_time(capture_period, ini_date='2022-08-12T00:00:00', date_fmt='%Y-%m-%dT%H:%M:%S'):
    """
    Compute the datetime of the next batch from now.
    
    Parameters
    ----------
    capture_period : float
        Number of hours between each batch.
    ini_date : str or datetime
        Initial date (if str, in format `date_fmt`), from which
        the following batches are scheduled.
    date_fmt : str
        Format of `ini_date`.
    
    Returns
    -------
    next_date : datetime
        When to run the next batch capture
    """
    next_date = parse_utc_time(ini_date, bsb2utc=False)
    now_date  = dt.datetime.now()
    while next_date < now_date:
        next_date = next_date + dt.timedelta(hours=capture_period)
        
    return next_date


def hate_score_batch(data_dir, input_term='/data/', output_term='/scored/', model='../modelos/bertimbau-hatespeech-v01', file_pattern='*.csv', verbose=True):
    """
    Apply hate speech model to all tweets in root folder 
    and save the original data with the extra score 
    column in another root folder.
    
    Parameters
    ----------
    data_dir : str or Path
        Root folder containing the tweets to be rated.
        All files inside it with pattern given by 
        `file_pattern` will be recursively accessed, 
        rated, and copied to another root folder
        while maintaining the same tree structure.
    input_term : str
        Part of the `data_dir` path to be replaced in 
        order to form the output root folder.
    output_term : str
        Replacement of `input_term`, to form the root 
        output folder.
    model : str or TFAutoModelForSequenceClassification
        Either a path to the folder containing the files 
        for a transformer BERT fine-tuned model or the 
        model itself already loaded.
    file_pattern : str
        Pattern followed by the files in `data_dir` 
        that should be rated by the model. The model
        expects a CSV file with a `text` column.
    verbose : bool
        Whether to print the filenames being rated.
        
    Returns
    -------
    n_instances : int
        The number of tweets rated.
    """
    # Teste se GPU está disponível:
    assert len(sw.tf.config.list_physical_devices('GPU')) > 0, 'Não encontrei nenhuma GPU.' 
    # Desativa progress bar do hugging face:
    set_verbosity_error()
    
    # Carrega modelo:
    if type(model) is str:
        model  = sw.HateSpeechModel(model)
    
    # Carrega dados:
    data_list = sorted(list(Path(data_dir).rglob(file_pattern)))

    # Loop sobre arquivos:
    n_instances = 0
    for filename in data_list:
        if verbose is True:
            print(filename, flush=True)
        # Gera nome de arquivo de saída:
        out_file = in_out_file_rename(filename, input_term, output_term)
        # Aplica modelo e salva resultado; retorna o número de tweets:
        n_instances += add_hate_score(model, filename, out_file)
    
    return n_instances


def path2dt(path, time_fmt='%Y-%m-%dT%H:%M:%S'):
    """
    Given a file `path` (str or Path) where the 
    last portion is a folder named as a datetime
    in format `time_fmt`, return a parsed datetime.
    """
    return parse_utc_time(str(path).split('/')[-1], time_fmt=time_fmt, bsb2utc=False)


def select_batches_to_score(first_batch, last_batch, data_dir, scored_dir, force_rate=False, data_folder_pattern='[!.]*'):
    """
    Create a list of batch folders to score 
    with the hate speech model.
    
    Parameters
    ----------
    first_batch : str or datetime
        Mininum batch date to look for in 
        `data_dir`.
    last_batch : str or datetime
        Maximum batch date to look for in 
        `data_dir`.
    data_dir : str
        Folder containing the tweets to be 
        scored, stored in subfolders named 
        by their batch times.
    scored_dir : str
        Folder where scored tweets will be 
        saved, following the same subfolder
        structure. Only used to check if 
        `force_rate` is False, to find out if 
        the subfolder was already scored.
    force_rate : bool
        Whether all batch subfolders in the 
        `first_batch` to `last_batch` time 
        interval should be scored, or just 
        those not yeat scored.
    
    Returns
    -------
    sel_batch_dirs : list of Paths
        Paths to subfolders in `data_dir` 
        containing the tweets to be rated.
    """
    
    # Standardize input:
    first_batch = parse_utc_time(first_batch, bsb2utc=False)
    last_batch  = parse_utc_time(last_batch, bsb2utc=False)
    
    # Get batch folders in data dir:
    batch_dirs = sorted(list(Path(data_dir).glob(data_folder_pattern)))
    
    # Filter batch folders by time:
    sel_batch_dirs = [path for path in batch_dirs if first_batch <= path2dt(path) and path2dt(path) <= last_batch]        
    
    # Remove batches already rated, unless otherwise requested: 
    if force_rate is False:
        already_scored = [Path(in_out_file_rename(f, data_dir, scored_dir)).exists() for f in sel_batch_dirs]
        sel_batch_dirs = [d for d,e in zip(sel_batch_dirs, already_scored) if e is False]
    
    return sel_batch_dirs


def log_print(string, start=False):
    print('{} {}: {}'.format('*' if start else ' ', dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), string), flush=True)


def driver():
    
    # Init config:
    config = read_config()
    batch_time = config['batch_ref_time']
    
    # Desativa mensagens do hugging face:
    set_verbosity_error()

    while True:
        
        # Wait for next batch:
        batch_time = next_batch_time(config['capture_period'], ini_date=batch_time)
        log_print('Next batch at [{}]. Sleeping...'.format(batch_time))
        sleep_time = (batch_time - dt.datetime.now()).total_seconds()
        time.sleep(sleep_time)
        
        # Load data and config:
        config = read_config()
        log_print('Batch config! data_dir: {}, scored_dir: {}, model_dir: {}, force_rate: {}'.format(config['data_dir'], config['scored_dir'], config['model_dir'], config['force_rate']))
        # Load model:
        model = sw.HateSpeechModel(config['model_dir'])

        # Get time interval to rate:
        first_batch = parse_utc_time(config['batch_ref_time'], bsb2utc=False)
        last_batch  = batch_time - dt.timedelta(hours=config['capture_period'])
        log_print('Batch time range: [{}] -- [{}]'.format(first_batch.strftime('%Y-%m-%d %H:%M:%S'), last_batch.strftime('%Y-%m-%d %H:%M:%S')))
        # Get batch folders to score:
        batches_to_score = select_batches_to_score(first_batch, last_batch, config['data_dir'], config['scored_dir'], config['force_rate'])
        log_print('Will score {} batches.'.format(len(batches_to_score)))
        
        # Run next batch:
        for b in batches_to_score:
            log_print('Scoring batch {}...'.format(b))
            n_instances = hate_score_batch(b, config['data_dir'], config['scored_dir'], model)
            log_print('Finished batch! Tweets rated: {:d}'.format(n_instances))


# If running this code as a script:
if __name__ == '__main__':

    driver()
