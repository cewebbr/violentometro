#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Funções de interação com o Google Drive
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
import csv
import gspread


def get_google_sheet(url):
    """
    Load a Google Sheets located at `url` (str) into a DataFrame.
    The sheets must allow access to the service account email stored
    at '~/.config/gspread/service_account.json'. 
    
    For more information on service account settings and keys:
    <https://docs.gspread.org/en/latest/oauth2.html>
    """
    
    gc = gspread.service_account()
    sh = gc.open_by_url(url)
    df = pd.DataFrame(sh.sheet1.get_all_records())
    
    return df


def load_data_from_local_or_drive(url, filename, force_drive=False, save_data=True, low_memory=False):
    """
    Loads data from local file if available or download it from Google Sheets.
    
    Parameters
    ----------    
    url: str
        The URL where to find the Google Sheet.  
    filename : str
        The path to the file where to save the downloaded data and from where to load it.  
    force_drive : bool
        Whether to download data from Google Sheets even if the local file exists.    
    save_data : bool
        Wheter to save downloaded data to local file or not.]
    low_memory : bool
        Whether or not to avoid reading all the data to define the data types
        when loading data from a local file.
        
    Returns
    -------
    df : DataFrame
        The data either loaded from `filename` or retrieved through `query`.
    """
    
    # Download data from Google Sheets and save it to local file:
    if os.path.isfile(filename) == False or force_drive == True:
        print('Loading data from Google Sheets...')
        df = get_google_sheet(url)
        if save_data:
            print('Saving data to local file...')
            df.to_csv(filename, quoting=csv.QUOTE_ALL, index=False)
    
    # Load data from local file:
    else:
        print('Loading data from local file...')
        df = pd.read_csv(filename, low_memory=low_memory)
        
    return df



# If running this code as a script:
if __name__ == '__main__':
    pass
