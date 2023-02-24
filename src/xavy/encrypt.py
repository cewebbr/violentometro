#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Functions for symmetric stochastic encryption.
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


from cryptography.fernet import Fernet


class SymmetricCrypto:
    
    def __init__(self, key_path=None, implementation='fernet', encoding='utf-8'):
        """
        Create a symmetric encryptor object.
        
        Parameters
        ----------
        key_path : Path or str
            Path to the text file containing the encryption key.
        implementation : str
            Name of the symmetric encryption implementation. 
            Possible options are: 'fernet'.
        
        encoding : str
            Name of the string encoding used (e.g. utf-8, latin-1).
        """
        
        # Copy variables:
        self.implementation = implementation
        self.encoding = encoding
        self.key_path = key_path
        
        # Init variables:
        self.key = None
        self.cipher_suite = None
        
        # Security checks:
        known = {'fernet'}
        assert implementation in known, "'{}' not one of the implementations known: {}".format(implementation, known)

        # Create encryptor if key is provided:
        if key_path is not None:
            self.load_key(self.key_path)
            
        
    def generate_key(self, output=False):
        """
        Generate encryption key and store it in this object.
        
        Parameters
        ----------
        output : bool
            If True, return the generated key.
            
        Outputs
        -------
        key : None or bytes
            The generated key, if `output` is True.
        """
        
        # Generate key:
        if self.implementation == 'fernet':
            self.key = Fernet.generate_key()
        
        # Unknown method:
        else:
            print("No key generation process for implementation '{}'".format(self.implementation))
        
        # Create encryptor:
        self.cipher_suite = Fernet(self.key)
        
        # Return key if requested:
        if output is True:
            return self.key
        

    def save_key(self, path):
        """
        Save the encryption key to a file.
        
        Parameters
        ----------
        path : Path or str
            Path to the file where to store the key
            (decoded to a string).
        """
        
        # Save key to file:
        if self.key is not None:
            with open(path, 'w') as f:
                f.write(self.key.decode(self.encoding))
        
        # No key stored in object:
        else:
            print('Key not found.')
    
    
    def gen_save_key(self, path):
        """
        Generate encryption key, store it in this object and 
        save to a text file.
        
        Parameters
        ----------
        path : Path or str
            Path to the file where to store the key
            (decoded to a string).
        """
        
        # Generate key:
        self.generate_key()
        
        # Save to file:
        self.save_key(path)
        
        
    def load_key(self, path, output=False):
        """
        Load encryption key from a text file.
        
        Parameters
        ----------
        path : Path or str
            Path to a text file from where to read the key.
        output : bool
            If True, return the generated key.
            
        Outputs
        -------
        key : None or bytes
            The generated key, if `output` is True.
        """
        
        # Read file:
        with open(path, 'r') as f:
            self.key = f.read().encode(self.encoding)
            self.key_path = path
        
        # Create encryptor:
        self.cipher_suite = Fernet(self.key)
        
        # Return key of requested:
        if output is True:
            return self.key
    
    
    def encrypt(self, data):
        """
        Encrypt data to a string.
        
        Parameters
        ----------
        data : str, int and others
            Data to be encrypted
            
        Outputs
        -------
        code : str
            Encrypted data in a str format.
        """
        
        # Encode str to bytes:
        b = str(data).encode(self.encoding)
        
        # Encrypt:
        e = self.cipher_suite.encrypt(b)
        
        # Decode encrypted bytes to string:
        r = e.decode(self.encoding)
        
        return r
        
    
    def decrypt(self, data):
        """
        Decrypt a string to another string.
        
        Parameters
        ----------
        data : str
            String to be decrypted.
            
        Outputs
        -------
        string : str
            Decrypted data in a str format.
        """
        
        # Encode str to bytes:
        b = str(data).encode(self.encoding)
        
        # Decrypt:
        e = self.cipher_suite.decrypt(b)
        
        # Decode bytes to str:
        r = e.decode(self.encoding)

        return r
    
    
    def decrypt_int(self, data):
        """
        Decrypt a string to an int (assuming it represented an
        int in the first place).
        
        Parameters
        ----------
        data : str
            String to be decrypted.
            
        Outputs
        -------
        string : str
            Decrypted data to int.
        """
        
        # Decrypt str to str:
        s = self.decrypt(data)
        
        # Convert to int:
        i = int(s)
        
        return i 


    def encrypt_df_cols(self, df, encrypt_cols, verbose=False):
        """
        Encrypt selected columns in a DataFrame.

        Parameters
        ----------
        df : DataFrame
            Table whose specified columns should be encrypted.
        encrypt_cols : iterable of (str or int)
            Columns in `df` to be encrypted.
        verbose : bool
            Whether to print the nams of the columns being encrypted 
            or not.

        Returns
        -------
        encrypted_df : DataFrame
            A copy of `df` but with the data under columns 
            listed in `encrypt_cols` replaced by their 
            encrypted version.
        """

        # Prepare output:
        encrypted_df = df.copy()

        # Select columns in DataFrame to be encrypted:
        target_cols = set(df.columns) & set(encrypt_cols)

        # LOOP over columns:
        for col in target_cols:
            if verbose is True:
                print(col, end='  ')
            # Encrypt cols:
            encrypted_df[col] = encrypted_df[col].apply(self.encrypt)

        return encrypted_df


    def decrypt_df_cols(self, df, decrypt_cols, verbose=False):
        """
        Decrypt selected columns in a DataFrame.

        Parameters
        ----------
        df : DataFrame
            Table whose specified columns should be decrypted.
        decrypt_cols : iterable of (str or int)
            Columns in `df` to be decrypted.
        verbose : bool
            Whether to print the nams of the columns being encrypted 
            or not.

        Returns
        -------
        decrypted_df : DataFrame
            A copy of `df` but with the data under columns 
            listed in `decrypt_cols` replaced by their 
            decrypted version. All these columns have type 
            str.
        """

        # Prepare output:
        decrypted_df = df.copy()

        # Select columns in DataFrame to be encrypted:
        target_cols = set(df.columns) & set(decrypt_cols)

        # LOOP over columns:
        for col in target_cols:
            if verbose is True:
                print(col, end='  ')
            # Encrypt cols:
            decrypted_df[col] = decrypted_df[col].apply(self.decrypt)

        return decrypted_df
