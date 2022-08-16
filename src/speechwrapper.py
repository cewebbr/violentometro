#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Classificador de discurso de ódio
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
import tensorflow as tf
# Hugging Face:
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from datasets import Dataset 
from transformers import DefaultDataCollator 


class HateSpeechModel:
    """
    A wrapper for a trained BERT-like transformer for easy predictions 
    (not to be used for training). All the pre-processing, like tokenizing,
    is performed in the object.
    
    Parameters
    ----------
    trained_model : TFAutoModelForSequenceClassification
        Trained Hugging Face transformer model for hate speech identification.
    tokenizer : AutoTokenizer
        Hugging Face tokenizer associated to the model above.
    do_lower_case : bool
        Whether the tokenizer should transform input to lowercase. This 
        should be the same as the tokenizer when training the model.
    verbose : bool
        Whether to print information about the initialization stages.
    """
    
    def __init__(self, trained_model, tokenizer='neuralmind/bert-base-portuguese-cased', do_lower_case=False, verbose=True):
         
        if verbose is True:
            print('Loading tokenizer from {}'.format(tokenizer))
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, do_lower_case=do_lower_case)

        if verbose is True:
            print('Loading trained model: {}'.format(trained_model))
        self.model = TFAutoModelForSequenceClassification.from_pretrained(trained_model)
    
    
    def build_tokenize_func(self, tokenizer, text_col, max_length):
        """
        Create a tokenizing function to be used 
        by the Hugging Face Dataset method 'map'.
        
        Parameters
        ----------
        tokenizer : HuggingFace AutoTokenizer
            A tokenizer loaded from 
            `transformers.AutoTokenizer.from_pretrained()`.
        text_col : str
            Name of the Dataset element containing 
            the sentences.
        max_length : int
            Maximum length of the sentences (smaller 
            sentences will be padded and longer ones
            will be truncated). This is required for 
            training, so batches have instances of the
            same shape.

        Returns
        -------
        
        func : Callable
            Function f(x), where x is the data
            whose element `text_col` contains 
            sentences. This function tokenizes
            the sentences.
        """
        
        def tokenize_function(examples):
            return tokenizer(examples[text_col], padding=True, max_length=max_length, truncation=True)
        
        return tokenize_function

    
    def process_pandas_to_tfdataset(self, df, max_length=80, shuffle=False, text_col='text', target_col='label', batch_size=8):
        """
        Prepare NLP data in a Pandas DataFrame to be used 
        in a TensorFlow transformer model.
        
        Parameters
        ----------
        df : DataFrame
            The corpus, containing the columns `text_col` 
            (the sentences) and `target_col` (the labels).
        max_length : int
            Maximum length of the sentences (smaller 
            sentences will be padded and longer ones
            will be truncated). This is required for 
            training, so batches have instances of the
            same shape.
        shuffle : bool
            Shuffle the dataset order when loading. 
            Recommended True for training, False for 
            validation/evaluation.
        text_col : str
            Name of `df` column containing the sentences.
        target_col : str
            Name of `df` column containing the labels of 
            the sentences.
        batch_size : int
            The size of the batch in the output 
            tensorflow dataset.
            
        Returns
        -------
        tf_dataset : TF dataset
            A dataset that can be fed into a transformer 
            model.
        """
        
        # Security checks:
        renamed_df = df.rename({target_col:'labels'}, axis=1) # Hugging Face requer esse nome p/ y.
        
        # Define função para processar os dados com o tokenizador:
        #def tokenize_function(examples):
        #    return tokenizer(examples[text_col], padding=True, max_length=max_length, truncation=True)
        tokenize_function = self.build_tokenize_func(self.tokenizer, text_col, max_length)
        
        # pandas -> hugging face:
        hugging_set = Dataset.from_pandas(renamed_df)
        
        # texto -> sequência de IDs: 
        encoded_set = hugging_set.map(tokenize_function, batched=True)
        
        # hugging face -> tensorflow dataset:
        data_collator = DefaultDataCollator(return_tensors="tf")
        tf_dataset = encoded_set.to_tf_dataset(columns=["attention_mask", "input_ids", "token_type_ids"], label_cols=["labels"], shuffle=shuffle, collate_fn=data_collator, batch_size=batch_size)
        
        return tf_dataset


    def predict_proba_from_tfd(self, tf_dataset, verbose='auto'):
        """
        Compute the probability that each instance 
        is hate speech.
    
        Parameters
        ----------
        tf_dataset : Tensorflow Dataset
            The data for which to make predictions
            (already tokenized).
        verbose : 'auto', 0, 1 or 2
            Verbosity mode. 0 = silent, 1 = progress bar, 2 = single 
            line. "auto" defaults to 1 for most cases, and to 2 when 
            used with ParameterServerStrategy. Note that the progress 
            bar is not particularly useful when logged to a file, so 
            verbose=2 is recommended when not running interactively 
            (e.g. in a production environment). 
        
        Returns
        -------
        probs : array
            Probability that the corresponding 
            instance falls in the 'hate speech' 
            binary class.
        """
    
        tf_predict = self.model.predict(tf_dataset, verbose=verbose).logits
        probs = tf.sigmoid(tf_predict)[:,0].numpy()
        
        return probs


    def predict_class_from_tfd(self, tf_dataset, threshold=0.5, verbose='auto'):
        """
        Predict if the input instances are 
        considered hate speech or not.
    
        Parameters
        ----------
        tf_dataset : Tensorflow Dataset
            The data for which to make predictions
            (already tokenized).
        threshold : float
            Probability value from 0 to 1 above 
            which the instance is considered hate
            speech.
        verbose : 'auto', 0, 1 or 2
            Verbosity mode. 0 = silent, 1 = progress bar, 2 = single 
            line. "auto" defaults to 1 for most cases, and to 2 when 
            used with ParameterServerStrategy. Note that the progress 
            bar is not particularly useful when logged to a file, so 
            verbose=2 is recommended when not running interactively 
            (e.g. in a production environment). 
        
        Returns
        -------
        preds : array
            Predicted class for the corresponding
            instances.
        """
    
        probs = self.predict_proba_from_tfd(tf_dataset, verbose=verbose)
        preds = (probs > threshold).astype(int)
    
        return preds


    def predict_proba(self, texts, verbose='auto'):
        """
        Return the probability that the provided 
        sentences are considered hate speech.
        
        Parameters
        ----------
        texts : str or list of str
            Sentences to classify.
        verbose : 'auto', 0, 1 or 2
            Verbosity mode. 0 = silent, 1 = progress bar, 2 = single 
            line. "auto" defaults to 1 for most cases, and to 2 when 
            used with ParameterServerStrategy. Note that the progress 
            bar is not particularly useful when logged to a file, so 
            verbose=2 is recommended when not running interactively 
            (e.g. in a production environment). 
            
        Returns
        -------
        probs : array
            Probabilities that the sentences in 
            `texts` contain violence.
        """
    
        
        # Standardize input:
        if type(texts) == str:
            texts = [texts]
        
        # Prepare date format and tokenize:
        input_df  = pd.DataFrame({'text': texts, 'label': [1] * len(texts)})
        input_tfd = self.process_pandas_to_tfdataset(input_df)
        # Predict with model:
        probs = self.predict_proba_from_tfd(input_tfd, verbose=verbose)
        
        return probs
    

    def predict_class(self, texts, threshold=0.5, verbose='auto'):
        """
        Predict if the input instances are 
        considered hate speech or not.
    
        Parameters
        ----------
        texts : str or list of str
            Sentences to classify.
        threshold : float
            Probability value from 0 to 1 above 
            which the instance is considered hate
            speech.
        
        Returns
        -------
        preds : array
            Predicted class for the corresponding
            instances.
        verbose : 'auto', 0, 1 or 2
            Verbosity mode. 0 = silent, 1 = progress bar, 2 = single 
            line. "auto" defaults to 1 for most cases, and to 2 when 
            used with ParameterServerStrategy. Note that the progress 
            bar is not particularly useful when logged to a file, so 
            verbose=2 is recommended when not running interactively 
            (e.g. in a production environment). 
        """
    
        probs = self.predict_proba(texts, verbose=verbose)
        preds = (probs > threshold).astype(int)
    
        return preds


# If running this code as a script:
if __name__ == '__main__':
    pass
