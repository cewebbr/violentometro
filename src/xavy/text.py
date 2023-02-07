import numpy as np
import re
import matplotlib.pyplot as pl


#####################
### Preprocessing ###
#####################

def replace_bold_unicode(string):
    """
    Replace unicode characters that represent bold 
    characters with normal characters.
    """
    
    # Characters to be replaced:
    bold_sans    = list('𝗔𝗕𝗖𝗗𝗘𝗙𝗚𝗛𝗜𝗝𝗞𝗟𝗠𝗡𝗢𝗣𝗤𝗥𝗦𝗧𝗨𝗩𝗪𝗫𝗬𝗭𝗮𝗯𝗰𝗱𝗲𝗳𝗴𝗵𝗶𝗷𝗸𝗹𝗺𝗻𝗼𝗽𝗾𝗿𝘀𝘁𝘂𝘃𝘄𝘅𝘆𝘇𝟬𝟭𝟮𝟯𝟰𝟱𝟲𝟳𝟴𝟵')
    bold_serif   = list('𝐀𝐁𝐂𝐃𝐄𝐅𝐆𝐇𝐈𝐉𝐊𝐋𝐌𝐍𝐎𝐏𝐐𝐑𝐒𝐓𝐔𝐕𝐖𝐗𝐘𝐙𝐚𝐛𝐜𝐝𝐞𝐟𝐠𝐡𝐢𝐣𝐤𝐥𝐦𝐧𝐨𝐩𝐪𝐬𝐫𝐭𝐮𝐯𝐰𝐱𝐲𝐳𝟎𝟏𝟐𝟑𝟒𝟓𝟔𝟕𝟖𝟗')
    italic_sans  = list('𝘈𝘉𝘊𝘋𝘌𝘍𝘎𝘏𝘐𝘑𝘒𝘓𝘔𝘕𝘖𝘗𝘘𝘙𝘚𝘛𝘜𝘝𝘞𝘟𝘠𝘡𝘢𝘣𝘤𝘥𝘦𝘧𝘨𝘩𝘪𝘫𝘬𝘭𝘮𝘯𝘰𝘱𝘲𝘴𝘳𝘵𝘶𝘷𝘸𝘹𝘺𝘻0123456789')
    italic_serif = list('𝐴𝐵𝐶𝐷𝐸𝐹𝐺𝐻𝐼𝐽𝐾𝐿𝑀𝑁𝑂𝑃𝑄𝑅𝑆𝑇𝑈𝑉𝑊𝑋𝑌𝑍𝑎𝑏𝑐𝑑𝑒𝑓𝑔ℎ𝑖𝑗𝑘𝑙𝑚𝑛𝑜𝑝𝑞𝑠𝑟𝑡𝑢𝑣𝑤𝑥𝑦𝑧0123456789')
    both_sans    = list('𝘼𝘽𝘾𝘿𝙀𝙁𝙂𝙃𝙄𝙅𝙆𝙇𝙈𝙉𝙊𝙋𝙌𝙍𝙎𝙏𝙐𝙑𝙒𝙓𝙔𝙕𝙖𝙗𝙘𝙙𝙚𝙛𝙜𝙝𝙞𝙟𝙠𝙡𝙢𝙣𝙤𝙥𝙦𝙨𝙧𝙩𝙪𝙫𝙬𝙭𝙮𝙯0123456789')
    both_serif   = list('𝑨𝑩𝑪𝑫𝑬𝑭𝑮𝑯𝑰𝑱𝑲𝑳𝑴𝑵𝑶𝑷𝑸𝑹𝑺𝑻𝑼𝑽𝑾𝑿𝒀𝒁𝒂𝒃𝒄𝒅𝒆𝒇𝒈𝒉𝒊𝒋𝒌𝒍𝒎𝒏𝒐𝒑𝒒𝒔𝒓𝒕𝒖𝒗𝒘𝒙𝒚𝒛0123456789')
    # Target characters (in the same order):
    norm_chars    = list('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789')

    charsets = [bold_sans, bold_serif, italic_sans, italic_serif, both_sans, both_serif]

    for charset in charsets:
        for b, n in zip(charset, norm_chars):
            string = string.replace(b,n)
    
    return string


def text2tag(text):
    """
    Simplify `text` to use it as part os filenames and so on
    (lowercase it, remove accents and spaces).
    """

    # Remove duplicated spaces:
    text = ' '.join(text.split())
    # Transform to tag:
    tag  = re.sub('[\.,;!:\(\)/]', '', remove_accents(text).lower().replace(' ', '_'))
    return tag


def remove_accents(string, i=0):
    """
    Input: string
    
    Returns the same string, but without all portuguese-valid accents.
    """
    
    # Missing values case:
    if type(string) == type(np.NaN):
        return string
    
    accent_list = [('Ç','C'),('Ã','A'),('Á','A'),('À','A'),('Â','A'),('É','E'),('Ê','E'),('Í','I'),('Õ','O'),('Ó','O'),
                   ('Ô','O'),('Ú','U'),('Ü','U'),('ç','c'),('ã','a'),('á','a'),('à','a'),('â','a'),('é','e'),('ê','e'),
                   ('í','i'),('õ','o'),('ó','o'),('ô','o'),('ú','u'),('ü','u'),('È','E'),('Ö','O'),('Ñ','N'),('è','e'),
                   ('ö','o'),('ñ','n'),('Ë','E'),('ë','e'),('Ä','A'),('ä','a')]
    if i >= len(accent_list):
        return string
    else:
        string = string.replace(*accent_list[i])
        return remove_accents(string, i + 1)

    
def remove_punctuation(text, whitespace=False):
    """
    Remove punctuations and special characters
    from string.
    
    Parameters
    ----------
    text : str
        String to be cleaned.
    whitespace : bool
        Whether to replace punctuations and 
        special characters with a whitespace 
        (True) or to just remove them (False).
    
    Returns
    -------
    cleaned : str
        Original string `text` with punctuations
        and special characters removed or 
        replaced by whitespaces.        
    """

    # Hard-coded:
    chars = '“”!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    
    # For missing values:
    if type(text) == type(np.NaN):
        return text
    
    if whitespace is False:
        return text.translate(str.maketrans('', '', chars))
    else:
        return text.translate(str.maketrans(chars, ' ' * len(chars)))


def lowercase(text):
    return text.lower()


def return_pt_stopwords():
    stopwords = ['de', 'a', 'o', 'que', 'e', 'é', 'do', 'da', 'em', 'um', 'para', 'com', 'não', 'uma', 'os', 'no', 'se',
                 'na', 'por', 'mais', 'as', 'dos', 'como', 'mas', 'ao', 'ele', 'das', 'à', 'seu', 'sua', 'ou', 'quando',
                 'muito', 'nos', 'já', 'eu', 'também', 'só', 'pelo', 'pela', 'até', 'isso', 'ela', 'entre', 'depois',
                 'sem', 'mesmo', 'aos', 'seus', 'quem', 'nas', 'me', 'esse', 'eles', 'você', 'essa', 'num', 'nem', 'suas',
                 'meu', 'às', 'minha', 'numa', 'pelos', 'elas', 'qual', 'nós', 'lhe', 'deles', 'essas', 'esses', 'pelas',
                 'este', 'dele', 'tu', 'te', 'vocês', 'vos', 'lhes', 'meus', 'minhas', 'teu', 'tua', 'teus', 'tuas',
                 'nosso', 'nossa', 'nossos', 'nossas', 'dela', 'delas', 'esta', 'estes', 'estas', 'aquele', 'aquela',
                 'aqueles', 'aquelas', 'isto', 'aquilo', 'estou', 'está', 'estamos', 'estão', 'estive', 'esteve',
                 'estivemos', 'estiveram', 'estava', 'estávamos', 'estavam', 'estivera', 'estivéramos', 'esteja',
                 'estejamos', 'estejam', 'estivesse', 'estivéssemos', 'estivessem', 'estiver', 'estivermos', 'estiverem',
                 'hei', 'há', 'havemos', 'hão', 'houve', 'houvemos', 'houveram', 'houvera', 'houvéramos', 'haja', 'hajamos',
                 'hajam', 'houvesse', 'houvéssemos', 'houvessem', 'houver', 'houvermos', 'houverem', 'houverei', 'houverá',
                 'houveremos', 'houverão', 'houveria', 'houveríamos', 'houveriam', 'sou', 'somos', 'são', 'era', 'éramos',
                 'eram', 'fui', 'foi', 'fomos', 'foram', 'fora', 'fôramos', 'seja', 'sejamos', 'sejam', 'fosse', 'fôssemos',
                 'fossem', 'for', 'formos', 'forem', 'serei', 'será', 'seremos', 'serão', 'seria', 'seríamos', 'seriam',
                 'tenho', 'tem', 'temos', 'tém', 'tinha', 'tínhamos', 'tinham', 'tive', 'teve', 'tivemos', 'tiveram',
                 'tivera', 'tivéramos', 'tenha', 'tenhamos', 'tenham', 'tivesse', 'tivéssemos', 'tivessem', 'tiver',
                 'tivermos', 'tiverem', 'terei', 'terá', 'teremos', 'terão', 'teria', 'teríamos', 'teriam']
    return stopwords


def remove_stopwords(text, stopwords=None):

    # Load stopwords if not provided:
    if stopwords == None:
        stopwords = return_pt_stopwords()
        
    word_list = text.split()
    word_list = [word for word in word_list if not word in set(stopwords)]
    return ' '.join(word_list)


def stem_words(text):

    import nltk
    
    stemmer   = nltk.stem.RSLPStemmer()
    #stemmer   = nltk.stem.PorterStemmer()

    word_list = text.split()
    word_list = [stemmer.stem(word) for word in word_list]
    
    return ' '.join(word_list)


def keep_only_letters(text):
    only_letters = re.sub('[^a-z A-ZÁÂÃÀÉÊÍÔÓÕÚÜÇáâãàéêíóôõúüç]', '', text)
    only_letters = ' '.join(only_letters.split())
    return only_letters


##########################
### Dealing with typos ###
##########################


def build_typos_dict(keyboard_row):
    """
    Given a sequence of letters `keyboard_row` 
    that next to each other on the keyboard 
    (e.g 'qwertyuiop[]\'), return a dict from 
    each character in `keyboard_row` to  list
    of this characters and the neighbouring 
    ones.
    """
    n_keys = len(keyboard_row)
    l = list(keyboard_row)
    typo_dict = {l[i]:l[max(0,i-1):min(n_keys, i+2)] for i in range(n_keys)}
    
    return typo_dict


def join_set_lists(l1, l2):
    """
    Return a list of the unique values
    present in the set formed by lists 
    `l1` and `l2`.
    """
    return list(set(l1) | set(l2))


def update_typo_dict(typo_dict, extra_dict):
    """
    Given two dicts `typo_dict` and `extra_dict`
    that have lists as values, update the first 
    with the latter by joining the lists from 
    the same key, but removing duplicated items
    in the final list.
    """
    for key, value in extra_dict.items():
        typo_dict.update({key: join_set_lists(typo_dict.get(key, []), value)})
    
    return typo_dict


def full_build_typos_dict(rows_list=['qwertyuiop', 'asdfghjkl;', 'zxcvbnm,', "qwertyuiop'", 'asdfghjklç', 'zxcvbnm,']):
    """
    Build a dict that goes from a character to
    a list of characters that may be typos of 
    the key character. 
    
    Only lower case characters were included as 
    keys, and those with a special meaning in 
    regex were ignored.
    
    It is assumed that typos only occur 
    between horizontal neighbouring keys in
    the US and BR keyboards.
    """
    
    # For reference only:
    us_row1 = 'qwertyuiop'
    us_row2 = 'asdfghjkl;'
    us_row3 = 'zxcvbnm,'
    br_row1 = "qwertyuiop'"
    br_row2 = 'asdfghjklç'
    br_row3 = 'zxcvbnm,'

    typo_dict = dict()
    for row in rows_list:
        update_typo_dict(typo_dict, build_typos_dict(row))
    
    return typo_dict


##################
### WordClouds ###
##################


class SimpleGroupedColorFunc(object):
    """Create a color function object which assigns EXACT colors
       to certain words based on the color to words mapping

       Parameters
       ----------
       color_to_words : dict(str -> list(str))
         A dictionary that maps a color to the list of words.

       default_color : str
         Color that will be assigned to a word that's not a member
         of any value from color_to_words.
    """

    def __init__(self, color_to_words, default_color):
        self.word_to_color = {word: color
                              for (color, words) in color_to_words.items()
                              for word in words}

        self.default_color = default_color

    def __call__(self, word, **kwargs):
        return self.word_to_color.get(word, self.default_color)


class GroupedColorFunc(object):
    """Create a color function object which assigns DIFFERENT SHADES of
       specified colors to certain words based on the color to words mapping.

       Uses wordcloud.get_single_color_func

       Parameters
       ----------
       color_to_words : dict(str -> list(str))
         A dictionary that maps a color to the list of words.

       default_color : str
         Color that will be assigned to a word that's not a member
         of any value from color_to_words.
    """

    def __init__(self, color_to_words, default_color):
        self.color_func_to_words = [
            (get_single_color_func(color), set(words))
            for (color, words) in color_to_words.items()]

        self.default_color_func = get_single_color_func(default_color)

    def get_color_func(self, word):
        """Returns a single_color_func associated with the word"""
        try:
            color_func = next(
                color_func for (color_func, words) in self.color_func_to_words
                if word in words)
        except StopIteration:
            color_func = self.default_color_func

        return color_func

    def __call__(self, word, **kwargs):
        return self.get_color_func(word)(word, **kwargs)

    
def plot_word_vector(vocab, vector):
    """
    Plot a wordcloud for words in `vocab` with frequency given by `vector`.
    Positive frequencies are plot in blue, negative frequencias as plot in red.
    """
    
    assert np.shape(vocab) == np.shape(np.array(vector)), '`vocab` and `vector` must have the same shape.'
    assert type(vocab[0]) == str or type(vocab[0]) == np.str_, '`vocab` must be a array-like of strings.'
    
    from wordcloud import (WordCloud, get_single_color_func)
    wc = WordCloud(background_color='white', relative_scaling=0.8, include_numbers=True, width=800, height=400)

    abs_vector = np.abs(vector)
    # Set word sizes:
    word_frequencies = {word:weight for word, weight in zip(vocab, abs_vector)}
    # Define colors of words (negatives are red, positives are blue):
    plus_words  = [word for word, weight in zip(vocab, vector) if weight > 0]
    minus_words = [word for word, weight in zip(vocab, vector) if weight < 0]
    color_to_words =  {'blue': plus_words, 'red': minus_words}
    default_color  = 'grey'
    
    # Generate wordcloud:
    wc.generate_from_frequencies(word_frequencies)
    # Color words according to their signs:
    grouped_color_func = SimpleGroupedColorFunc(color_to_words, default_color)
    wc.recolor(color_func=grouped_color_func)
    
    #pl.figure(figsize=(12,7))
    pl.imshow(wc, interpolation="bilinear")
    pl.axis("off")

