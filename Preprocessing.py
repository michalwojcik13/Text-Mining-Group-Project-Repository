import re
import nltk
import pandas as pd
import numpy as np

from collections import defaultdict, Counter
from tqdm import tqdm
from unidecode import unidecode
from nltk.tokenize.treebank import TreebankWordDetokenizer
from spellchecker import SpellChecker
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Spell checker initialization
spell = SpellChecker()

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Function to clean text using regex
def regex_cleaner(raw_text, 
            no_emojis, 
            no_hashtags,
            hashtag_retain_words,
            no_newlines,
            no_urls,
            no_punctuation):
    
    #patterns 
    newline_pattern = "(\\n)"
    hashtags_at_pattern = "([#\@@\u0040\uFF20\uFE6B])"
    hashtags_ats_and_word_pattern = "([#@]\w+)"
    emojis_pattern = "([\u2600-\u27FF])"
    url_pattern = "(?:\w+:\/{2})?(?:www)?(?:\.)?([a-z\d]+)(?:\.)([a-z\d\.]{2,})(\/[a-zA-Z\/\d]+)?" ##Note that this URL pattern is *even better*
    punctuation_pattern = "[\u0021-\u0026\u0028-\u002C\u002E-\u002F\u003A-\u003F\u005B-\u005F\u2010-\u2028\ufeff`]+"
    apostrophe_pattern = "'(?=[A-Z\s])|(?<=[a-z\.\?\!\,\s])'"
    separated_words_pattern = "(?<=\w\s)([A-Z]\s){2,}"
    ##note that this punctuation_pattern doesn't capture ' this time to allow our tokenizer to separate "don't" into ["do", "n't"]
    
    if no_emojis == True:
        clean_text = re.sub(emojis_pattern,"",raw_text)
    else:
        clean_text = raw_text

    if no_hashtags == True:
        if hashtag_retain_words == True:
            clean_text = re.sub(hashtags_at_pattern,"",clean_text)
        else:
            clean_text = re.sub(hashtags_ats_and_word_pattern,"",clean_text)
        
    if no_newlines == True:
        clean_text = re.sub(newline_pattern," ",clean_text)

    if no_urls == True:
        clean_text = re.sub(url_pattern,"",clean_text)
    
    if no_punctuation == True:
        clean_text = re.sub(punctuation_pattern,"",clean_text)
        clean_text = re.sub(apostrophe_pattern,"",clean_text)

    return clean_text

# Lemmatize all tokens
def lemmatize_all(token, list_pos=["n","v","a","r","s"]):
    
    wordnet_lem = nltk.stem.WordNetLemmatizer()
    for arg_1 in list_pos:
        token = wordnet_lem.lemmatize(token, arg_1)
    return token

# Function to correct spelling mistakes
def correct_spelling(text):
    return ' '.join([spell.correction(word) if spell.correction(word) is not None else word for word in text.split()])

# Function to handle slang and emojis using VADER
def handle_vader_sentiment(text):
    vader_scores = analyzer.polarity_scores(text)
    if vader_scores['compound'] >= 0.05:
        return "positive"
    elif vader_scores['compound'] <= -0.05:
        return "negative"
    else:
        return "neutral"

# Main preprocessing pipeline
def main_pipeline(raw_text, 
                  no_emojis = True,
                  no_hashtags = True,
                  hashtag_retain_words = True,
                  no_newlines = True,
                  no_urls = True,
                  no_punctuation = True,
                  print_output = True, 
                  no_stopwords = True,
                  custom_stopwords = [],
                  convert_diacritics = True, 
                  lowercase = True, 
                  lemmatized = True,
                  list_pos = ["n","v","a","r","s"],
                  stemmed = False, 
                  pos_tags_list = "no_pos",
                  tokenized_output = False,
                  **kwargs):
    
    """Preprocess strings according to the parameters"""

    # Correct spelling mistakes
    clean_text = correct_spelling(raw_text)
    
    # Handle VADER sentiment for slang and emojis
    vader_result = handle_vader_sentiment(clean_text)
    
    clean_text = regex_cleaner(clean_text, 
                               no_emojis=no_emojis,
                               no_hashtags=no_hashtags,
                               hashtag_retain_words=hashtag_retain_words,
                               no_newlines=no_newlines,
                               no_urls=no_urls,
                               no_punctuation=no_punctuation,
                                 **kwargs)
    
    clean_text = re.sub("'m","am",clean_text)
    clean_text = re.sub("n't","not",clean_text)
    clean_text = re.sub("'s","is",clean_text)

    if convert_diacritics == True:
        clean_text = unidecode(clean_text)

    if lemmatized == True:
        clean_text = ' '.join([lemmatize_all(token, list_pos=list_pos) for token in clean_text.split()])
    
    if stemmed == True:
        porterstemmer = nltk.stem.PorterStemmer()
        clean_text = ' '.join([porterstemmer.stem(token) for token in clean_text.split()])
 
    if lowercase == True:
        clean_text = clean_text.lower()

    if print_output == True:
        print(raw_text)
        print(clean_text)
    
    return clean_text
        

def cooccurrence_matrix_sentence_generator(preproc_sentences, sentence_cooc=False, window_size=5):

    co_occurrences = defaultdict(Counter)

    # Compute co-occurrences
    if sentence_cooc == True:
        for sentence in tqdm(preproc_sentences):
            for token_1 in sentence:
                for token_2 in sentence:
                    if token_1 != token_2:
                        co_occurrences[token_1][token_2] += 1
    else:
        for sentence in tqdm(preproc_sentences):
            for i, word in enumerate(sentence):
                for j in range(max(0, i - window_size), min(len(sentence), i + window_size + 1)):
                    if i != j:
                        co_occurrences[word][sentence[j]] += 1

    #ensure that words are unique
    unique_words = list(set([word for sentence in preproc_sentences for word in sentence]))

    # Initialize the co-occurrence matrix
    co_matrix = np.zeros((len(unique_words), len(unique_words)), dtype=int)

    # Populate the co-occurrence matrix
    word_index = {word: idx for idx, word in enumerate(unique_words)}
    for word, neighbors in co_occurrences.items():
        for neighbor, count in neighbors.items():
            co_matrix[word_index[word]][word_index[neighbor]] = count

    # Create a DataFrame for better readability
    co_matrix_df = pd.DataFrame(co_matrix, index=unique_words, columns=unique_words)

    co_matrix_df = co_matrix_df.reindex(co_matrix_df.sum().sort_values(ascending=False).index, axis=1)
    co_matrix_df = co_matrix_df.reindex(co_matrix_df.sum().sort_values(ascending=False).index, axis=0)

    # Return the co-occurrence matrix
    return co_matrix_df

