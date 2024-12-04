# python preprocessing
import pandas as pd
import re
import nltk

# Initialize lemmatizer
lemmatizer = nltk.stem.WordNetLemmatizer()

# Function to clean text using regex, ensuring emojis are kept intact
def regex_cleaner(raw_text, 
                  no_hashtags,
                  hashtag_retain_words,
                  no_newlines,
                  no_urls,
                  no_punctuation,
                  no_emojis):
    # Patterns for specific elements, ensuring emojis remain intact
    newline_pattern = "(\\n)"
    hashtags_at_pattern = "([#\@@@＠﹫])"
    hashtags_ats_and_word_pattern = "([#@]\w+)"
    url_pattern = "(?:\w+:\/\/)?(?:www)?(?:\.)?([a-z\d]+)(?:\.)([a-z\d\.]{2,})(\/[a-zA-Z\/\d]+)?"
    punctuation_pattern = "[\u0021-\u0026\u0028-\u002C\u002E-\u002F\u003A-\u003F\u005B-\u005F\u2010-\u2028\ufeff`]+"
    apostrophe_pattern = "'(?=[A-Z\s])|(?<=[a-z\.\?\!\,\s])'"
    
    # Avoid processing emojis
    emoji_pattern = re.compile("[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F700-\U0001F77F\U0001F780-\U0001F7FF\U0001F800-\U0001F8FF\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U00002702-\U000027B0\U000024C2-\U0001F251]+", flags=re.UNICODE)
    clean_text = re.sub(emoji_pattern, lambda m: f' {m.group(0)} ', raw_text)

    if no_hashtags == True:
        if hashtag_retain_words == True:
            clean_text = re.sub(hashtags_at_pattern, "", clean_text)
        else:
            clean_text = re.sub(hashtags_ats_and_word_pattern, "", clean_text)
        
    if no_newlines == True:
        clean_text = re.sub(newline_pattern, " ", clean_text)

    if no_urls == True:
        clean_text = re.sub(url_pattern, "", clean_text)
    
    if no_punctuation == True:
        # Remove punctuation but leave emojis intact
        clean_text = re.sub(punctuation_pattern, "", clean_text)
        clean_text = re.sub(apostrophe_pattern, "", clean_text)
        
    if no_emojis == True:
        clean_text = re.sub(emoji_pattern, "", clean_text)

    return clean_text

# Function to split UPPERCASE words
def splitting_words_process(word):
    # only upper case letters
    if word.isupper():
        return word

    # more than one upper case letter inside
    elif re.search(r'[A-Z][a-z]*[A-Z]', word):
        split_word = re.findall(r'[A-Z][a-z]*', word)
        return ' '.join(split_word)

    # <2 upper case letters
    else:
        return word

# Function to replace 'gud', 'goo', 'gd' with the appropriate 'good'
def replace_gud_with_good(text):
    if isinstance(text, str):
        pattern = re.compile(r'\b([Gg][Uu][Dd]|[Gg][Oo][Oo]|[Gg][Dd])\b(?!\w)')
        return pattern.sub('good', text)
    return text

# Function to replace 'kk', 'Oke', 'k', 'Ok' with 'ok'
def replace_to_ok(text):
    if isinstance(text, str):
        pattern = re.compile(r'\b(k|kk|Ok|Oke)\b', re.IGNORECASE)
        return pattern.sub('ok', text)
    return text

# Function to add space after punctuation
# add space after ! | " | # | $ | % | & | ( | ) | * | + | , | . | : | ; followed immediately by a word
def add_space_after_punctuation(text):

    text = re.sub(r'([\u0021-\u0026\u0028-\u002C\u002E\u003A-\u003F]+(?=\w))', r'\1 ', text)
    return text

# Function to lemmatize all tokens
def lemmatize_all(text, list_pos):
    lemmatized_text = []
    for token in text.split():
        for pos in list_pos:
            token = lemmatizer.lemmatize(token, pos)
        lemmatized_text.append(token)
    return ' '.join(lemmatized_text)

# Main preprocessing pipeline
def main_pipeline(raw_text, 
                  no_hashtags=True,
                  hashtag_retain_words=True,
                  no_newlines=True,
                  no_urls=True,
                  no_punctuation=True,
                  print_output=True, 
                  no_stopwords=False,
                  lowercase=True, 
                  lemmatized=True,
                  list_pos=["n", "v", "a", "r", "s"],
                  stemmed=False, 
                  no_emojis = True,
                  **kwargs):
    """Preprocess strings according to the parameters"""

    # Apply splitting words process
    clean_text = ' '.join([splitting_words_process(word) for word in raw_text.split()])
    
    # Replace variants of 'good'
    clean_text = replace_gud_with_good(clean_text)

    # Replace variants of 'ok'
    clean_text = replace_to_ok(clean_text)

    # Add space after punctuation
    clean_text = add_space_after_punctuation(clean_text)
    
    # Clean text using regex
    clean_text = regex_cleaner(clean_text, 
                               no_hashtags=no_hashtags,
                               hashtag_retain_words=hashtag_retain_words,
                               no_newlines=no_newlines,
                               no_urls=no_urls,
                               no_punctuation=no_punctuation,
                               no_emojis=no_emojis,
                               **kwargs)
    
    clean_text = re.sub(r"'m", " am", clean_text)
    clean_text = re.sub(r"n't", " not", clean_text)
    clean_text = re.sub(r"'s", " is", clean_text)
    clean_text = re.sub(r'\b\s{2,}\b', ' ', clean_text)
    # Remove space before punctuation marks
    clean_text = re.sub(r'\s+([.,!])', r'\1', clean_text)
    #clean_text = re.sub(r"\.(?=\w)", ". ", clean_text)  # Ensure space after period

    if lemmatized == True:
        clean_text = lemmatize_all(clean_text,["n", "v", "a", "r", "s"])

    if stemmed == True:
        porterstemmer = nltk.stem.PorterStemmer()
        clean_text = ' '.join([porterstemmer.stem(token) for token in clean_text.split()])
 
    if lowercase == True:
        clean_text = clean_text.lower()

    if no_stopwords == True:
        stopwords = nltk.corpus.stopwords.words('english')
        clean_text = ' '.join([word for word in clean_text.split() if word not in stopwords])
        
    if print_output == True:
        print(raw_text)
        print(clean_text)
    
    return clean_text
