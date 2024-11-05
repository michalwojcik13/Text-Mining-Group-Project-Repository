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
                  no_punctuation):
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
def add_space_after_punctuation(text):
    # Aggiungi uno spazio dopo i punti, virgole, punti esclamativi, punti interrogativi, ecc.
    text = re.sub(r'([.!?,;:])(?=\S)', r'\1 ', text)  # Aggiungi spazio se non c'è uno spazio dopo la punteggiatura
    return text

# Function to lemmatize all tokens
def lemmatize_all(text, list_pos=["n", "v", "a", "r", "s"]):
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
                  custom_stopwords=[],
                  lowercase=True, 
                  lemmatized=True,
                  stemmed=False, 
                  pos_tags_list="no_pos",
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
                               **kwargs)
    
    clean_text = re.sub(r"'m", " am", clean_text)
    clean_text = re.sub(r"n't", " not", clean_text)
    clean_text = re.sub(r"'s", " is", clean_text)
    clean_text = re.sub(r"\.(?=\w)", ". ", clean_text)  # Ensure space after period

    if lemmatized == True:
        clean_text = lemmatize_all(clean_text)

    if stemmed == True:
        porterstemmer = nltk.stem.PorterStemmer()
        clean_text = ' '.join([porterstemmer.stem(token) for token in clean_text.split()])
 
    if lowercase == True:
        clean_text = clean_text.lower()

    if print_output == True:
        print(raw_text)
        print(clean_text)
    
    return clean_text

# Load dataset and limit to first 30 rows for testing
reviews_raw = pd.read_csv("data_hyderabad/10k_reviews.csv").head(30)

# Apply preprocessing pipeline to dataset
reviews_raw['Cleaned_Review'] = reviews_raw['Review'].apply(lambda x: main_pipeline(x, print_output=False))

# Create a dataset with only the original and cleaned reviews
output_dataset = reviews_raw[['Review', 'Cleaned_Review']]

# Display the dataset in a tabular format
pd.set_option('display.max_colwidth', None)
print(output_dataset)

