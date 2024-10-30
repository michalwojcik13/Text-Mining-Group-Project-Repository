#!pip install transformers datasets

import pandas as pd

restaurants_raw = pd.read_csv(r"data_hyderabad/105_restaurants.csv")
reviews_raw = pd.read_csv(r"data_hyderabad/10k_reviews.csv")

reviews_raw.head(5)
restaurants_raw.head(5)

restaurants_name = restaurants_raw['Name'].tolist()

reviews = reviews_raw['Review'].tolist()

# transformers testing

from transformers import pipeline, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

encoded_input = tokenizer(str(reviews_raw['Review']))

encoded_input

ner = pipeline('ner', aggregation_strategy='simple')


text= "Hello, I just bought food from McDonalds in New York city and it was delicious"

for i in range(10):
    print(ner(reviews_raw['Review'][i]))


#### lemmatization 

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

word = "yummmmmmyyyy"

lemmatized_word = lemmatizer.lemmatize(word)
print(lemmatized_word)


from nltk.corpus import words
import nltk
nltk.download('words')

word == words.words()


### sentiment analysis check - transforemers

from transformers import pipeline
pipeline_sent = pipeline('sentiment-analysis')

words = ['good', 'bad', 'nice', 'ugly', 'awesome', 'terrible', 'wonderful']

for word in words:
    print(pipeline_sent(word))
    
#spellchecker

from spellchecker import SpellChecker
spell = SpellChecker()

text = "yummmy"

corrected_text = spell.correction(text)
print(corrected_text)



### emoticons
"""""
import re
from emot.emo_unicode import UNICODE_EMO, EMOTICONS

def convert_emoticons(text):
 for emot in EMOTICONS:
 text = re.sub(u'('+emot+')', "_".join(EMOTICONS[emot].replace(",","").split()), text)
 return text# Example
text = "Hello :-) :-)"
convert_emoticons(text)
"""""

### Food NER testing

from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
model = AutoModelForTokenClassification.from_pretrained("Dizex/FoodBaseBERT")

pipe = pipeline("ner", model=model, tokenizer=tokenizer)
example = "Today I had burger for dinner"

ner_entity_results = pipe(example)
print(ner_entity_results)


### food NER finder - Roberta - testing
tokenizer_roberta = AutoTokenizer.from_pretrained("Dizex/InstaFoodRoBERTa-NER")
model_roberta = AutoModelForTokenClassification.from_pretrained("Dizex/InstaFoodRoBERTa-NER")

pipe = pipeline("ner", model=model, tokenizer=tokenizer)
example = "Today I had burger for dinner and kebab for breakfast" 
example2 = "We tried the Pizza, Chicken Breast stuffed with spinach, spaghetti followed by deserts."


ner_entity_results = pipe(example2, aggregation_strategy="simple")
print(ner_entity_results)

def convert_entities_to_list(text, entities: list[dict]) -> list[str]:
        ents = []
        for ent in entities:
            e = {"start": ent["start"], "end": ent["end"], "label": ent["entity_group"]}
            if ents and (-1 <= ent["start"] - ents[-1]["end"] <= 1) and (ents[-1]["label"] == e["label"]):
                ents[-1]["end"] = e["end"]
                continue
            ents.append(e)

        return [text[e["start"]:e["end"]] for e in ents]

print(convert_entities_to_list(example2, ner_entity_results))
