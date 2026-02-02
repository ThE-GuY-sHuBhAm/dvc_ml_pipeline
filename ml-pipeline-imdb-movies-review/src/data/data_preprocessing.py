import __main__
import pandas as pd
import numpy as np
import os
from nltk.stem import WordNetLemmatizer
import yaml
import logging
import re
import string

logger = logging.getLogger('data_preprocessing')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('preprocessing.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def lemmatization(texts):
    """Lemmatize the text."""
    lemmatizer = WordNetLemmatizer()
    text = texts.split()
    text = [lemmatizer.lemmatize(word) for word in text]
    return " ".join(text)

def remove_stop_words(texts):
    """Remove stop words from the text."""
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))
    text = texts.split()
    text = [word for word in text if word.lower() not in stop_words]
    return " ".join(text)

def lower_case(text):
    """Convert text to lower case."""
    text = text.split()
    text = [word.lower() for word in text]
    return " ".join(text)

def removing_punctuations(text):
    """Remove punctuations from the text."""
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    text = text.replace('؛', "")
    text = re.sub('\s+', ' ', text).strip()
    return text

def removing_urls(text):
    """Remove URLs from the text."""
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def remove_html_tags(text):
    """Remove HTML tags from the text."""
    html_pattern = re.compile(r'<.*?>')
    return html_pattern.sub(r'', text)

def remove_small_sentences(df):
    """Remove sentences with less than 3 words."""
    for i in range(len(df)):
        if len(df.text.iloc[i].split()) < 3:
            df.text.iloc[i] = np.nan


def remove_chatwords(text):
    # chat word treatment
    chat_words = {
    "A3": "Anytime, Anywhere, Anyplace",
    "ADIH": "Another Day In Hell",
    "AFK": "Away From Keyboard",
    "AFAIK": "As Far As I Know",
    "ASAP": "As Soon As Possible",
    "ASL": "Age, Sex, Location",
    "ATK": "At The Keyboard",
    "ATM": "At The Moment",
    "BAE": "Before Anyone Else",
    "BAK": "Back At Keyboard",
    "BBL": "Be Back Later",
    "BBS": "Be Back Soon",
    "BFN": "Bye For Now",
    "B4N": "Bye For Now",
    "BRB": "Be Right Back",
    "BRUH": "Bro",
    "BRT": "Be Right There",
    "BSAAW": "Big Smile And A Wink",
    "BTW": "By The Way",
    "BWL": "Bursting With Laughter",
    "CSL": "Can’t Stop Laughing",
    "CU": "See You",
    "CUL8R": "See You Later",
    "CYA": "See You",
    "DM": "Direct Message",
    "FAQ": "Frequently Asked Questions",
    "FC": "Fingers Crossed",
    "FIMH": "Forever In My Heart",
    "FOMO": "Fear Of Missing Out",
    "FR": "For Real",
    "FWIW": "For What It's Worth",
    "FYP": "For You Page",
    "FYI": "For Your Information",
    "G9": "Genius",
    "GAL": "Get A Life",
    "GG": "Good Game",
    "GMTA": "Great Minds Think Alike",
    "GN": "Good Night",
    "GOAT": "Greatest Of All Time",
    "GR8": "Great!",
    "HBD": "Happy Birthday",
    "IC": "I See",
    "ICQ": "I Seek You",
    "IDC": "I Don’t Care",
    "IDK": "I Don't Know",
    "IFYP": "I Feel Your Pain",
    "ILU": "I Love You",
    "ILY": "I Love You",
    "IMHO": "In My Honest/Humble Opinion",
    "IMU": "I Miss You",
    "IMO": "In My Opinion",
    "IOW": "In Other Words",
    "IRL": "In Real Life",
    "IYKYK": "If You Know, You Know",
    "JK": "Just Kidding",
    "KISS": "Keep It Simple, Stupid",
    "L": "Loss",
    "L8R": "Later",
    "LDR": "Long Distance Relationship",
    "LMK": "Let Me Know",
    "LMAO": "Laughing My A** Off",
    "LOL": "Laughing Out Loud",
    "LTNS": "Long Time No See",
    "M8": "Mate",
    "MFW": "My Face When",
    "MID": "Mediocre",
    "MRW": "My Reaction When",
    "MTE": "My Thoughts Exactly",
    "NVM": "Never Mind",
    "NRN": "No Reply Necessary",
    "NPC": "Non-Player Character",
    "OIC": "Oh I See",
    "OP": "Overpowered",
    "PITA": "Pain In The A**",
    "POV": "Point Of View",
    "PRT": "Party",
    "PRW": "Parents Are Watching",
    "ROFL": "Rolling On The Floor Laughing",
    "ROFLOL": "Rolling On The Floor Laughing Out Loud",
    "ROTFLMAO": "Rolling On The Floor Laughing My A** Off",
    "RN": "Right Now",
    "SK8": "Skate",
    "STATS": "Your Sex And Age",
    "SUS": "Suspicious",
    "TBH": "To Be Honest",
    "TFW": "That Feeling When",
    "THX": "Thank You",
    "TIME": "Tears In My Eyes",
    "TLDR": "Too Long, Didn’t Read",
    "TNTL": "Trying Not To Laugh",
    "TTFN": "Ta-Ta For Now!",
    "TTYL": "Talk To You Later",
    "U": "You",
    "U2": "You Too",
    "U4E": "Yours For Ever",
    "W": "Win",
    "W8": "Wait...",
    "WB": "Welcome Back",
    "WTF": "What The F**k",
    "WTG": "Way To Go!",
    "WUF": "Where Are You From?",
    "WYD": "What You Doing?",
    "WYWH": "Wish You Were Here",
    "ZZZ": "Sleeping, Bored, Tired"
    }
    new_text = []
    for w in text.split():
        if w.upper() in chat_words:
            new_text.append(chat_words[w.upper()])
        else:
            new_text.append(w)
    return " ".join(new_text)

def removing_numbers(text):
    """Remove numbers from the text."""
    text = ''.join([char for char in text if not char.isdigit()])
    return text



def normalize_text(df):
    """Apply all preprocessing functions to the dataframe."""
    try:
        logger.info("Starting text normalization")
        df['review'] = df['review'].apply(lower_case)
        logger.debug('converted to lower case')
        df['review'] = df['review'].apply(remove_html_tags)
        logger.debug('removed html tags') 
        df['review'] = df['review'].apply(removing_urls)
        logger.debug('removed urls') 
        df['review'] = df['review'].apply(removing_punctuations)
        logger.debug('removed punctuations')
        df['review'] = df['review'].apply(removing_numbers)
        logger.debug('removed numbers')
        df['review'] = df['review'].apply(remove_chatwords)
        logger.debug('removed chatwords')
        df['review'] = df['review'].apply(remove_stop_words)
        logger.debug('removed stop words')
        df['review'] = df['review'].apply(lemmatization)
        logger.debug('lemmatization performed')
        logger.debug('Text normalization completed')
        return df
    except Exception as e:
        logger.error(f"Error in text normalization: {e}")
        raise


def main():
    try:
        #Fetch the data from raw folder
        train_data = pd.read_csv('ml-pipeline-imdb-movies-review/data/raw/train.csv')
        test_data = pd.read_csv('ml-pipeline-imdb-movies-review/data/raw/test.csv')

        #Normalize the text data
        train_processed_data = normalize_text(train_data)
        test_processed_data = normalize_text(test_data)

        #Store the data inside processed folder
        data_path = os.path.join('ml-pipeline-imdb-movies-review', 'data', 'interim')
        os.makedirs(data_path, exist_ok=True)


        train_processed_data.to_csv(os.path.join(data_path,'train_processed.csv'), index=False)
        test_processed_data.to_csv(os.path.join(data_path,'test_processed.csv'), index=False)

    except Exception as e:
        logger.error(f"Error in data preprocessing pipeline: {e}")
        raise


if __name__ == "__main__":
    main()
    