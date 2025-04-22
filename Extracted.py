import os
import re
import time
import pandas as pd
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from typing import Tuple, Dict, Set
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure NLTK
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Constants
STOPWORDS_DIR = 'StopWords'
MASTER_DICT_DIR = 'MasterDictionary'
OUTPUT_DIR = 'Extracted_Articles'
OUTPUT_FILE = 'Output_Data_Structure.xlsx'
MAX_RETRIES = 3
REQUEST_TIMEOUT = 10


class TextAnalyzer:
    def __init__(self):
        self.stopwords = self._load_stopwords()
        self.positive_words, self.negative_words = self._load_master_dictionary()
        self.session = self._configure_session()

    def _configure_session(self) -> requests.Session:
        """Configure requests session with retry logic"""
        session = requests.Session()
        retries = Retry(
            total=MAX_RETRIES,
            backoff_factor=1,
            status_forcelist=[500, 502, 503, 504]
        )
        session.mount('https://', HTTPAdapter(max_retries=retries))
        return session

    def _load_stopwords(self) -> Set[str]:
        """Load stopwords from directory"""
        stopwords_set = set()
        if not os.path.exists(STOPWORDS_DIR):
            raise FileNotFoundError(f"StopWords directory not found at {STOPWORDS_DIR}")
            
        for filename in os.listdir(STOPWORDS_DIR):
            if filename.endswith('.txt'):
                try:
                    with open(os.path.join(STOPWORDS_DIR, filename), 'r', encoding='ISO-8859-1') as f:
                        stopwords_set.update(word.strip().lower() for word in f if word.strip())
                except Exception as e:
                    print(f"Error loading stopwords file {filename}: {e}")
        return stopwords_set

    def _load_master_dictionary(self) -> Tuple[Set[str], Set[str]]:
        """Load positive and negative words from dictionary"""
        positive_words = set()
        negative_words = set()
        
        if not os.path.exists(MASTER_DICT_DIR):
            raise FileNotFoundError(f"MasterDictionary directory not found at {MASTER_DICT_DIR}")
        
        try:
            with open(os.path.join(MASTER_DICT_DIR, 'positive-words.txt'), 'r', encoding='ISO-8859-1') as f:
                positive_words.update(word.strip().lower() for word in f if word.strip())
        except Exception as e:
            print(f"Error loading positive words: {e}")
        
        try:
            with open(os.path.join(MASTER_DICT_DIR, 'negative-words.txt'), 'r', encoding='ISO-8859-1') as f:
                negative_words.update(word.strip().lower() for word in f if word.strip())
        except Exception as e:
            print(f"Error loading negative words: {e}")
            
        return positive_words, negative_words

    def _clean_text(self, text: str) -> list:
        """Tokenize and clean text"""
        words = word_tokenize(text.lower())
        return [word for word in words if word.isalpha() and word not in self.stopwords]

    @staticmethod
    def _count_syllables(word: str) -> int:
        """Count syllables in a word with edge case handling"""
        word = word.lower()
        if len(word) <= 3:
            return 1
            
        # Handle common exceptions
        if word.endswith(('es', 'ed')) and len(word) > 2:
            word = word[:-2]
            
        vowel_count = 0
        prev_char_was_vowel = False
        for char in word:
            if char in 'aeiouy':
                if not prev_char_was_vowel:
                    vowel_count += 1
                    prev_char_was_vowel = True
            else:
                prev_char_was_vowel = False
        return max(1, vowel_count)

    @staticmethod
    def _count_personal_pronouns(text: str) -> int:
        """Count personal pronouns while excluding 'US' (country)"""
        pattern = r'\b(I|we|my|ours|us)(?![A-Za-z])'
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        return len(matches)

    def extract_article(self, url: str) -> Tuple[str, str]:
        """Extract article title and text from URL"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = self.session.get(
                url, 
                headers=headers, 
                timeout=REQUEST_TIMEOUT
            )
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')

            # Extract title
            title = soup.find('h1')
            if not title:
                title = soup.find('title')
            title_text = title.get_text().strip() if title else "No Title Found"

            # Extract main content
            article = soup.find('article') or \
                     soup.find('div', class_=re.compile('content|article|main|post', re.I)) or \
                     soup.find('main') or \
                     soup

            # Remove unwanted elements
            for element in article.find_all(['header', 'footer', 'nav', 'aside', 'script', 'style', 'form', 'iframe']):
                element.decompose()

            # Extract paragraphs
            paragraphs = article.find_all(['p', 'h2', 'h3', 'li'])
            article_text = ' '.join(p.get_text(' ', strip=True) for p in paragraphs)
            
            return title_text, article_text.strip()
            
        except Exception as e:
            print(f"Error extracting {url}: {str(e)}")
            return "Extraction Failed", ""

    def analyze_text(self, text: str) -> Dict[str, float]:
        """Perform all required text analysis"""
        if not text or text == "Extraction Failed":
            return {key: 0 for key in [
                'POSITIVE SCORE', 'NEGATIVE SCORE', 'POLARITY SCORE', 
                'SUBJECTIVITY SCORE', 'AVG SENTENCE LENGTH', 
                'PERCENTAGE OF COMPLEX WORDS', 'FOG INDEX',
                'AVG NUMBER OF WORDS PER SENTENCE', 'COMPLEX WORD COUNT',
                'WORD COUNT', 'SYLLABLE PER WORD', 'PERSONAL PRONOUNS',
                'AVG WORD LENGTH'
            ]}
            
        # Tokenization and cleaning
        cleaned_words = self._clean_text(text)
        word_count = len(cleaned_words)
        
        # Sentiment analysis
        positive_score = sum(1 for word in cleaned_words if word in self.positive_words)
        negative_score = sum(1 for word in cleaned_words if word in self.negative_words)
        
        # Calculate derived variables
        polarity_score = (positive_score - negative_score) / ((positive_score + negative_score) + 1e-6)
        subjectivity_score = (positive_score + negative_score) / (word_count + 1e-6)
        
        # Readability analysis
        sentences = sent_tokenize(text)
        sentence_count = len(sentences)
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
        
        # Complex word analysis
        complex_words = [word for word in cleaned_words if self._count_syllables(word) > 2]
        complex_word_count = len(complex_words)
        percentage_complex = (complex_word_count / word_count * 100) if word_count > 0 else 0
        fog_index = 0.4 * (avg_sentence_length + percentage_complex)
        
        # Syllable and word analysis
        syllable_counts = [self._count_syllables(word) for word in cleaned_words]
        avg_syllables = sum(syllable_counts) / word_count if word_count > 0 else 0
        personal_pronouns = self._count_personal_pronouns(text)
        avg_word_length = sum(len(word) for word in cleaned_words) / word_count if word_count > 0 else 0
        
        return {
            'POSITIVE SCORE': positive_score,
            'NEGATIVE SCORE': negative_score,
            'POLARITY SCORE': round(polarity_score, 3),
            'SUBJECTIVITY SCORE': round(subjectivity_score, 3),
            'AVG SENTENCE LENGTH': round(avg_sentence_length, 2),
            'PERCENTAGE OF COMPLEX WORDS': round(percentage_complex, 2),
            'FOG INDEX': round(fog_index, 2),
            'AVG NUMBER OF WORDS PER SENTENCE': round(avg_sentence_length, 2),
            'COMPLEX WORD COUNT': complex_word_count,
            'WORD COUNT': word_count,
            'SYLLABLE PER WORD': round(avg_syllables, 2),
            'PERSONAL PRONOUNS': personal_pronouns,
            'AVG WORD LENGTH': round(avg_word_length, 2)
        }


def save_article(url_id: str, content: str) -> None:
    """Save extracted article to file"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    file_path = os.path.join(OUTPUT_DIR, f"{url_id}.txt")
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
    except Exception as e:
        print(f"Error saving article {url_id}: {e}")


def main():
    try:
        # Initialize analyzer
        analyzer = TextAnalyzer()
        
        # Load input data
        if not os.path.exists('Input.xlsx'):
            raise FileNotFoundError("Input.xlsx not found")
            
        input_df = pd.read_excel('Input.xlsx')
        
        # Prepare output dataframe
        output_columns = [
            'URL_ID', 'URL', 'POSITIVE SCORE', 'NEGATIVE SCORE', 'POLARITY SCORE',
            'SUBJECTIVITY SCORE', 'AVG SENTENCE LENGTH', 'PERCENTAGE OF COMPLEX WORDS',
            'FOG INDEX', 'AVG NUMBER OF WORDS PER SENTENCE', 'COMPLEX WORD COUNT',
            'WORD COUNT', 'SYLLABLE PER WORD', 'PERSONAL PRONOUNS', 'AVG WORD LENGTH'
        ]
        output_df = pd.DataFrame(columns=output_columns)
        
        # Process each URL
        total_urls = len(input_df)
        for idx, row in input_df.iterrows():
            url_id = row['URL_ID']
            url = row['URL']
            
            print(f"Processing {idx+1}/{total_urls}: {url_id} - {url}")
            
            # Extract article
            title, article_text = analyzer.extract_article(url)
            save_article(url_id, f"{title}\n\n{article_text}")
            
            # Analyze text if extraction succeeded
            if article_text and article_text != "Extraction Failed":
                analysis = analyzer.analyze_text(article_text)
                output_row = {'URL_ID': url_id, 'URL': url, **analysis}
                output_df = pd.concat([output_df, pd.DataFrame([output_row])], ignore_index=True)
            else:
                print(f"Skipping analysis for {url_id} due to extraction failure")
        
        # Save results
        output_df.to_excel(OUTPUT_FILE, index=False)
        print(f"\nAnalysis complete. Results saved to {OUTPUT_FILE}")
        print(f"Extracted articles saved in {OUTPUT_DIR} directory")
        
    except Exception as e:
        print(f"\nError in main execution: {e}")
        raise


if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"\nExecution time: {time.time() - start_time:.2f} seconds")