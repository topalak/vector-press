import feedparser
import logging
from abc import ABC
from typing import List, Dict

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import requests
from bs4 import BeautifulSoup


logger = logging.getLogger(__name__)


class BaseRSSClient(ABC):
    def __init__(self, embedding_model, similarity_threshold: float = 0.5,):
        self.similarity_threshold = similarity_threshold
        self.embedding_model = embedding_model

    @staticmethod
    def _fetch_feed(feed_urls: list[str]) -> List[Dict]:  # When you define a method in the base class at the class level (not inside __init__), it's automatically available to all subclasses.
        all_entries = []

        for feed_idx, feed_url in enumerate(feed_urls):
            feed = feedparser.parse(feed_url)
            if feed.bozo:
                logger.warning(f"Bozo failed to fetch feed {feed_url} : {feed.bozo_exception}")

            for entry_idx, entry in enumerate(feed.entries):
                text = f"{entry['title']} {entry['summary']}"
                all_entries.append({
                    'title_and_summary': text,
                    'link': entry.link,
                })
        return all_entries

    def _embed(self, all_entries: list[dict], validation):
        base_parms = validation.model_dump()
        query = base_parms['query']
        query_embedding = self.embedding_model.embed_query(query)   #we changed the variable name because of these reasons
        '''  Reasons:
              - Type confusion: query starts as a str but becomes a numpy.ndarray or list[float]. This is confusing.
              - Debugging: If you need to log/debug, you've lost the original query text
              - Clarity: query_embedding clearly indicates it's an embedding vector, not text
              - Maintainability: Future developers (including you) will understand the code better  '''

        entry_text = [item['title_and_summary'] for item in all_entries]
        embeddings = self.embedding_model.embed_documents(entry_text)


        return query_embedding, embeddings

    def _cosine_similarity(self, entries : list, query : list):
        entries_array = np.array(entries)
        query_array = np.array(query).reshape(1, -1)  #we are creating 2d matrix

        similarities = cosine_similarity(query_array, entries_array)[0]
        mask = similarities >= self.similarity_threshold
        filtered_indices = np.where(mask)[0]
        filtered_scores = similarities[mask]

        sorted_order = np.argsort(filtered_scores)[::-1]

        '''
          return filtered_indices[sorted_order], filtered_scores[sorted_order]
          return (filtered_indices[sorted_order], filtered_scores[sorted_order])  These are same, returns tuple
          if we define  best_matches = self._cosine_similarity(entries=entries,query=query)  it returns tuple
        '''
        return filtered_indices[sorted_order], filtered_scores[sorted_order]

    @staticmethod
    def _fetch_article_content(url: str) -> str:
        """
        Fetch full article content from URL.

        Args:
            url: Article URL to fetch

        Returns:
            Full article text content, empty string if fetching fails

        Raises:
            RequestException: If HTTP request fails
        """
        try:
            response = requests.get(url, timeout=10, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                              'AppleWebKit/537.36'
            })
            response.raise_for_status()

            soup = BeautifulSoup(response.content, features="html.parser")

            # Get raw text content from soup
            text = soup.text
            return text

        except requests.RequestException as e:
            logger.warning(f"Failed to fetch article from {url}: {e}")
            return ""

    def _search(self, feed_urls: list[str], validation) -> list[str]:
        """
        Search RSS feeds and return full article contents above similarity threshold.

        Args:
            feed_urls: List of RSS feed URLs to search
            validation: Validated RSS feed parameters containing the query

        Returns:
            List of full article text contents for items above threshold
        """
        all_entries = self._fetch_feed(feed_urls)
        query, entries = self._embed(all_entries=all_entries, validation=validation)
        filtered_indices, filtered_scores = self._cosine_similarity(entries=entries, query=query)

        article_contents = []

        for position, score in zip(filtered_indices, filtered_scores):
            entry = all_entries[position]

            # Fetch full article content from URL
            full_content = self._fetch_article_content(entry['link'])

            if full_content:
                article_contents.append(full_content)
                logger.info(f"Successfully fetched article from {entry['link']}")
            else:
                logger.warning(f"Skipping article due to fetch failure: {entry['link']}")

        logger.info(
            f"Retrieved {len(article_contents)} articles out of "
            f"{len(filtered_indices)} matches above threshold"
        )

        return article_contents

class TechnologyRSSClient(BaseRSSClient):
    """Technology based RSS client"""
    #validation_instance = TechnologyRSSFeed(query='Elon musk')

    def __init__(self, embedding_model, similarity_threshold: float = 0.35, ):
        super().__init__(similarity_threshold=similarity_threshold, embedding_model=embedding_model)
        self.feed_url = [
        "https://feeds.bbci.co.uk/news/technology/rss.xml",
    ]
    def search(self, validation) -> list[str]:
        result = self._search(feed_urls = self.feed_url, validation=validation)
        return result


class SportsRSSClient(BaseRSSClient):
    """Sports based RSS client"""

    def __init__(self, embedding_model, similarity_threshold: float = 0.5):
        super().__init__(similarity_threshold = similarity_threshold,embedding_model = embedding_model)
        self.feed_url = [
         "https://sports.yahoo.com/rss/",
         "https://feeds.bbci.co.uk/sport/rss.xml",
    ]

    def search(self, validation) -> list[str]:
        result = self._search(feed_urls = self.feed_url, validation=validation)
        return result

def main():
    from src.vector_press.model_config import ModelConfig
    from config import settings
    embedding_model_config = ModelConfig(model="all-minilm:33m", model_provider_url=settings.OLLAMA_HOST)
    embedding_model = embedding_model_config.get_embedding()
    rss = TechnologyRSSClient(embedding_model=embedding_model)
    #a = rss.search(validation=TechnologyRSSFeed(query='cyber security'))
    #print(a)

    return


if __name__ == '__main__':
    main()