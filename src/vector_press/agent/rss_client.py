import feedparser
import logging
from abc import ABC, abstractmethod
from typing import List, Dict
from config import settings

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import requests
from bs4 import BeautifulSoup

from src.vector_press.model_config import ModelConfig
# TechnologyRSSFeedSchema imported lazily in main() to avoid circular import

logger = logging.getLogger(__name__)

embedding_model_config = ModelConfig(model="embeddinggemma", model_provider_url=settings.OLLAMA_HOST, num_ctx=2)
class BaseRSSClient(ABC):
    def __init__(self, embedding:bool, similarity_threshold:float):
        if embedding:
            self.embedding_model = embedding_model_config.get_embedding()
        self.similarity_threshold = similarity_threshold

    @abstractmethod
    def search(self, validation) -> list[Dict]:
        pass

    @staticmethod
    def _fetch_feed(feed_urls: list[str]) -> List[Dict]:  # When you define a method in the base class at the class level (not inside __init__), it's automatically available to all subclasses.
        all_entries = []

        for feed_idx, feed_url in enumerate(feed_urls):
            feed = feedparser.parse(feed_url)
            if feed.bozo:
                logger.warning(f"Bozo failed to fetch feed {feed_url} : {feed.bozo_exception}")

            for entry_idx, entry in enumerate(feed.entries):
                all_entries.append({
                    'link': entry.link,
                    'title': entry.get('title', ''),
                    #'description': entry.get('summary', ''),
                    'published': entry.get('published', ''),
                    #'feed_url': feed_url,
                })
        return all_entries

    def _search(self, feed_urls: list[str], validation) -> list[Dict]:
        """
        Search RSS feeds and return full article contents above similarity threshold.

        Args:
            feed_urls: List of RSS feed URLs to search
            validation: Validated RSS feed parameters containing the query

        Returns:
            List of full article text contents for items above threshold
        """

        base_params = validation.model_dump()

        if embedding:
            query = base_params["query"]

            all_entries = self._fetch_feed(feed_urls)
            query, entries = self._embed(all_entries=all_entries, query=query)
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

        else:

            all_topics = self._fetch_feed(feed_urls)

            logger.info(f"Retrieved {len(all_topics)} entries from {len(feed_urls)} feed(s)")

            return all_topics

    def _embed(self, all_entries: list[dict],query:str):
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

class TechnologyRSSClient(BaseRSSClient):
    """Technology based RSS client"""

    def __init__(self):
        super().__init__(embedding=False, similarity_threshold=0.7)
        self.feed_url = [
        "https://feeds.bbci.co.uk/news/technology/rss.xml",
    ]
    def search(self, validation) -> list[Dict]:
        """
        Search technology RSS feeds for articles.

        Args:
            validation: Validated RSS feed parameters containing the query

        Returns:
            List of dictionaries containing link, title, description, published date, and feed_url
        """
        result = self._search(feed_urls = self.feed_url, validation=validation)
        return result


class SportsRSSClient(BaseRSSClient):
    """Sports based RSS client"""

    def __init__(self):
        super().__init__(embedding=False, similarity_threshold=0.7)
        self.feed_url = [
         "https://sports.yahoo.com/rss/",
         "https://feeds.bbci.co.uk/sport/rss.xml",
    ]
    def search(self, validation, embedding:bool) -> list[Dict]:
        """
        Search sports RSS feeds for articles.

        Args:
            embedding: enabling embedding
            validation: Validated RSS feed parameters containing the query

        Returns:
            List of dictionaries containing link, title, description, published date, and feed_url
        """

        result = self._search(feed_urls = self.feed_url, validation=validation, embedding=embedding)
        return result

def main():
    # Lazy import to avoid circular dependency
    from src.vector_press.agent.tools import TechnologyRSSFeedSchema

    rss = TechnologyRSSClient()

    a = rss.search(validation=TechnologyRSSFeedSchema(query='cyber security', do_we_look_for_a_topic=True))

    print(a)

    return


#TODO timestampleri alip current time dan 24 saat cikarip almamiz gerekiyor veya 7 gun, user ne istiyorsa bunu schema olarak vermemiz lazim
# todo title, description, timestamp (current datetime), category = "tech" or "sport" these will pass as default inside the related rss method and link (to search it later if user specifically specifies it)


if __name__ == '__main__':
    main()