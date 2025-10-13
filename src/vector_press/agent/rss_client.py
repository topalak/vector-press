import feedparser
import logging
from abc import ABC
from typing import List, Dict

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from src.vector_press.agent.tools_validation import TechnologyRSSFeed

logger = logging.getLogger(__name__)




class BaseRSSClient(ABC):
    def __init__(self, embedding_model, similarity_threshold: float = 0.5,):
        self.similarity_threshold = similarity_threshold
        self.embedding_model = embedding_model

        logger.info('Embedding model loaded')

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
                    'idx': entry_idx,
                    'source': feed_url,
                })

        return all_entries

    def _embed(self, all_entries: list[dict], validation: TechnologyRSSFeed):
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
        query_array = np.array(query).reshape(1, -1)

        similarities = cosine_similarity(query_array, entries_array)[0]
        print(similarities)
        self.similarity_threshold



        return

    def search(self, feed_urls: list[str], validation : TechnologyRSSFeed): # -> List[Dict]
        all_entries = self._fetch_feed(feed_urls)
        query, entries = self._embed(all_entries=all_entries, validation=validation)
        best_matches = cosine_similarity(query, entries)















class TechnologyRSSClient(BaseRSSClient):
    """Technology based RSS client"""
    #validation_instance = TechnologyRSSFeed(query='Elon musk')

    def __init__(self, embedding_model, similarity_threshold: float = 0.7, ):
        super().__init__(similarity_threshold=similarity_threshold, embedding_model=embedding_model)
        self.feed_url = [
        "https://www.ft.com/technology?format=rss",
        #"https://feeds.bbci.co.uk/news/technology/rss.xml",
    ]
    def fetch_tech_feeds(self, validation: TechnologyRSSFeed) -> list[str]:
        all_entries = self.search(feed_urls = self.feed_url, validation=validation)
        return all_entries
































class SportsRSSClient(BaseRSSClient):
    """Sports based RSS client"""
    feed_urls = [
         "https://sports.yahoo.com/rss/",
         "https://feeds.bbci.co.uk/sport/rss.xml",
    ]

    def __init__(self, embedding_model, similarity_threshold: float = 0.5):
        super().__init__(similarity_threshold = similarity_threshold,embedding_model = embedding_model)


def main():
    from src.vector_press.model_config import ModelConfig
    from config import settings
    embedding_model_config = ModelConfig(model="all-minilm:33m", model_provider_url=settings.OLLAMA_HOST)
    embedding_model = embedding_model_config.get_embedding()
    rss = TechnologyRSSClient(embedding_model=embedding_model)
    rss.fetch_tech_feeds(validation=TechnologyRSSFeed(query='Elon musk'))


    return


if __name__ == '__main__':
    main()