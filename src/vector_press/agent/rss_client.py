import feedparser
import logging
from abc import ABC, abstractmethod

from .tools_validation import TechnologyRSSFeed

logger = logging.getLogger(__name__)

class BaseRSSClient(ABC):
    def __init__(self, embedding_model, similarity_threshold: float = 0.5,):
        self.similarity_threshold = similarity_threshold
        self.embedding_model = embedding_model

        logger.info('Embedding model loaded')

    def _fetch_feed(self, feed_url: dict, validation): # -> List[Dict]:    #  When you define a method in the base class at the class level (not inside __init__), it's automatically available to all subclasses.
        base_params = validation.model_dump()
        query = base_params['query']

        for page in feed_url:
            feed = feedparser.parse(page)
            if feed.bozo:
                logger.warning(f"Bozo failed to fetch feed {feed_url} : {feed.bozo_exception}")

            embeddings = []
        return feed


class TechnologyRSSClient(BaseRSSClient):
    """Technology based RSS client"""
    feed_urls = [
        "https://www.ft.com/technology?format=rss",
        "https://feeds.bbci.co.uk/news/technology/rss.xml",
    ]

    def __init__(self, embedding_model, similarity_threshold: float = 0.7, ):
        super().__init__(similarity_threshold=similarity_threshold, embedding_model=embedding_model)

    def fetch_tech_feeds(self):
        feed = self._fetch_feed(self.feed_urls, validation=TechnologyRSSFeed(validation))
        return feed


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
    rss.fetch_tech_feeds()


    return


if __name__ == '__main__':
    main()