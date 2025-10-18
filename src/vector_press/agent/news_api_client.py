from abc import ABC, abstractmethod

from typing import Dict
#from datetime import datetime
import requests


from config import settings

# TODO if we develop mcp server it will save us from typing different api formats for every news sources
def _extract_article_text(article_data: Dict) -> Dict | None:
    """
    Extract and clean text from Guardian API article response

    Args:
        article_data: Article data dictionary from Guardian API response

    Returns:
        Dictionary containing:
            - metadata: Article metadata (ID, title, publication date, etc.)
            - content: Combined full text content
        Returns None if extraction fails
    """

    try:
        # Basic article info
        source = article_data.get("webUrl", "")
        publication_date = article_data.get("webPublicationDate", "")
        # Extract fields if available
        fields = article_data.get("fields", {})  #we are getting whole fields dict from article_data
        body_text = fields.get("bodyText", "")

        # Combine all text content
        # TODO ask BBB can we type it more elegant way
        full_text = {
            "source" : source,
            "publication_date" : publication_date,
            "body_text" : body_text,
        }

        # Create structured metadata
        meta_data = {
            #"article_id": article_id,
            # Guardian API ID as article_id (e.g., "world/2022/oct/21/russia-ukraine-war-latest...")
            #"title": title,
            #"section": section_name,
            #"publication_date": publication_date,
            #"url": url,
            #"summary": standfirst,
            #"body_text": body_text,
            #"trail_text": trail_text,
            #"word_count": word_count,
            #"char_count": char_count,
            #"fetch_time": datetime.now().isoformat()
        }
        # TODO this is where i left, i have returned dict but, I didn't fix after that method which is at line 111
        return full_text
            #'metadata': meta_data

    except Exception as e:
        print(f"🔥 [DEBUG] Error extracting article text: {e}")
        return None

class BaseNewsAPIClient(ABC):   # ABC = Abstract Base Class, ABC prevents creating instances of incomplete classes and forces subclasses to implement all required abstract methods.
    def __init__(self, api_key: str, base_url: str):
        self._api_key = api_key  # attributes names
        self._base_url = base_url

    @abstractmethod
    def search(self, validation) -> Dict:
        pass

    #@abstractmethod
    #def _extract_article_text(self, article_data: Dict) -> Dict:
      #  """ Extract and clean text from API source's article response """
      #  pass


class GuardianAPIClient(BaseNewsAPIClient):
    def __init__(self):
        super().__init__(
            api_key=settings.GUARDIAN_API_KEY,  #these are parameter's name, not like base class's attribute names
            base_url="https://content.guardianapis.com"
        )

    def search(self, validation) -> list[dict] | None:
        """
        Search articles using validation object.

        Args:
            validation: GuardianSearchRequest object with search parameters
        """



        endpoint = f"{self._base_url}/search"

        base_params = validation.model_dump()
        base_params["q"] = base_params.pop("query")  #pop removes the key and return its value
        base_params["show-fields"] = base_params.pop("show_fields")
        base_params["page-size"] = base_params.pop("page_size")
        base_params["api-key"] = self._api_key

        max_pages = base_params.pop("max_pages")

        all_extracted_articles = []
        try:
            for page in range(1, max_pages + 1):
                params = {**base_params, "page": page}  #"**base_params" unpacks the dict
                response = requests.get(endpoint, params=params, timeout=1)

                if response.status_code == 200:
                    api_data = response.json()
                    articles_data = api_data.get('response', {}).get('results', [])  #response is dict because of that we are getting it by {}, results is list .......

                    # Process each article using the extraction function
                    for article, article_data in enumerate(articles_data):
                        extracted = _extract_article_text(article_data)  #we will return dict
                        if extracted:
                            all_extracted_articles.append(extracted)
                            print('ossuruk')
                        else:
                            print(f"[DEBUG] Failed to extract article {article + 1} from page {page}")

                else:
                    print(f"❌ [DEBUG] Page {page} failed with status {response.status_code}: {response.text}")
                    if page == 1:  # If first page fails, return None
                        return None
                    else:  # If next page fails, continue with what we have
                        break
            print(all_extracted_articles)
            return all_extracted_articles if all_extracted_articles else None

        except requests.exceptions.RequestException as e:
            print(f"🔥 [DEBUG] Request exception occurred: {e}")
            return None


class NewYorkTimesAPIClient(BaseNewsAPIClient):
    def __init__(self):
        super().__init__(api_key=settings.NEWYORKTIMES_API,
                         base_url= "https://api.nytimes.com/svc/search/v2/articlesearch.json?")


    def search(self, validation) -> list[Dict] | None:

        endpoint = f"{self._base_url}"

        base_params = validation.model_dump()
        base_params['api-key'] = self._api_key

        all_extracted_articles = []

        def _extract_article_text(article_data: Dict) -> Dict:
            """ Extract and clean text from API source's article response """
            return []

        response = requests.get(url=endpoint, params=base_params, timeout=1)
        if response.status_code == 200:
            api_data = response.json()
            article_data = api_data.get('response', {}).get('docs', [])

            for article, article_data in enumerate(article_data):
                web_url = article_data.get('web_url', '')
                response_of_article = requests.get(url=web_url)

                extracted = _extract_article_text(article_data)