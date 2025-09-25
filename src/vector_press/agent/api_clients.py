from abc import ABC, abstractmethod

from typing import Dict
from datetime import datetime
import time
import requests

from config import settings


def extract_article_text(article_data: Dict) -> Dict | None:
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
    print(f"\n🔍 [DEBUG] Extracting text from article...")

    import time
    start_time = time.time()

    try:
        # Basic article info
        article_id = article_data.get("id", "")
        title = article_data.get("webTitle", "")
        url = article_data.get("webUrl", "")
        publication_date = article_data.get("webPublicationDate", "")
        section_name = article_data.get("sectionName", "")

        print(f"🔍 [DEBUG] Article ID: {article_id}")

        # Extract fields if available
        fields = article_data.get("fields", {})

        # Get different text content - convert strings to integers
        word_count = int(fields.get("wordcount", "0") or 0)
        char_count = int(fields.get("charCount", "0") or 0)
        standfirst = fields.get("standfirst", "")  # Summary/subtitle
        body_text = fields.get("bodyText", "")
        trail_text = fields.get("trailText", "")  # Preview text

        print(f"🔍 [DEBUG] Standfirst length: {len(standfirst)} chars")
        print(f"🔍 [DEBUG] Body text length: {len(body_text)} chars")
        print(f"🔍 [DEBUG] Trail text length: {len(trail_text)} chars")

        # Combine all text content
        full_text_parts = []
        if title:
            full_text_parts.append(title)
        if standfirst:
            full_text_parts.append(standfirst)
        if body_text:
            full_text_parts.append(body_text)
        elif trail_text:  # Fallback if no body text
            full_text_parts.append(trail_text)

        full_text = "\n\n".join(full_text_parts)

        print(f"🔍 [DEBUG] Combined text length: {len(full_text)} chars")
        print(f"🔍 [DEBUG] Text preview (first 200 chars): {full_text[:200]}...")

        # Create structured metadata
        meta_data = {
            "article_id": article_id,
            # Guardian API ID as article_id (e.g., "world/2022/oct/21/russia-ukraine-war-latest...")
            "title": title,
            "section": section_name,
            "publication_date": publication_date,
            "url": url,
            "summary": standfirst,
            "body_text": body_text,
            "trail_text": trail_text,
            "word_count": word_count,
            "char_count": char_count,
            "fetch_time": datetime.now().isoformat()
        }

        print(f"✅ [DEBUG] Article extraction completed!")
        print(f"✅ [DEBUG] Final word count: {meta_data['word_count']} words")
        print(f"⏱️ [DEBUG] extract_article_text took {time.time() - start_time:.4f} seconds")

        return {
            'metadata': meta_data,
            'content': full_text
        }

    except Exception as e:
        print(f"🔥 [DEBUG] Error extracting article text: {e}")
        return None

class BaseAPIClient(ABC):   # ABC = Abstract Base Class, ABC prevents creating instances of incomplete classes and forces subclasses to implement all required abstract methods.
    def __init__(self, api_key:str, base_url:str):
        self.api_key = api_key
        self.base_url = base_url

    @abstractmethod
    def search_articles(self) -> Dict:
        pass

class GuardianAPIClient(BaseAPIClient):
    def __init__(self):
        super().__init__(
            api_key=settings.GUARDIAN_API_KEY,
            base_url="https://content.guardianapis.com"
        )

        print(f"🔧 [DEBUG] Guardian API Client initialized")

    def search_articles(self,
                        query: str = None,
                        section: str = None,
                        page_size: int = 200,
                        from_date : str = None,
                        show_fields: str = "all",
                        order_by: str = None,
                        max_pages: int = 20) -> list[Dict] | None:

        print(f"\n📡 [DEBUG] Starting API search for {max_pages} page(s)...")

        # Build API endpoint
        endpoint = f"{self.base_url}/search"

        # Build base parameters
        base_params = {
            "api-key": self.api_key
        }

        # Add optional parameters
        if query:
            base_params["q"] = query

        if section:
            base_params["section"] = section

        if from_date:
            base_params["from-date"] = from_date

        if page_size:
            base_params["page-size"] = page_size

        if show_fields:
            base_params["show-fields"] = show_fields

        if order_by:
            base_params["order-by"] = order_by

        if max_pages:
            base_params["max-pages"] = max_pages

        # Collect articles from all pages
        all_extracted_articles = []
        total_start_time = time.time()
        page = 0  # Initialize page counter

        try:
            for page in range(1, max_pages + 1):
                print(f"\n📄 [DEBUG] Fetching page {page}/{max_pages}...")

                # Add page parameter
                params = {**base_params, "page": page}

                page_start_time = time.time()
                response = requests.get(endpoint, params=params, timeout=30)
                page_end_time = time.time()

                print(f"[DEBUG] Page {page} request took {page_end_time - page_start_time:.2f} seconds")

                if response.status_code == 200:
                    api_data = response.json()
                    # to see raw response
                    articles_data = api_data.get('response', {}).get('results', [])

                    if not articles_data:
                        print(f"[DEBUG] No articles found on page {page}. Stopping pagination.")
                        break
                    print(f"[DEBUG] Found {len(articles_data)} articles on page {page}")

                    if not articles_data:
                        print(f"[DEBUG] All articles on page {page} already exist. Skipping to next page.")
                        continue

                    print(
                        f"[DEBUG] Processing {len(articles_data)} new articles from page {page} (after duplicate check)")

                    # Process each article using the extraction function
                    for i, article_data in enumerate(articles_data):
                        print(f"[DEBUG] Processing article {i + 1}/{len(articles_data)} from page {page}")
                        extracted = extract_article_text(article_data)
                        if extracted:
                            all_extracted_articles.append(extracted)
                        else:
                            print(f"[DEBUG] Failed to extract article {i + 1} from page {page}")

                else:
                    print(f"❌ [DEBUG] Page {page} failed with status {response.status_code}: {response.text}")
                    if page == 1:  # If first page fails, return None
                        return None
                    else:  # If later page fails, continue with what we have
                        break

            total_end_time = time.time()
            total_time = total_end_time - total_start_time

            print(f"\n🎉 [DEBUG] Pagination completed!")
            print(f"📊 [DEBUG] Total pages fetched: {min(page, max_pages)}")
            print(f"📊 [DEBUG] Total articles extracted: {len(all_extracted_articles)}")
            print(f"📊 [DEBUG] Total time: {total_time:.2f} seconds")

            return all_extracted_articles if all_extracted_articles else None

        except requests.exceptions.RequestException as e:
            print(f"🔥 [DEBUG] Request exception occurred: {e}")
            return None