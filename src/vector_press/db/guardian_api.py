import requests
from datetime import datetime
from typing import Dict
import time
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import settings


class GuardianAPIClient:
    def __init__(self):
        self.api_key = settings.GUARDIAN_API_KEY
        self.base_url = "https://content.guardianapis.com"

        print(f"🔧 [DEBUG] Guardian API Client initialized")

    def search_articles(self,
                        query: str = None,
                        section: str = "technology",
                        from_date: str = None,
                        page_size: int = 200,
                        show_fields: str = "all",
                        order_by: str = None) -> Dict | None:
        """
        Search for articles using Guardian API
        """
        print(f"\n📡 [DEBUG] Starting API search...")

        # Build API endpoint
        endpoint = f"{self.base_url}/search"

        # Build required parameters
        params = {
            "api-key": self.api_key,
            "section": section,
            "page-size": page_size,
            "show-fields": show_fields
        }

        # Add optional parameters
        if query:
            params["q"] = query

        if from_date:
            params["from-date"] = from_date

        if order_by:
            params["order-by"] = order_by

        print(f"[DEBUG] Final API URL: {endpoint}")

        try:
            print(f"[DEBUG] Sending HTTP GET request...")
            start_time = time.time()

            response = requests.get(endpoint, params=params, timeout=30)

            end_time = time.time()
            print(f"[DEBUG] Request took {end_time - start_time:.2f} seconds")
            print(f"[DEBUG] Response headers: {dict(response.headers)}")

            if response.status_code == 200:
                print(f"[DEBUG] API request successful!")
                return response.json()
            else:
                print(f"[DEBUG] Error details: {response.text}")
                return None

        except requests.exceptions.RequestException as e:
            print(f"[DEBUG] Request exception occurred: {e}")
            return None

    def _extract_article_text(self, article_data: Dict) -> Dict | None:
        """
        Extract and clean text from Guardian API article response
        """
        print(f"\n🔍 [DEBUG] Extracting text from article...")

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
            print(f"🔍 [DEBUG] Available fields: {list(fields.keys())}")

            # Get different text content
            headline = fields.get("headline", title)
            standfirst = fields.get("standfirst", "")  # Summary/subtitle
            body_text = fields.get("bodyText", "")
            trail_text = fields.get("trailText", "")  # Preview text

            print(f"🔍 [DEBUG] Headline length: {len(headline)} chars")
            print(f"🔍 [DEBUG] Standfirst length: {len(standfirst)} chars")
            print(f"🔍 [DEBUG] Body text length: {len(body_text)} chars")
            print(f"🔍 [DEBUG] Trail text length: {len(trail_text)} chars")

            # Combine all text content
            # ADD HERE PUBLICATION TIME WHICH IT WILL NE NEEDED
            full_text_parts = []
            if headline:
                full_text_parts.append(headline)
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
                "article_id": article_id,         # Guardian API ID as article_id (e.g., "world/2022/oct/21/russia-ukraine-war-latest...")
                "title": title,
                "headline": headline,
                "section": section_name,
                "publication_date": publication_date,
                "url": url,
                "summary": standfirst,
                "body_text": body_text,
                "trail_text": trail_text,
                "word_count": len(full_text.split()),
                "char_count": len(full_text),
                "fetch_time": datetime.now().isoformat()
            }

            print(f"✅ [DEBUG] Article extraction completed!")
            print(f"✅ [DEBUG] Final word count: {meta_data['word_count']} words")

            return {
                'metadata': meta_data,
                'content': full_text
            }

        except Exception as e:
            print(f"🔥 [DEBUG] Error extracting article text: {e}")
            return None