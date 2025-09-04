# Import classes and functions that are actually needed
from .guardian_api import GuardianAPIClient, extract_article_text
from .supabase_db import SupabaseVectorStore

__all__ = [
    'GuardianAPIClient',
    'SupabaseVectorStore', 
    'extract_article_text',
]