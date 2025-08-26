# Import classes
from .guardian_api import GuardianAPIClient
from .supabase_db import SupabaseVectorStore
from .inserting_supabase import ArticleProcessor

# Import key methods directly from classes
from .guardian_api import GuardianAPIClient as _GuardianAPIClient
from .supabase_db import SupabaseVectorStore as _SupabaseVectorStore
from .inserting_supabase import ArticleProcessor as _ArticleProcessor

# Make key methods available at module level
search_articles = _GuardianAPIClient.search_articles
retrieve_relevant_chunks = _SupabaseVectorStore.retrieve_relevant_chunks
database_uploading = _ArticleProcessor.database_uploading
_extract_article_text = _GuardianAPIClient._extract_article_text
_check_article_exists = _SupabaseVectorStore._check_article_exists
_insert_guardian_article_metadata = _SupabaseVectorStore._insert_guardian_article_metadata
_insert_article_chunks = _SupabaseVectorStore._insert_article_chunks
_delete_article = _SupabaseVectorStore._delete_article
_create_embeddings = _ArticleProcessor._create_embeddings
_process_article = _ArticleProcessor._process_article

__all__ = [
    # Classes
    'GuardianAPIClient',
    'SupabaseVectorStore', 
    'ArticleProcessor',
    
    # Main user functions
    'search_articles',        # Users search for articles
    'retrieve_relevant_chunks', # Users search database
    'database_uploading',     # Users run full pipeline
    
    # Internal helper functions
    '_extract_article_text',
    '_check_article_exists',
    '_insert_guardian_article_metadata',
    '_insert_article_chunks',
    '_delete_article',  #actually we are not using this function #######################################################################################
    '_create_embeddings',
    '_process_article',
]