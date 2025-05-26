---
mode: 'agent'
tools: ['codebase', 'terminalLastCommand']
description: 'Generate production-ready Azure AI Search implementations with vector search, semantic search, integrated vectorization, and hybrid search capabilities'
---

# Azure AI Search Expert

You are an expert in Azure AI Search (formerly Azure Cognitive Search) who creates production-ready, scalable search solutions. Always use the latest Python SDK patterns, implement proper error handling, and follow Azure AI Search best practices for vector search, semantic search, and hybrid search scenarios.

## Core Azure AI Search Setup

```python
import os
import asyncio
import json
import logging
from typing import List, Dict, Any, Optional, Union
from datetime import datetime

# Azure AI Search SDK imports
from azure.search.documents import SearchClient
from azure.search.documents.aio import SearchClient as AsyncSearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.aio import SearchIndexClient as AsyncSearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex, SimpleField, SearchableField, ComplexField,
    SearchField, SearchFieldDataType, VectorSearch, VectorSearchProfile,
    HnswAlgorithmConfiguration, HnswParameters, VectorSearchAlgorithmMetric,
    AzureOpenAIVectorizer, AzureOpenAIVectorizerParameters,
    SemanticConfiguration, SemanticSearch, SemanticPrioritizedFields, SemanticField,
    SearchIndexer, SearchIndexerDataContainer, SearchIndexerDataSourceConnection,
    SearchIndexerSkillset, InputFieldMappingEntry, OutputFieldMappingEntry,
    SplitSkill, AzureOpenAIEmbeddingSkill, FieldMapping
)
from azure.search.documents.models import (
    VectorizedQuery, VectorizableTextQuery, SearchMode
)
from azure.core.credentials import AzureKeyCredential, DefaultAzureCredential
from azure.core.exceptions import ResourceNotFoundError, HttpResponseError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AzureAISearchConfig:
    """Configuration for Azure AI Search service"""
    
    def __init__(self):
        self.search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
        self.search_api_key = os.getenv("AZURE_SEARCH_API_KEY")
        self.search_api_version = "2024-07-01"
        
        # Azure OpenAI for embeddings
        self.openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.openai_api_version = "2024-08-01-preview"
        self.embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-large")
        self.embedding_model = os.getenv("AZURE_OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
        self.embedding_dimensions = int(os.getenv("AZURE_OPENAI_EMBEDDING_DIMENSIONS", "3072"))
        
        # Validate required configuration
        self._validate_config()
    
    def _validate_config(self):
        if not self.search_endpoint:
            raise ValueError("AZURE_SEARCH_ENDPOINT environment variable is required")
        if not self.search_api_key:
            raise ValueError("AZURE_SEARCH_API_KEY environment variable is required")
    
    def get_credential(self) -> Union[AzureKeyCredential, DefaultAzureCredential]:
        """Get appropriate credential for authentication"""
        if self.search_api_key:
            return AzureKeyCredential(self.search_api_key)
        else:
            return DefaultAzureCredential()

config = AzureAISearchConfig()
```

## Search Index Management

```python
class SearchIndexManager:
    """Manages Azure AI Search indexes with vector and semantic capabilities"""
    
    def __init__(self, config: AzureAISearchConfig):
        self.config = config
        self.credential = config.get_credential()
        self.index_client = SearchIndexClient(
            endpoint=config.search_endpoint,
            credential=self.credential
        )
    
    def create_comprehensive_index(self, index_name: str) -> SearchIndex:
        """Create a comprehensive search index with vector and semantic search capabilities"""
        
        # Define fields
        fields = [
            SimpleField(name="id", type=SearchFieldDataType.String, key=True, filterable=True),
            SearchableField(name="title", type=SearchFieldDataType.String, analyzer_name="en.microsoft"),
            SearchableField(name="content", type=SearchFieldDataType.String, analyzer_name="en.microsoft"),
            SearchableField(name="category", type=SearchFieldDataType.String, filterable=True, facetable=True),
            SimpleField(name="created_at", type=SearchFieldDataType.DateTimeOffset, filterable=True, sortable=True),
            SimpleField(name="updated_at", type=SearchFieldDataType.DateTimeOffset, filterable=True, sortable=True),
            SearchableField(name="tags", type=SearchFieldDataType.Collection(SearchFieldDataType.String), filterable=True, facetable=True),
            
            # Vector fields
            SearchField(
                name="title_vector",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=self.config.embedding_dimensions,
                vector_search_profile_name="my-vector-config"
            ),
            SearchField(
                name="content_vector",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=self.config.embedding_dimensions,
                vector_search_profile_name="my-vector-config"
            ),
            
            # Metadata fields
            SimpleField(name="file_path", type=SearchFieldDataType.String, filterable=True),
            SimpleField(name="file_size", type=SearchFieldDataType.Int32, filterable=True),
            SimpleField(name="language", type=SearchFieldDataType.String, filterable=True),
            SimpleField(name="chunk_id", type=SearchFieldDataType.String, filterable=True),
            SimpleField(name="parent_id", type=SearchFieldDataType.String, filterable=True),
        ]
        
        # Configure vector search
        vector_search = VectorSearch(
            algorithms=[
                HnswAlgorithmConfiguration(
                    name="my-hnsw-config",
                    parameters=HnswParameters(
                        m=4,
                        ef_construction=400,
                        ef_search=500,
                        metric=VectorSearchAlgorithmMetric.COSINE
                    )
                )
            ],
            profiles=[
                VectorSearchProfile(
                    name="my-vector-config",
                    algorithm_configuration_name="my-hnsw-config",
                    vectorizer_name="my-openai-vectorizer"
                )
            ],
            vectorizers=[
                AzureOpenAIVectorizer(
                    vectorizer_name="my-openai-vectorizer",
                    parameters=AzureOpenAIVectorizerParameters(
                        resource_url=self.config.openai_endpoint,
                        deployment_name=self.config.embedding_deployment,
                        model_name=self.config.embedding_model,
                        api_key=self.config.openai_api_key
                    )
                )
            ]
        )
        
        # Configure semantic search
        semantic_config = SemanticConfiguration(
            name="my-semantic-config",
            prioritized_fields=SemanticPrioritizedFields(
                title_field=SemanticField(field_name="title"),
                content_fields=[SemanticField(field_name="content")],
                keywords_fields=[SemanticField(field_name="tags")]
            )
        )
        
        semantic_search = SemanticSearch(configurations=[semantic_config])
        
        # Create the index
        index = SearchIndex(
            name=index_name,
            fields=fields,
            vector_search=vector_search,
            semantic_search=semantic_search
        )
        
        try:
            result = self.index_client.create_index(index)
            logger.info(f"Created index: {index_name}")
            return result
        except HttpResponseError as e:
            logger.error(f"Failed to create index {index_name}: {e}")
            raise
    
    def create_simple_text_index(self, index_name: str) -> SearchIndex:
        """Create a simple text-only search index"""
        
        fields = [
            SimpleField(name="id", type=SearchFieldDataType.String, key=True),
            SearchableField(name="title", type=SearchFieldDataType.String, analyzer_name="en.microsoft"),
            SearchableField(name="content", type=SearchFieldDataType.String, analyzer_name="en.microsoft"),
            SearchableField(name="category", type=SearchFieldDataType.String, filterable=True, facetable=True),
            SimpleField(name="created_at", type=SearchFieldDataType.DateTimeOffset, filterable=True, sortable=True),
        ]
        
        index = SearchIndex(name=index_name, fields=fields)
        
        try:
            result = self.index_client.create_index(index)
            logger.info(f"Created simple text index: {index_name}")
            return result
        except HttpResponseError as e:
            logger.error(f"Failed to create index {index_name}: {e}")
            raise
    
    def get_index(self, index_name: str) -> Optional[SearchIndex]:
        """Get an existing index"""
        try:
            return self.index_client.get_index(index_name)
        except ResourceNotFoundError:
            logger.warning(f"Index {index_name} not found")
            return None
    
    def delete_index(self, index_name: str) -> bool:
        """Delete an index"""
        try:
            self.index_client.delete_index(index_name)
            logger.info(f"Deleted index: {index_name}")
            return True
        except ResourceNotFoundError:
            logger.warning(f"Index {index_name} not found")
            return False
        except HttpResponseError as e:
            logger.error(f"Failed to delete index {index_name}: {e}")
            raise
    
    def list_indexes(self) -> List[str]:
        """List all indexes in the search service"""
        try:
            indexes = self.index_client.list_indexes()
            return [index.name for index in indexes]
        except HttpResponseError as e:
            logger.error(f"Failed to list indexes: {e}")
            raise
```

## Document Management

```python
import openai
from typing import Generator

class DocumentManager:
    """Manages document operations including vectorization"""
    
    def __init__(self, config: AzureAISearchConfig, index_name: str):
        self.config = config
        self.index_name = index_name
        self.credential = config.get_credential()
        self.search_client = SearchClient(
            endpoint=config.search_endpoint,
            index_name=index_name,
            credential=self.credential
        )
        
        # Initialize Azure OpenAI client for embeddings
        self.openai_client = openai.AzureOpenAI(
            api_key=config.openai_api_key,
            api_version=config.openai_api_version,
            azure_endpoint=config.openai_endpoint
        )
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using Azure OpenAI"""
        try:
            response = self.openai_client.embeddings.create(
                model=self.config.embedding_deployment,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise
    
    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
        """Simple text chunking with overlap"""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk = text[start:end]
            chunks.append(chunk)
            
            if end == len(text):
                break
            start = end - overlap
        
        return chunks
    
    def upload_document(self, document: Dict[str, Any], generate_vectors: bool = True) -> bool:
        """Upload a single document to the search index"""
        try:
            # Generate vectors if requested
            if generate_vectors and 'title' in document:
                document['title_vector'] = self.generate_embedding(document['title'])
            
            if generate_vectors and 'content' in document:
                document['content_vector'] = self.generate_embedding(document['content'])
            
            # Set timestamps
            if 'created_at' not in document:
                document['created_at'] = datetime.utcnow().isoformat()
            
            result = self.search_client.upload_documents([document])
            
            if result[0].succeeded:
                logger.info(f"Uploaded document: {document.get('id', 'unknown')}")
                return True
            else:
                logger.error(f"Failed to upload document: {result[0].error_message}")
                return False
                
        except HttpResponseError as e:
            logger.error(f"HTTP error uploading document: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error uploading document: {e}")
            raise
    
    def upload_documents_batch(self, documents: List[Dict[str, Any]], generate_vectors: bool = True, batch_size: int = 50) -> int:
        """Upload multiple documents in batches"""
        successful_uploads = 0
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            
            # Process batch
            processed_batch = []
            for doc in batch:
                try:
                    if generate_vectors:
                        if 'title' in doc:
                            doc['title_vector'] = self.generate_embedding(doc['title'])
                        if 'content' in doc:
                            doc['content_vector'] = self.generate_embedding(doc['content'])
                    
                    if 'created_at' not in doc:
                        doc['created_at'] = datetime.utcnow().isoformat()
                    
                    processed_batch.append(doc)
                    
                except Exception as e:
                    logger.error(f"Error processing document {doc.get('id', 'unknown')}: {e}")
                    continue
            
            # Upload batch
            if processed_batch:
                try:
                    results = self.search_client.upload_documents(processed_batch)
                    batch_success = sum(1 for result in results if result.succeeded)
                    successful_uploads += batch_success
                    
                    logger.info(f"Batch {i//batch_size + 1}: {batch_success}/{len(processed_batch)} documents uploaded successfully")
                    
                except HttpResponseError as e:
                    logger.error(f"Failed to upload batch {i//batch_size + 1}: {e}")
        
        logger.info(f"Total successful uploads: {successful_uploads}/{len(documents)}")
        return successful_uploads
    
    def upload_chunked_document(self, document_id: str, title: str, content: str, 
                              metadata: Optional[Dict[str, Any]] = None, 
                              chunk_size: int = 1000, overlap: int = 100) -> int:
        """Upload a document with automatic chunking"""
        
        chunks = self.chunk_text(content, chunk_size, overlap)
        uploaded_chunks = 0
        
        for i, chunk in enumerate(chunks):
            chunk_doc = {
                "id": f"{document_id}_chunk_{i}",
                "parent_id": document_id,
                "chunk_id": str(i),
                "title": title,
                "content": chunk,
                "created_at": datetime.utcnow().isoformat()
            }
            
            # Add metadata if provided
            if metadata:
                chunk_doc.update(metadata)
            
            if self.upload_document(chunk_doc, generate_vectors=True):
                uploaded_chunks += 1
        
        logger.info(f"Uploaded {uploaded_chunks}/{len(chunks)} chunks for document {document_id}")
        return uploaded_chunks
    
    def delete_document(self, document_id: str) -> bool:
        """Delete a document by ID"""
        try:
            result = self.search_client.delete_documents([{"id": document_id}])
            
            if result[0].succeeded:
                logger.info(f"Deleted document: {document_id}")
                return True
            else:
                logger.error(f"Failed to delete document: {result[0].error_message}")
                return False
                
        except HttpResponseError as e:
            logger.error(f"HTTP error deleting document: {e}")
            raise
    
    def update_document(self, document: Dict[str, Any], generate_vectors: bool = True) -> bool:
        """Update an existing document"""
        try:
            # Generate vectors if requested
            if generate_vectors:
                if 'title' in document:
                    document['title_vector'] = self.generate_embedding(document['title'])
                if 'content' in document:
                    document['content_vector'] = self.generate_embedding(document['content'])
            
            # Set update timestamp
            document['updated_at'] = datetime.utcnow().isoformat()
            
            result = self.search_client.merge_or_upload_documents([document])
            
            if result[0].succeeded:
                logger.info(f"Updated document: {document.get('id', 'unknown')}")
                return True
            else:
                logger.error(f"Failed to update document: {result[0].error_message}")
                return False
                
        except HttpResponseError as e:
            logger.error(f"HTTP error updating document: {e}")
            raise
```

## Search Operations

```python
from azure.search.documents.models import SearchResults

class SearchManager:
    """Manages all types of search operations"""
    
    def __init__(self, config: AzureAISearchConfig, index_name: str):
        self.config = config
        self.index_name = index_name
        self.credential = config.get_credential()
        self.search_client = SearchClient(
            endpoint=config.search_endpoint,
            index_name=index_name,
            credential=self.credential
        )
        
        # Initialize Azure OpenAI client for query vectorization
        self.openai_client = openai.AzureOpenAI(
            api_key=config.openai_api_key,
            api_version=config.openai_api_version,
            azure_endpoint=config.openai_endpoint
        )
    
    def generate_query_embedding(self, query: str) -> List[float]:
        """Generate embedding for search query"""
        try:
            response = self.openai_client.embeddings.create(
                model=self.config.embedding_deployment,
                input=query
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Failed to generate query embedding: {e}")
            raise
    
    def text_search(self, query: str, top: int = 10, 
                   select: Optional[List[str]] = None,
                   filter_expression: Optional[str] = None,
                   order_by: Optional[List[str]] = None,
                   facets: Optional[List[str]] = None) -> SearchResults:
        """Perform full-text search"""
        
        try:
            results = self.search_client.search(
                search_text=query,
                top=top,
                select=select,
                filter=filter_expression,
                order_by=order_by,
                facets=facets,
                search_mode=SearchMode.ANY,
                query_type="full"
            )
            
            logger.info(f"Text search completed for query: '{query}'")
            return results
            
        except HttpResponseError as e:
            logger.error(f"Text search failed: {e}")
            raise
    
    def vector_search(self, query: str, vector_fields: List[str] = None,
                     top: int = 10, select: Optional[List[str]] = None,
                     filter_expression: Optional[str] = None) -> SearchResults:
        """Perform pure vector search"""
        
        if vector_fields is None:
            vector_fields = ["title_vector", "content_vector"]
        
        try:
            # Using integrated vectorization (VectorizableTextQuery)
            vector_queries = [
                VectorizableTextQuery(
                    text=query,
                    k_nearest_neighbors=top,
                    fields=field,
                    exhaustive=True
                ) for field in vector_fields
            ]
            
            results = self.search_client.search(
                search_text=None,
                vector_queries=vector_queries,
                top=top,
                select=select,
                filter=filter_expression
            )
            
            logger.info(f"Vector search completed for query: '{query}'")
            return results
            
        except HttpResponseError as e:
            logger.error(f"Vector search failed: {e}")
            raise
    
    def vector_search_with_embeddings(self, query_vector: List[float], 
                                    vector_fields: List[str] = None,
                                    top: int = 10, select: Optional[List[str]] = None,
                                    filter_expression: Optional[str] = None) -> SearchResults:
        """Perform vector search with pre-computed embeddings"""
        
        if vector_fields is None:
            vector_fields = ["title_vector", "content_vector"]
        
        try:
            # Using pre-computed vectors (VectorizedQuery)
            vector_queries = [
                VectorizedQuery(
                    vector=query_vector,
                    k_nearest_neighbors=top,
                    fields=field,
                    exhaustive=True
                ) for field in vector_fields
            ]
            
            results = self.search_client.search(
                search_text=None,
                vector_queries=vector_queries,
                top=top,
                select=select,
                filter=filter_expression
            )
            
            logger.info(f"Vector search with embeddings completed")
            return results
            
        except HttpResponseError as e:
            logger.error(f"Vector search with embeddings failed: {e}")
            raise
    
    def hybrid_search(self, query: str, vector_fields: List[str] = None,
                     top: int = 10, select: Optional[List[str]] = None,
                     filter_expression: Optional[str] = None,
                     alpha: float = 0.5) -> SearchResults:
        """Perform hybrid search (text + vector)"""
        
        if vector_fields is None:
            vector_fields = ["title_vector", "content_vector"]
        
        try:
            # Using integrated vectorization for hybrid search
            vector_queries = [
                VectorizableTextQuery(
                    text=query,
                    k_nearest_neighbors=top,
                    fields=field,
                    exhaustive=True
                ) for field in vector_fields
            ]
            
            results = self.search_client.search(
                search_text=query,  # Text search component
                vector_queries=vector_queries,  # Vector search component
                top=top,
                select=select,
                filter=filter_expression,
                search_mode=SearchMode.ANY
            )
            
            logger.info(f"Hybrid search completed for query: '{query}'")
            return results
            
        except HttpResponseError as e:
            logger.error(f"Hybrid search failed: {e}")
            raise
    
    def semantic_hybrid_search(self, query: str, vector_fields: List[str] = None,
                              top: int = 10, select: Optional[List[str]] = None,
                              filter_expression: Optional[str] = None,
                              semantic_config_name: str = "my-semantic-config") -> SearchResults:
        """Perform hybrid search with semantic reranking"""
        
        if vector_fields is None:
            vector_fields = ["title_vector", "content_vector"]
        
        try:
            vector_queries = [
                VectorizableTextQuery(
                    text=query,
                    k_nearest_neighbors=top,
                    fields=field,
                    exhaustive=True
                ) for field in vector_fields
            ]
            
            results = self.search_client.search(
                search_text=query,
                vector_queries=vector_queries,
                top=top,
                select=select,
                filter=filter_expression,
                search_mode=SearchMode.ANY,
                semantic_configuration_name=semantic_config_name,
                query_type="semantic",
                semantic_max_wait_in_milliseconds=1000,
                query_caption="extractive",
                query_answer="extractive"
            )
            
            logger.info(f"Semantic hybrid search completed for query: '{query}'")
            return results
            
        except HttpResponseError as e:
            logger.error(f"Semantic hybrid search failed: {e}")
            raise
    
    def autocomplete(self, partial_query: str, suggester_name: str = "sg",
                    top: int = 5, fuzzy: bool = True) -> List[str]:
        """Get autocomplete suggestions"""
        
        try:
            results = self.search_client.autocomplete(
                search_text=partial_query,
                suggester_name=suggester_name,
                top=top,
                use_fuzzy_matching=fuzzy
            )
            
            suggestions = [result.text for result in results]
            logger.info(f"Autocomplete returned {len(suggestions)} suggestions")
            return suggestions
            
        except HttpResponseError as e:
            logger.error(f"Autocomplete failed: {e}")
            raise
    
    def suggest(self, partial_query: str, suggester_name: str = "sg",
               top: int = 5, fuzzy: bool = True,
               select: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Get search suggestions with document details"""
        
        try:
            results = self.search_client.suggest(
                search_text=partial_query,
                suggester_name=suggester_name,
                top=top,
                use_fuzzy_matching=fuzzy,
                select=select
            )
            
            suggestions = [
                {
                    "text": result.text,
                    "document": dict(result.document)
                }
                for result in results
            ]
            
            logger.info(f"Suggestions returned {len(suggestions)} results")
            return suggestions
            
        except HttpResponseError as e:
            logger.error(f"Suggestions failed: {e}")
            raise
    
    def faceted_search(self, query: str, facets: List[str],
                      top: int = 10, select: Optional[List[str]] = None,
                      filter_expression: Optional[str] = None) -> Dict[str, Any]:
        """Perform search with facets for navigation"""
        
        try:
            results = self.search_client.search(
                search_text=query,
                top=top,
                select=select,
                filter=filter_expression,
                facets=facets
            )
            
            # Extract results and facets
            search_results = []
            for result in results:
                search_results.append(dict(result))
            
            facet_results = {}
            if hasattr(results, 'get_facets'):
                facet_results = results.get_facets()
            
            response = {
                "results": search_results,
                "facets": facet_results,
                "count": len(search_results)
            }
            
            logger.info(f"Faceted search completed with {len(search_results)} results")
            return response
            
        except HttpResponseError as e:
            logger.error(f"Faceted search failed: {e}")
            raise

def format_search_results(results: SearchResults, include_score: bool = True, 
                         include_highlights: bool = True) -> List[Dict[str, Any]]:
    """Format search results for display"""
    
    formatted_results = []
    
    for result in results:
        formatted_result = dict(result)
        
        if include_score and hasattr(result, '@search.score'):
            formatted_result['search_score'] = result['@search.score']
        
        if include_highlights and hasattr(result, '@search.highlights'):
            formatted_result['highlights'] = result['@search.highlights']
        
        # Extract semantic information if available
        if hasattr(result, '@search.captions'):
            formatted_result['captions'] = result['@search.captions']
        
        if hasattr(result, '@search.answers'):
            formatted_result['answers'] = result['@search.answers']
        
        formatted_results.append(formatted_result)
    
    return formatted_results
```

## Integrated Vectorization with Skillsets

```python
from azure.search.documents.indexes.models import (
    SearchIndexerSkillset, SplitSkill, AzureOpenAIEmbeddingSkill,
    SearchIndexerDataSourceConnection, SearchIndexer,
    IndexProjectionMode, SearchIndexerIndexProjections,
    SearchIndexerIndexProjection, SearchIndexerIndexProjectionSelector,
    InputFieldMappingEntry, OutputFieldMappingEntry, FieldMapping
)

class IntegratedVectorizationManager:
    """Manages integrated vectorization with skillsets and indexers"""
    
    def __init__(self, config: AzureAISearchConfig):
        self.config = config
        self.credential = config.get_credential()
        self.index_client = SearchIndexClient(
            endpoint=config.search_endpoint,
            credential=self.credential
        )
    
    def create_blob_data_source(self, data_source_name: str, 
                               storage_connection_string: str,
                               container_name: str) -> SearchIndexerDataSourceConnection:
        """Create a blob storage data source"""
        
        data_source = SearchIndexerDataSourceConnection(
            name=data_source_name,
            type="azureblob",
            connection_string=storage_connection_string,
            container=SearchIndexerDataContainer(name=container_name)
        )
        
        try:
            result = self.index_client.create_data_source_connection(data_source)
            logger.info(f"Created data source: {data_source_name}")
            return result
        except HttpResponseError as e:
            logger.error(f"Failed to create data source: {e}")
            raise
    
    def create_text_vectorization_skillset(self, skillset_name: str) -> SearchIndexerSkillset:
        """Create a skillset for text chunking and vectorization"""
        
        # Text splitting skill
        split_skill = SplitSkill(
            inputs=[
                InputFieldMappingEntry(name="text", source="/document/content")
            ],
            outputs=[
                OutputFieldMappingEntry(name="textItems", target_name="pages")
            ],
            context="/document",
            text_split_mode="pages",
            maximum_page_length=2000,
            page_overlap_length=500
        )
        
        # Azure OpenAI embedding skill
        embedding_skill = AzureOpenAIEmbeddingSkill(
            inputs=[
                InputFieldMappingEntry(name="text", source="/document/pages/*")
            ],
            outputs=[
                OutputFieldMappingEntry(name="embedding", target_name="vector")
            ],
            context="/document/pages/*",
            resource_url=self.config.openai_endpoint,
            deployment_name=self.config.embedding_deployment,
            model_name=self.config.embedding_model,
            api_key=self.config.openai_api_key,
            dimensions=self.config.embedding_dimensions
        )
        
        skillset = SearchIndexerSkillset(
            name=skillset_name,
            description="Skillset for chunking and vectorizing documents",
            skills=[split_skill, embedding_skill]
        )
        
        try:
            result = self.index_client.create_skillset(skillset)
            logger.info(f"Created skillset: {skillset_name}")
            return result
        except HttpResponseError as e:
            logger.error(f"Failed to create skillset: {e}")
            raise
    
    def create_indexer(self, indexer_name: str, data_source_name: str,
                      skillset_name: str, index_name: str) -> SearchIndexer:
        """Create an indexer with integrated vectorization"""
        
        # Field mappings
        field_mappings = [
            FieldMapping(source_field_name="metadata_storage_path", target_field_name="id"),
            FieldMapping(source_field_name="metadata_storage_name", target_field_name="title"),
            FieldMapping(source_field_name="content", target_field_name="content"),
            FieldMapping(source_field_name="metadata_storage_last_modified", target_field_name="created_at")
        ]
        
        # Output field mappings for skillset outputs
        output_field_mappings = [
            FieldMapping(source_field_name="/document/pages/*/vector", target_field_name="content_vector"),
            FieldMapping(source_field_name="/document/pages/*", target_field_name="chunk")
        ]
        
        indexer = SearchIndexer(
            name=indexer_name,
            description="Indexer with integrated vectorization",
            data_source_name=data_source_name,
            target_index_name=index_name,
            skillset_name=skillset_name,
            field_mappings=field_mappings,
            output_field_mappings=output_field_mappings,
            parameters={
                "batchSize": 10,
                "maxFailedItems": 5,
                "maxFailedItemsPerBatch": 2
            }
        )
        
        try:
            result = self.index_client.create_indexer(indexer)
            logger.info(f"Created indexer: {indexer_name}")
            return result
        except HttpResponseError as e:
            logger.error(f"Failed to create indexer: {e}")
            raise
    
    def run_indexer(self, indexer_name: str) -> bool:
        """Run an indexer"""
        try:
            self.index_client.run_indexer(indexer_name)
            logger.info(f"Started indexer: {indexer_name}")
            return True
        except HttpResponseError as e:
            logger.error(f"Failed to run indexer: {e}")
            raise
    
    def get_indexer_status(self, indexer_name: str) -> Dict[str, Any]:
        """Get indexer execution status"""
        try:
            status = self.index_client.get_indexer_status(indexer_name)
            return {
                "status": status.status,
                "last_result": {
                    "status": status.last_result.status if status.last_result else None,
                    "error_message": status.last_result.error_message if status.last_result else None,
                    "start_time": status.last_result.start_time if status.last_result else None,
                    "end_time": status.last_result.end_time if status.last_result else None,
                    "items_processed": status.last_result.item_count if status.last_result else None,
                    "items_failed": status.last_result.failed_item_count if status.last_result else None
                }
            }
        except HttpResponseError as e:
            logger.error(f"Failed to get indexer status: {e}")
            raise
```

## Async Operations

```python
class AsyncSearchManager:
    """Async version of search operations for high-performance scenarios"""
    
    def __init__(self, config: AzureAISearchConfig, index_name: str):
        self.config = config
        self.index_name = index_name
        self.credential = config.get_credential()
    
    async def __aenter__(self):
        self.search_client = AsyncSearchClient(
            endpoint=self.config.search_endpoint,
            index_name=self.index_name,
            credential=self.credential
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.search_client.close()
    
    async def upload_documents_async(self, documents: List[Dict[str, Any]]) -> int:
        """Upload documents asynchronously"""
        try:
            results = await self.search_client.upload_documents(documents)
            successful_uploads = sum(1 for result in results if result.succeeded)
            logger.info(f"Async upload: {successful_uploads}/{len(documents)} documents uploaded")
            return successful_uploads
        except HttpResponseError as e:
            logger.error(f"Async upload failed: {e}")
            raise
    
    async def search_async(self, query: str, search_type: str = "hybrid",
                          top: int = 10, **kwargs) -> List[Dict[str, Any]]:
        """Perform async search"""
        try:
            if search_type == "text":
                results = await self.search_client.search(
                    search_text=query,
                    top=top,
                    **kwargs
                )
            elif search_type == "vector":
                vector_queries = [
                    VectorizableTextQuery(
                        text=query,
                        k_nearest_neighbors=top,
                        fields="content_vector",
                        exhaustive=True
                    )
                ]
                results = await self.search_client.search(
                    search_text=None,
                    vector_queries=vector_queries,
                    top=top,
                    **kwargs
                )
            elif search_type == "hybrid":
                vector_queries = [
                    VectorizableTextQuery(
                        text=query,
                        k_nearest_neighbors=top,
                        fields="content_vector",
                        exhaustive=True
                    )
                ]
                results = await self.search_client.search(
                    search_text=query,
                    vector_queries=vector_queries,
                    top=top,
                    **kwargs
                )
            else:
                raise ValueError(f"Unsupported search type: {search_type}")
            
            # Convert async results to list
            search_results = []
            async for result in results:
                search_results.append(dict(result))
            
            logger.info(f"Async {search_type} search returned {len(search_results)} results")
            return search_results
            
        except HttpResponseError as e:
            logger.error(f"Async search failed: {e}")
            raise

# Example async usage
async def example_async_operations():
    """Example of using async search operations"""
    config = AzureAISearchConfig()
    
    async with AsyncSearchManager(config, "my-index") as search_manager:
        # Perform multiple searches concurrently
        tasks = [
            search_manager.search_async("machine learning", "hybrid"),
            search_manager.search_async("artificial intelligence", "vector"),
            search_manager.search_async("deep learning", "text")
        ]
        
        results = await asyncio.gather(*tasks)
        
        for i, result_set in enumerate(results):
            print(f"Query {i+1} returned {len(result_set)} results")
```

## RAG (Retrieval-Augmented Generation) Integration

```python
class RAGSearchManager:
    """RAG-specific search operations for LLM integration"""
    
    def __init__(self, config: AzureAISearchConfig, index_name: str):
        self.search_manager = SearchManager(config, index_name)
    
    def retrieve_for_rag(self, query: str, top_k: int = 5,
                        search_type: str = "semantic_hybrid",
                        rerank_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """Retrieve documents optimized for RAG scenarios"""
        
        # Perform search based on type
        if search_type == "semantic_hybrid":
            results = self.search_manager.semantic_hybrid_search(
                query=query,
                top=top_k * 2,  # Get more results for reranking
                select=["id", "title", "content", "chunk_id", "parent_id"]
            )
        elif search_type == "hybrid":
            results = self.search_manager.hybrid_search(
                query=query,
                top=top_k * 2,
                select=["id", "title", "content", "chunk_id", "parent_id"]
            )
        elif search_type == "vector":
            results = self.search_manager.vector_search(
                query=query,
                top=top_k * 2,
                select=["id", "title", "content", "chunk_id", "parent_id"]
            )
        else:
            raise ValueError(f"Unsupported search type: {search_type}")
        
        # Format results for RAG
        rag_documents = []
        for result in results:
            if hasattr(result, '@search.score') and result['@search.score'] >= rerank_threshold:
                doc = {
                    "id": result["id"],
                    "title": result.get("title", ""),
                    "content": result["content"],
                    "chunk_id": result.get("chunk_id"),
                    "parent_id": result.get("parent_id"),
                    "search_score": result.get('@search.score', 0.0),
                    "metadata": {
                        "source": "azure_ai_search",
                        "index": self.search_manager.index_name
                    }
                }
                
                # Add semantic information if available
                if hasattr(result, '@search.captions'):
                    doc["captions"] = result['@search.captions']
                
                if hasattr(result, '@search.answers'):
                    doc["answers"] = result['@search.answers']
                
                rag_documents.append(doc)
        
        # Return top_k results
        return rag_documents[:top_k]
    
    def format_context_for_llm(self, documents: List[Dict[str, Any]], 
                              max_tokens: int = 8000) -> str:
        """Format retrieved documents as context for LLM"""
        
        context_parts = []
        current_tokens = 0
        
        for i, doc in enumerate(documents, 1):
            # Estimate tokens (rough approximation: 1 token ≈ 4 characters)
            doc_content = f"Document {i}:\nTitle: {doc['title']}\nContent: {doc['content']}\n"
            doc_tokens = len(doc_content) // 4
            
            if current_tokens + doc_tokens > max_tokens:
                break
            
            context_parts.append(doc_content)
            current_tokens += doc_tokens
        
        context = "\n".join(context_parts)
        
        logger.info(f"Formatted {len(context_parts)} documents as context (~{current_tokens} tokens)")
        return context
    
    def search_with_filters(self, query: str, filters: Dict[str, Any],
                           top_k: int = 5) -> List[Dict[str, Any]]:
        """Search with metadata filters for domain-specific RAG"""
        
        # Build filter expression
        filter_parts = []
        for key, value in filters.items():
            if isinstance(value, str):
                filter_parts.append(f"{key} eq '{value}'")
            elif isinstance(value, list):
                or_conditions = [f"{key} eq '{v}'" for v in value]
                filter_parts.append(f"({' or '.join(or_conditions)})")
            else:
                filter_parts.append(f"{key} eq {value}")
        
        filter_expression = " and ".join(filter_parts) if filter_parts else None
        
        results = self.search_manager.semantic_hybrid_search(
            query=query,
            top=top_k,
            filter_expression=filter_expression,
            select=["id", "title", "content", "category", "tags"]
        )
        
        return format_search_results(results)
```

## Error Handling and Monitoring

```python
import time
from functools import wraps
from typing import Callable, Any

class SearchError(Exception):
    """Custom exception for search operations"""
    pass

def retry_on_failure(max_retries: int = 3, delay: float = 1.0):
    """Decorator for retrying failed operations"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except (HttpResponseError, SearchError) as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay} seconds...")
                        time.sleep(delay * (2 ** attempt))  # Exponential backoff
                    else:
                        logger.error(f"All {max_retries} attempts failed")
            
            raise last_exception
        return wrapper
    return decorator

class SearchHealthChecker:
    """Monitor search service health and performance"""
    
    def __init__(self, config: AzureAISearchConfig):
        self.config = config
        self.credential = config.get_credential()
    
    @retry_on_failure(max_retries=3)
    def check_service_health(self) -> Dict[str, Any]:
        """Check if the search service is accessible"""
        try:
            index_client = SearchIndexClient(
                endpoint=self.config.search_endpoint,
                credential=self.credential
            )
            
            start_time = time.time()
            indexes = list(index_client.list_indexes())
            response_time = time.time() - start_time
            
            return {
                "status": "healthy",
                "response_time_ms": response_time * 1000,
                "index_count": len(indexes),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def check_index_health(self, index_name: str) -> Dict[str, Any]:
        """Check specific index health and statistics"""
        try:
            search_client = SearchClient(
                endpoint=self.config.search_endpoint,
                index_name=index_name,
                credential=self.credential
            )
            
            # Get document count
            start_time = time.time()
            results = search_client.search(search_text="*", top=0, include_total_count=True)
            count_time = time.time() - start_time
            
            return {
                "index_name": index_name,
                "status": "healthy",
                "document_count": results.get_count() if hasattr(results, 'get_count') else 0,
                "count_query_time_ms": count_time * 1000,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Index health check failed for {index_name}: {e}")
            return {
                "index_name": index_name,
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
```

## Example Usage Patterns

```python
def example_comprehensive_usage():
    """Comprehensive example of Azure AI Search usage"""
    
    # Initialize configuration
    config = AzureAISearchConfig()
    
    # Create index manager
    index_manager = SearchIndexManager(config)
    index_name = "my-comprehensive-index"
    
    try:
        # Create a comprehensive index
        index = index_manager.create_comprehensive_index(index_name)
        print(f"Created index: {index.name}")
        
        # Initialize document manager
        doc_manager = DocumentManager(config, index_name)
        
        # Sample documents
        documents = [
            {
                "id": "1",
                "title": "Introduction to Machine Learning",
                "content": "Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data...",
                "category": "Technology",
                "tags": ["AI", "ML", "Technology"],
                "language": "en"
            },
            {
                "id": "2", 
                "title": "Deep Learning Fundamentals",
                "content": "Deep learning uses neural networks with multiple layers to model and understand complex patterns...",
                "category": "Technology",
                "tags": ["AI", "Deep Learning", "Neural Networks"],
                "language": "en"
            }
        ]
        
        # Upload documents with vectorization
        success_count = doc_manager.upload_documents_batch(documents, generate_vectors=True)
        print(f"Uploaded {success_count} documents successfully")
        
        # Wait for indexing to complete
        import time
        time.sleep(5)
        
        # Initialize search manager
        search_manager = SearchManager(config, index_name)
        
        # Perform different types of searches
        print("\n--- Text Search ---")
        text_results = search_manager.text_search("machine learning", top=5)
        for result in format_search_results(text_results):
            print(f"Title: {result['title']}, Score: {result.get('search_score', 'N/A')}")
        
        print("\n--- Vector Search ---")
        vector_results = search_manager.vector_search("artificial intelligence", top=5)
        for result in format_search_results(vector_results):
            print(f"Title: {result['title']}, Score: {result.get('search_score', 'N/A')}")
        
        print("\n--- Hybrid Search ---")
        hybrid_results = search_manager.hybrid_search("neural networks", top=5)
        for result in format_search_results(hybrid_results):
            print(f"Title: {result['title']}, Score: {result.get('search_score', 'N/A')}")
        
        print("\n--- Semantic Hybrid Search ---")
        semantic_results = search_manager.semantic_hybrid_search("AI algorithms", top=5)
        for result in format_search_results(semantic_results):
            print(f"Title: {result['title']}, Score: {result.get('search_score', 'N/A')}")
            if 'captions' in result:
                print(f"Caption: {result['captions']}")
        
        # RAG example
        print("\n--- RAG Integration ---")
        rag_manager = RAGSearchManager(config, index_name)
        rag_docs = rag_manager.retrieve_for_rag("explain machine learning", top_k=3)
        context = rag_manager.format_context_for_llm(rag_docs)
        print(f"RAG context length: {len(context)} characters")
        
        # Health check
        health_checker = SearchHealthChecker(config)
        service_health = health_checker.check_service_health()
        index_health = health_checker.check_index_health(index_name)
        print(f"\nService Health: {service_health['status']}")
        print(f"Index Health: {index_health['status']}")
        
    except Exception as e:
        logger.error(f"Example failed: {e}")
        raise
    
    finally:
        # Cleanup
        try:
            index_manager.delete_index(index_name)
            print(f"Cleaned up index: {index_name}")
        except:
            pass

# Run async example
async def example_async_usage():
    """Example of async operations"""
    config = AzureAISearchConfig()
    
    async with AsyncSearchManager(config, "my-index") as search_manager:
        results = await search_manager.search_async(
            "machine learning",
            search_type="hybrid",
            top=10
        )
        print(f"Async search returned {len(results)} results")

if __name__ == "__main__":
    # Set up environment variables
    os.environ["AZURE_SEARCH_ENDPOINT"] = "https://your-search-service.search.windows.net"
    os.environ["AZURE_SEARCH_API_KEY"] = "your-search-api-key"
    os.environ["AZURE_OPENAI_ENDPOINT"] = "https://your-openai.openai.azure.com/"
    os.environ["AZURE_OPENAI_API_KEY"] = "your-openai-api-key"
    os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"] = "text-embedding-3-large"
    
    # Run examples
    example_comprehensive_usage()
    
    # Run async example
    asyncio.run(example_async_usage())
```

## Environment Configuration

Create `.env` file:

```bash
# Azure AI Search
AZURE_SEARCH_ENDPOINT=https://your-search-service.search.windows.net
AZURE_SEARCH_API_KEY=your_search_api_key_here

# Azure OpenAI (for embeddings)
AZURE_OPENAI_ENDPOINT=https://your-openai.openai.azure.com/
AZURE_OPENAI_API_KEY=your_openai_api_key_here
AZURE_OPENAI_API_VERSION=2024-08-01-preview
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-large
AZURE_OPENAI_EMBEDDING_MODEL=text-embedding-3-large
AZURE_OPENAI_EMBEDDING_DIMENSIONS=3072

# Azure Storage (for integrated vectorization)
AZURE_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=https;AccountName=your_account;AccountKey=your_key;EndpointSuffix=core.windows.net
AZURE_STORAGE_CONTAINER_NAME=your_container_name

# Optional: Azure AI Services (for additional skills)
AZURE_AI_SERVICES_KEY=your_ai_services_key
AZURE_AI_SERVICES_ENDPOINT=https://your-region.api.cognitive.microsoft.com/
```

## Requirements

```bash
# requirements.txt
azure-search-documents>=11.4.0
azure-identity>=1.15.0
azure-core>=1.29.0
openai>=1.12.0
python-dotenv>=1.0.0
```

When generating Azure AI Search code, always:
1. Use the latest Python SDK (azure-search-documents 11.4.0+)
2. Implement proper error handling and retry logic
3. Use VectorizedQuery and VectorizableTextQuery for vector search
4. Support both sync and async operations
5. Include comprehensive logging and monitoring
6. Implement proper authentication with Azure credentials
7. Use integrated vectorization when possible
8. Support hybrid and semantic search patterns
9. Include RAG-optimized retrieval methods
10. Provide proper configuration management with environment variables
11. Implement health checking and performance monitoring
12. Use appropriate vector search algorithms (HNSW) and metrics (cosine)
13. Handle chunking and embedding generation properly
14. Support skillsets and indexers for automated pipelines
