import os
import pickle
import logging
from typing import List, Dict, Tuple, Optional
import numpy as np
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer
import torch
from dataclasses import dataclass

from .document_processor import DocumentChunk, DocumentMetadata

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Represents a search result with similarity score"""
    chunk: DocumentChunk
    similarity_score: float
    rank: int

class EmbeddingEngine:
    """Handles document embeddings and semantic search using FAISS"""
    
    def __init__(self, 
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 index_type: str = "flat",
                 dimension: int = 384):
        """
        Initialize the embedding engine
        
        Args:
            model_name: Name of the sentence transformer model
            index_type: Type of FAISS index ('flat', 'ivf', 'hnsw')
            dimension: Dimension of embeddings (384 for MiniLM-L6-v2)
        """
        self.model_name = model_name
        self.index_type = index_type
        self.dimension = dimension
        
        # Initialize the sentence transformer model
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        
        # Initialize FAISS index
        self.index = self._create_faiss_index()
        self.chunks_metadata: List[DocumentChunk] = []
        self.document_metadata: Dict[str, DocumentMetadata] = {}
        
        # Cache for embeddings
        self.embedding_cache = {}
        
    def _create_faiss_index(self) -> faiss.Index:
        """Create FAISS index based on the specified type"""
        if self.index_type.lower() == "flat":
            # L2 distance index (exact search)
            return faiss.IndexFlatL2(self.dimension)
        elif self.index_type.lower() == "ivf":
            # Inverted File index (approximate search, faster for large datasets)
            quantizer = faiss.IndexFlatL2(self.dimension)
            return faiss.IndexIVFFlat(quantizer, self.dimension, 100)  # 100 clusters
        elif self.index_type.lower() == "hnsw":
            # Hierarchical Navigable Small World (very fast approximate search)
            return faiss.IndexHNSWFlat(self.dimension, 32)
        else:
            logger.warning(f"Unknown index type {self.index_type}, using flat index")
            return faiss.IndexFlatL2(self.dimension)
    
    def add_documents(self, 
                     chunks: List[DocumentChunk], 
                     metadata: DocumentMetadata,
                     batch_size: int = 32) -> None:
        """
        Add document chunks to the search index
        
        Args:
            chunks: List of document chunks to add
            metadata: Document metadata
            batch_size: Batch size for embedding generation
        """
        try:
            logger.info(f"Adding {len(chunks)} chunks to index")
            
            # Store metadata
            self.document_metadata[metadata.file_path] = metadata
            
            # Generate embeddings in batches
            all_embeddings = []
            texts = [chunk.text for chunk in chunks]
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_embeddings = self.model.encode(
                    batch_texts,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False
                )
                all_embeddings.extend(batch_embeddings)
            
            # Convert to numpy array
            embeddings_array = np.array(all_embeddings).astype('float32')
            
            # Add to FAISS index
            if self.index_type.lower() == "ivf" and not self.index.is_trained:
                # Train IVF index if not already trained
                logger.info("Training IVF index...")
                self.index.train(embeddings_array)
            
            self.index.add(embeddings_array)
            
            # Store chunk metadata
            self.chunks_metadata.extend(chunks)
            
            logger.info(f"Successfully added {len(chunks)} chunks. Total index size: {self.index.ntotal}")
            
        except Exception as e:
            logger.error(f"Error adding documents to index: {str(e)}")
            raise
    
    def search(self, 
               query: str, 
               top_k: int = 10,
               similarity_threshold: float = 0.0,
               domain_filter: Optional[str] = None) -> List[SearchResult]:
        """
        Search for relevant chunks based on query
        
        Args:
            query: Search query
            top_k: Number of top results to return
            similarity_threshold: Minimum similarity score threshold
            domain_filter: Optional domain filter for results
            
        Returns:
            List of SearchResult objects
        """
        try:
            if self.index.ntotal == 0:
                logger.warning("Index is empty, no documents to search")
                return []
            
            # Generate query embedding
            query_embedding = self.model.encode(
                [query],
                convert_to_numpy=True,
                normalize_embeddings=True
            ).astype('float32')
            
            # Search in FAISS index
            similarities, indices = self.index.search(query_embedding, top_k)
            
            # Convert to search results
            results = []
            for rank, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
                if idx >= len(self.chunks_metadata):
                    continue
                    
                # Convert FAISS L2 distance to cosine similarity
                # For normalized vectors: cosine_sim = 1 - (L2_distance^2 / 2)
                cosine_similarity = 1 - (similarity / 2)
                
                if cosine_similarity >= similarity_threshold:
                    chunk = self.chunks_metadata[idx]
                    
                    # Apply domain filter if specified
                    if domain_filter:
                        # Simple domain filtering based on document content
                        if not self._matches_domain(chunk.text, domain_filter):
                            continue
                    
                    results.append(SearchResult(
                        chunk=chunk,
                        similarity_score=float(cosine_similarity),
                        rank=rank
                    ))
            
            logger.info(f"Found {len(results)} results for query: {query[:50]}...")
            return results
            
        except Exception as e:
            logger.error(f"Error during search: {str(e)}")
            raise
    
    def _matches_domain(self, text: str, domain: str) -> bool:
        """Check if text matches the specified domain"""
        domain_keywords = {
            "insurance": ["policy", "coverage", "claim", "premium", "deductible", "benefit"],
            "legal": ["contract", "agreement", "liability", "terms", "conditions", "clause"],
            "hr": ["employee", "benefits", "leave", "performance", "workplace", "policy"],
            "compliance": ["regulation", "compliance", "audit", "gdpr", "privacy", "data protection"]
        }
        
        keywords = domain_keywords.get(domain.lower(), [])
        text_lower = text.lower()
        
        return any(keyword in text_lower for keyword in keywords)
    
    def get_document_statistics(self) -> Dict:
        """Get statistics about indexed documents"""
        stats = {
            "total_chunks": len(self.chunks_metadata),
            "total_documents": len(self.document_metadata),
            "index_size": self.index.ntotal,
            "model_name": self.model_name,
            "index_type": self.index_type,
            "dimension": self.dimension
        }
        
        # Document type breakdown
        doc_types = {}
        for metadata in self.document_metadata.values():
            doc_type = metadata.file_type
            doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
        
        stats["document_types"] = doc_types
        
        return stats
    
    def save_index(self, index_path: str) -> None:
        """Save the FAISS index and metadata to disk"""
        try:
            index_path = Path(index_path)
            index_path.mkdir(parents=True, exist_ok=True)
            
            # Save FAISS index
            faiss.write_index(self.index, str(index_path / "faiss_index.bin"))
            
            # Save metadata
            with open(index_path / "chunks_metadata.pkl", "wb") as f:
                pickle.dump(self.chunks_metadata, f)
            
            with open(index_path / "document_metadata.pkl", "wb") as f:
                pickle.dump(self.document_metadata, f)
            
            # Save configuration
            config = {
                "model_name": self.model_name,
                "index_type": self.index_type,
                "dimension": self.dimension
            }
            
            with open(index_path / "config.pkl", "wb") as f:
                pickle.dump(config, f)
            
            logger.info(f"Index saved to {index_path}")
            
        except Exception as e:
            logger.error(f"Error saving index: {str(e)}")
            raise
    
    def load_index(self, index_path: str) -> None:
        """Load the FAISS index and metadata from disk"""
        try:
            index_path = Path(index_path)
            
            if not index_path.exists():
                raise FileNotFoundError(f"Index path not found: {index_path}")
            
            # Load configuration
            with open(index_path / "config.pkl", "rb") as f:
                config = pickle.load(f)
            
            # Verify configuration matches
            if config["model_name"] != self.model_name:
                logger.warning(f"Model mismatch: expected {self.model_name}, found {config['model_name']}")
            
            # Load FAISS index
            self.index = faiss.read_index(str(index_path / "faiss_index.bin"))
            
            # Load metadata
            with open(index_path / "chunks_metadata.pkl", "rb") as f:
                self.chunks_metadata = pickle.load(f)
            
            with open(index_path / "document_metadata.pkl", "rb") as f:
                self.document_metadata = pickle.load(f)
            
            logger.info(f"Index loaded from {index_path}")
            logger.info(f"Loaded {len(self.chunks_metadata)} chunks from {len(self.document_metadata)} documents")
            
        except Exception as e:
            logger.error(f"Error loading index: {str(e)}")
            raise
    
    def rerank_results(self, 
                      query: str, 
                      results: List[SearchResult], 
                      rerank_top_k: int = 5) -> List[SearchResult]:
        """
        Rerank top results using cross-encoder for better relevance
        
        Args:
            query: Original query
            results: Initial search results
            rerank_top_k: Number of top results to rerank
            
        Returns:
            Reranked search results
        """
        if len(results) <= 1 or rerank_top_k <= 1:
            return results
        
        try:
            # For now, we'll use a simple text similarity approach
            # In a production system, you might use a cross-encoder model
            
            # Take top results for reranking
            top_results = results[:rerank_top_k]
            remaining_results = results[rerank_top_k:]
            
            # Simple keyword-based reranking
            query_words = set(query.lower().split())
            
            for result in top_results:
                chunk_words = set(result.chunk.text.lower().split())
                word_overlap = len(query_words.intersection(chunk_words))
                
                # Boost similarity score based on word overlap
                overlap_bonus = min(0.1 * word_overlap / len(query_words), 0.2)
                result.similarity_score += overlap_bonus
            
            # Re-sort top results
            top_results.sort(key=lambda x: x.similarity_score, reverse=True)
            
            # Update ranks
            for i, result in enumerate(top_results + remaining_results):
                result.rank = i
            
            return top_results + remaining_results
            
        except Exception as e:
            logger.error(f"Error during reranking: {str(e)}")
            return results
    
    def clear_index(self) -> None:
        """Clear the index and all metadata"""
        self.index = self._create_faiss_index()
        self.chunks_metadata.clear()
        self.document_metadata.clear()
        self.embedding_cache.clear()
        logger.info("Index cleared")
    
    def get_similar_chunks(self, chunk_id: str, top_k: int = 5) -> List[SearchResult]:
        """Find chunks similar to a specific chunk"""
        try:
            # Find the chunk
            target_chunk = None
            target_idx = None
            
            for idx, chunk in enumerate(self.chunks_metadata):
                if chunk.chunk_id == chunk_id:
                    target_chunk = chunk
                    target_idx = idx
                    break
            
            if target_chunk is None:
                logger.warning(f"Chunk {chunk_id} not found")
                return []
            
            # Get the embedding for this chunk
            chunk_embedding = self.model.encode(
                [target_chunk.text],
                convert_to_numpy=True,
                normalize_embeddings=True
            ).astype('float32')
            
            # Search for similar chunks
            similarities, indices = self.index.search(chunk_embedding, top_k + 1)  # +1 to exclude self
            
            results = []
            for rank, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
                if idx == target_idx:  # Skip the chunk itself
                    continue
                    
                if idx >= len(self.chunks_metadata):
                    continue
                
                cosine_similarity = 1 - (similarity / 2)
                chunk = self.chunks_metadata[idx]
                
                results.append(SearchResult(
                    chunk=chunk,
                    similarity_score=float(cosine_similarity),
                    rank=len(results)
                ))
                
                if len(results) >= top_k:
                    break
            
            return results
            
        except Exception as e:
            logger.error(f"Error finding similar chunks: {str(e)}")
            return []