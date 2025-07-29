import os
import logging
from typing import Dict, List, Optional, Any
import json
from pathlib import Path
from dataclasses import asdict

from .document_processor import DocumentProcessor, DocumentChunk, DocumentMetadata
from .embedding_engine import EmbeddingEngine, SearchResult
from .llm_engine import LLMEngine, QueryIntent, DecisionResult

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IntelligentQuerySystem:
    """
    Main system that integrates document processing, semantic search, and LLM reasoning
    for intelligent query handling in insurance, legal, HR, and compliance domains.
    """
    
    def __init__(self, 
                 config: Optional[Dict] = None,
                 index_path: str = "./data/index",
                 documents_path: str = "./data/documents"):
        """
        Initialize the intelligent query system
        
        Args:
            config: Configuration dictionary for customizing components
            index_path: Path to store/load the search index
            documents_path: Path to document storage
        """
        self.config = config or self._get_default_config()
        self.index_path = Path(index_path)
        self.documents_path = Path(documents_path)
        
        # Create directories
        self.index_path.mkdir(parents=True, exist_ok=True)
        self.documents_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self._initialize_components()
        
        # Load existing index if available
        self._load_existing_index()
        
        logger.info("Intelligent Query System initialized successfully")
    
    def _get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
            "document_processor": {
                "chunk_size": 1000,
                "chunk_overlap": 200
            },
            "embedding_engine": {
                "model_name": "sentence-transformers/all-MiniLM-L6-v2",
                "index_type": "flat",
                "dimension": 384
            },
            "llm_engine": {
                "model_name": "microsoft/DialoGPT-medium",
                "max_length": 512
            },
            "search": {
                "top_k": 10,
                "similarity_threshold": 0.3,
                "rerank_top_k": 5
            }
        }
    
    def _initialize_components(self):
        """Initialize all system components"""
        # Document processor
        doc_config = self.config["document_processor"]
        self.document_processor = DocumentProcessor(
            chunk_size=doc_config["chunk_size"],
            chunk_overlap=doc_config["chunk_overlap"]
        )
        
        # Embedding engine
        emb_config = self.config["embedding_engine"]
        self.embedding_engine = EmbeddingEngine(
            model_name=emb_config["model_name"],
            index_type=emb_config["index_type"],
            dimension=emb_config["dimension"]
        )
        
        # LLM engine
        llm_config = self.config["llm_engine"]
        self.llm_engine = LLMEngine(
            model_name=llm_config["model_name"],
            max_length=llm_config["max_length"]
        )
    
    def _load_existing_index(self):
        """Load existing index if available"""
        try:
            if (self.index_path / "faiss_index.bin").exists():
                logger.info("Loading existing search index...")
                self.embedding_engine.load_index(str(self.index_path))
                logger.info(f"Loaded index with {self.embedding_engine.index.ntotal} documents")
        except Exception as e:
            logger.warning(f"Could not load existing index: {e}")
    
    def add_document(self, file_path: str, domain: str = "insurance") -> Dict[str, Any]:
        """
        Add a document to the system
        
        Args:
            file_path: Path to the document file
            domain: Domain category (insurance, legal, hr, compliance)
            
        Returns:
            Dictionary with processing results
        """
        try:
            logger.info(f"Processing document: {file_path}")
            
            # Process the document
            chunks, metadata = self.document_processor.process_document(file_path)
            
            # Add domain information to metadata
            metadata.subject = f"{metadata.subject} | Domain: {domain}" if metadata.subject else f"Domain: {domain}"
            
            # Add to embedding index
            self.embedding_engine.add_documents(chunks, metadata)
            
            # Save updated index
            self.embedding_engine.save_index(str(self.index_path))
            
            result = {
                "success": True,
                "document_id": metadata.file_path,
                "chunks_created": len(chunks),
                "domain": domain,
                "metadata": asdict(metadata)
            }
            
            logger.info(f"Successfully processed document with {len(chunks)} chunks")
            return result
            
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "document_id": file_path
            }
    
    def process_query(self, query: str, domain: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a natural language query and return structured results
        
        Args:
            query: Natural language query
            domain: Optional domain filter
            
        Returns:
            Structured JSON response with answer and explanations
        """
        try:
            logger.info(f"Processing query: {query[:100]}...")
            
            # Analyze query intent
            query_intent = self.llm_engine.analyze_query(query)
            logger.info(f"Detected intent: {query_intent.intent_type}, domain: {query_intent.domain}")
            
            # Use detected domain or provided domain
            effective_domain = domain or query_intent.domain
            
            # Search for relevant documents
            search_config = self.config["search"]
            search_results = self.embedding_engine.search(
                query=query,
                top_k=search_config["top_k"],
                similarity_threshold=search_config["similarity_threshold"],
                domain_filter=effective_domain
            )
            
            # Rerank results for better relevance
            if search_results:
                search_results = self.embedding_engine.rerank_results(
                    query=query,
                    results=search_results,
                    rerank_top_k=search_config["rerank_top_k"]
                )
            
            # Generate decision using LLM
            decision_result = self.llm_engine.generate_answer(
                query=query,
                search_results=search_results,
                query_intent=query_intent
            )
            
            # Format response
            response = self._format_response(query, query_intent, decision_result, search_results)
            
            logger.info(f"Query processed successfully with confidence: {decision_result.confidence:.2f}")
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "query": query
            }
    
    def _format_response(self, 
                        query: str, 
                        query_intent: QueryIntent, 
                        decision_result: DecisionResult, 
                        search_results: List[SearchResult]) -> Dict[str, Any]:
        """Format the final response"""
        
        # Extract unique document sources
        document_sources = []
        for result in search_results[:5]:
            doc_path = result.chunk.document_id
            if doc_path not in document_sources:
                doc_metadata = self.embedding_engine.document_metadata.get(doc_path)
                source_info = {
                    "document_id": doc_path,
                    "file_name": Path(doc_path).name if doc_metadata else "Unknown",
                    "file_type": doc_metadata.file_type if doc_metadata else "Unknown"
                }
                document_sources.append(source_info)
        
        response = {
            "success": True,
            "query": query,
            "answer": decision_result.answer,
            "confidence": round(decision_result.confidence, 3),
            "query_analysis": {
                "intent_type": query_intent.intent_type,
                "detected_domain": query_intent.domain,
                "entities": query_intent.entities,
                "intent_confidence": round(query_intent.confidence, 3)
            },
            "relevant_clauses": decision_result.relevant_clauses,
            "conditions": decision_result.conditions,
            "rationale": decision_result.rationale,
            "document_sources": document_sources,
            "decision_type": decision_result.decision_type,
            "search_results_count": len(search_results),
            "timestamp": self._get_timestamp()
        }
        
        return response
    
    def get_document_statistics(self) -> Dict[str, Any]:
        """Get statistics about indexed documents"""
        stats = self.embedding_engine.get_document_statistics()
        
        # Add domain breakdown
        domain_breakdown = {}
        for doc_path, metadata in self.embedding_engine.document_metadata.items():
            domain = "unknown"
            if metadata.subject and "Domain:" in metadata.subject:
                domain = metadata.subject.split("Domain:")[-1].strip()
            
            domain_breakdown[domain] = domain_breakdown.get(domain, 0) + 1
        
        stats["domain_breakdown"] = domain_breakdown
        stats["system_config"] = self.config
        
        return stats
    
    def search_similar_content(self, content: str, top_k: int = 5) -> List[Dict]:
        """Search for content similar to the provided text"""
        try:
            search_results = self.embedding_engine.search(
                query=content,
                top_k=top_k,
                similarity_threshold=0.0
            )
            
            results = []
            for result in search_results:
                results.append({
                    "text": result.chunk.text[:200] + "..." if len(result.chunk.text) > 200 else result.chunk.text,
                    "similarity_score": round(result.similarity_score, 3),
                    "document_source": result.chunk.document_id,
                    "chunk_id": result.chunk.chunk_id
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error in similarity search: {str(e)}")
            return []
    
    def get_document_content(self, document_id: str) -> Optional[Dict]:
        """Get content and metadata for a specific document"""
        try:
            metadata = self.embedding_engine.document_metadata.get(document_id)
            if not metadata:
                return None
            
            # Get all chunks for this document
            chunks = [
                chunk for chunk in self.embedding_engine.chunks_metadata 
                if chunk.document_id == document_id
            ]
            
            return {
                "metadata": asdict(metadata),
                "chunks_count": len(chunks),
                "chunks": [
                    {
                        "chunk_id": chunk.chunk_id,
                        "text": chunk.text,
                        "page_number": chunk.page_number
                    } for chunk in chunks[:10]  # Limit to first 10 chunks
                ]
            }
            
        except Exception as e:
            logger.error(f"Error retrieving document content: {str(e)}")
            return None
    
    def clear_index(self) -> Dict[str, Any]:
        """Clear all indexed documents"""
        try:
            self.embedding_engine.clear_index()
            
            # Remove index files
            import shutil
            if self.index_path.exists():
                shutil.rmtree(self.index_path)
                self.index_path.mkdir(parents=True, exist_ok=True)
            
            return {"success": True, "message": "Index cleared successfully"}
            
        except Exception as e:
            logger.error(f"Error clearing index: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def update_config(self, new_config: Dict) -> Dict[str, Any]:
        """Update system configuration"""
        try:
            self.config.update(new_config)
            logger.info("Configuration updated")
            return {"success": True, "message": "Configuration updated"}
            
        except Exception as e:
            logger.error(f"Error updating configuration: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def health_check(self) -> Dict[str, Any]:
        """Perform system health check"""
        try:
            health_status = {
                "status": "healthy",
                "components": {
                    "document_processor": "ok",
                    "embedding_engine": "ok",
                    "llm_engine": "ok"
                },
                "index_size": self.embedding_engine.index.ntotal,
                "documents_count": len(self.embedding_engine.document_metadata),
                "config": self.config
            }
            
            # Test embedding generation
            try:
                test_embedding = self.embedding_engine.model.encode(["test"])
                health_status["components"]["embedding_model"] = "ok"
            except:
                health_status["components"]["embedding_model"] = "error"
                health_status["status"] = "degraded"
            
            # Test LLM availability
            if self.llm_engine.generator is None:
                health_status["components"]["llm_engine"] = "fallback_mode"
                health_status["status"] = "degraded"
            
            return health_status
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def export_knowledge_base(self, export_path: str) -> Dict[str, Any]:
        """Export the knowledge base for backup or sharing"""
        try:
            export_path = Path(export_path)
            export_path.mkdir(parents=True, exist_ok=True)
            
            # Export index
            self.embedding_engine.save_index(str(export_path / "index"))
            
            # Export configuration
            with open(export_path / "config.json", "w") as f:
                json.dump(self.config, f, indent=2)
            
            # Export statistics
            stats = self.get_document_statistics()
            with open(export_path / "statistics.json", "w") as f:
                json.dump(stats, f, indent=2)
            
            return {
                "success": True,
                "export_path": str(export_path),
                "files_exported": ["index/", "config.json", "statistics.json"]
            }
            
        except Exception as e:
            logger.error(f"Error exporting knowledge base: {str(e)}")
            return {"success": False, "error": str(e)}