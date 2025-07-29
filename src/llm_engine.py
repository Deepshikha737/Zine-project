import os
import logging
from typing import Dict, List, Optional, Tuple, Any
import json
import re
from dataclasses import dataclass
from datetime import datetime

from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    pipeline,
    Pipeline
)
import torch

from .embedding_engine import SearchResult
from .document_processor import DocumentChunk

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class QueryIntent:
    """Represents the intent and entities extracted from a query"""
    intent_type: str  # 'coverage_check', 'condition_lookup', 'comparison', 'explanation'
    domain: str  # 'insurance', 'legal', 'hr', 'compliance'
    entities: Dict[str, Any]  # extracted entities like procedures, conditions, etc.
    confidence: float

@dataclass
class DecisionResult:
    """Represents the final decision result with explanation"""
    answer: str
    confidence: float
    relevant_clauses: List[Dict]
    conditions: List[str]
    rationale: str
    document_sources: List[str]
    decision_type: str

class LLMEngine:
    """Free LLM-powered engine for natural language understanding and decision making"""
    
    def __init__(self, 
                 model_name: str = "microsoft/DialoGPT-medium",
                 max_length: int = 512,
                 device: str = None):
        """
        Initialize the LLM engine with a free model
        
        Args:
            model_name: Name of the free/open source model to use
            max_length: Maximum token length for generation
            device: Device to run the model on (auto-detected if None)
        """
        self.model_name = model_name
        self.max_length = max_length
        
        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        logger.info(f"Initializing LLM engine with {model_name} on {self.device}")
        
        # Initialize the model and tokenizer
        self._load_model()
        
        # Domain-specific patterns and templates
        self._init_domain_patterns()
        
    def _load_model(self):
        """Load the tokenizer and model"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                low_cpu_mem_usage=True
            )
            self.model.to(self.device)
            
            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Create text generation pipeline
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1,
                max_length=self.max_length,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            # Fallback to a simpler approach
            self._init_fallback_model()
    
    def _init_fallback_model(self):
        """Initialize a fallback model or rule-based system"""
        logger.warning("Using fallback rule-based system")
        self.generator = None
        self.model = None
        self.tokenizer = None
    
    def _init_domain_patterns(self):
        """Initialize domain-specific patterns for entity extraction"""
        self.domain_patterns = {
            "insurance": {
                "intent_patterns": {
                    "coverage_check": [
                        r"does.*cover",
                        r"is.*covered",
                        r"coverage.*for",
                        r"what.*covered"
                    ],
                    "condition_lookup": [
                        r"what.*conditions",
                        r"requirements.*for",
                        r"eligibility",
                        r"prerequisites"
                    ],
                    "limit_check": [
                        r"what.*limit",
                        r"maximum.*coverage",
                        r"how much.*covered"
                    ]
                },
                "entities": {
                    "procedures": [
                        r"surgery", r"operation", r"procedure", r"treatment",
                        r"knee surgery", r"dental", r"vision", r"prescription"
                    ],
                    "conditions": [
                        r"pre-existing", r"chronic", r"acute", r"emergency",
                        r"preventive", r"diagnostic"
                    ],
                    "body_parts": [
                        r"knee", r"back", r"heart", r"eye", r"tooth", r"dental"
                    ]
                }
            },
            "legal": {
                "intent_patterns": {
                    "clause_lookup": [
                        r"what.*clause", r"find.*terms", r"contract.*states"
                    ],
                    "liability_check": [
                        r"liable.*for", r"responsibility", r"who.*responsible"
                    ],
                    "termination_check": [
                        r"terminate", r"end.*contract", r"cancel"
                    ]
                },
                "entities": {
                    "legal_terms": [
                        r"liability", r"indemnification", r"breach", r"damages",
                        r"termination", r"assignment", r"confidentiality"
                    ]
                }
            },
            "hr": {
                "intent_patterns": {
                    "policy_lookup": [
                        r"what.*policy", r"rules.*about", r"allowed.*to"
                    ],
                    "benefit_check": [
                        r"benefits.*include", r"entitled.*to", r"what.*benefits"
                    ],
                    "leave_check": [
                        r"leave.*policy", r"time.*off", r"vacation", r"sick.*leave"
                    ]
                },
                "entities": {
                    "hr_terms": [
                        r"vacation", r"sick leave", r"maternity", r"paternity",
                        r"performance", r"benefits", r"remote work"
                    ]
                }
            }
        }
    
    def analyze_query(self, query: str) -> QueryIntent:
        """Analyze the query to extract intent and entities"""
        query_lower = query.lower()
        
        # Detect domain
        domain = self._detect_domain(query_lower)
        
        # Extract intent
        intent_type = self._extract_intent(query_lower, domain)
        
        # Extract entities
        entities = self._extract_entities(query_lower, domain)
        
        # Calculate confidence
        confidence = self._calculate_intent_confidence(query_lower, domain, intent_type)
        
        return QueryIntent(
            intent_type=intent_type,
            domain=domain,
            entities=entities,
            confidence=confidence
        )
    
    def _detect_domain(self, query: str) -> str:
        """Detect the domain of the query"""
        domain_keywords = {
            "insurance": ["policy", "coverage", "claim", "deductible", "premium", "benefit", "medical", "health"],
            "legal": ["contract", "agreement", "liability", "terms", "clause", "legal", "law"],
            "hr": ["employee", "benefits", "leave", "vacation", "sick", "performance", "workplace"],
            "compliance": ["regulation", "compliance", "audit", "gdpr", "privacy", "data"]
        }
        
        domain_scores = {}
        for domain, keywords in domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword in query)
            domain_scores[domain] = score
        
        # Return the domain with highest score, default to insurance
        return max(domain_scores, key=domain_scores.get) if max(domain_scores.values()) > 0 else "insurance"
    
    def _extract_intent(self, query: str, domain: str) -> str:
        """Extract the intent from the query"""
        if domain not in self.domain_patterns:
            return "general_query"
        
        patterns = self.domain_patterns[domain]["intent_patterns"]
        
        for intent, intent_patterns in patterns.items():
            for pattern in intent_patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    return intent
        
        return "general_query"
    
    def _extract_entities(self, query: str, domain: str) -> Dict[str, Any]:
        """Extract entities from the query"""
        entities = {}
        
        if domain not in self.domain_patterns:
            return entities
        
        entity_patterns = self.domain_patterns[domain].get("entities", {})
        
        for entity_type, patterns in entity_patterns.items():
            found_entities = []
            for pattern in patterns:
                matches = re.findall(pattern, query, re.IGNORECASE)
                found_entities.extend(matches)
            
            if found_entities:
                entities[entity_type] = list(set(found_entities))  # Remove duplicates
        
        return entities
    
    def _calculate_intent_confidence(self, query: str, domain: str, intent_type: str) -> float:
        """Calculate confidence score for the intent classification"""
        base_confidence = 0.5
        
        # Boost confidence based on keyword matches
        if domain in self.domain_patterns:
            patterns = self.domain_patterns[domain]["intent_patterns"].get(intent_type, [])
            matches = sum(1 for pattern in patterns if re.search(pattern, query, re.IGNORECASE))
            if matches > 0:
                base_confidence += min(0.3 * matches, 0.4)
        
        # Boost confidence based on query length and specificity
        word_count = len(query.split())
        if word_count > 5:
            base_confidence += 0.1
        
        return min(base_confidence, 1.0)
    
    def generate_answer(self, 
                       query: str,
                       search_results: List[SearchResult],
                       query_intent: QueryIntent) -> DecisionResult:
        """Generate a comprehensive answer based on search results and query intent"""
        
        if not search_results:
            return self._generate_no_results_answer(query, query_intent)
        
        # Extract relevant information from search results
        relevant_text = self._extract_relevant_context(search_results, query_intent)
        
        # Generate the main answer
        answer = self._generate_main_answer(query, relevant_text, query_intent)
        
        # Extract conditions and requirements
        conditions = self._extract_conditions(relevant_text, query_intent)
        
        # Generate rationale
        rationale = self._generate_rationale(query, relevant_text, answer, query_intent)
        
        # Prepare relevant clauses
        relevant_clauses = self._prepare_relevant_clauses(search_results)
        
        # Extract document sources
        document_sources = list(set([result.chunk.document_id for result in search_results[:5]]))
        
        # Calculate overall confidence
        confidence = self._calculate_answer_confidence(search_results, query_intent, answer)
        
        return DecisionResult(
            answer=answer,
            confidence=confidence,
            relevant_clauses=relevant_clauses,
            conditions=conditions,
            rationale=rationale,
            document_sources=document_sources,
            decision_type=query_intent.intent_type
        )
    
    def _extract_relevant_context(self, search_results: List[SearchResult], query_intent: QueryIntent) -> str:
        """Extract the most relevant context from search results"""
        # Take top 3 results for context
        top_results = search_results[:3]
        
        context_parts = []
        for result in top_results:
            # Add chunk text with some metadata
            context_parts.append(f"[Source: {result.chunk.document_id}]\n{result.chunk.text}")
        
        return "\n\n".join(context_parts)
    
    def _generate_main_answer(self, query: str, context: str, query_intent: QueryIntent) -> str:
        """Generate the main answer using LLM or rule-based approach"""
        
        if self.generator is not None:
            return self._llm_generate_answer(query, context, query_intent)
        else:
            return self._rule_based_answer(query, context, query_intent)
    
    def _llm_generate_answer(self, query: str, context: str, query_intent: QueryIntent) -> str:
        """Generate answer using the LLM"""
        try:
            # Create a prompt for the LLM
            prompt = self._create_answer_prompt(query, context, query_intent)
            
            # Generate response
            response = self.generator(
                prompt,
                max_length=len(self.tokenizer.encode(prompt)) + 100,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Extract the generated text
            generated_text = response[0]['generated_text']
            answer = generated_text[len(prompt):].strip()
            
            # Clean up the answer
            answer = self._clean_generated_answer(answer)
            
            return answer if answer else self._rule_based_answer(query, context, query_intent)
            
        except Exception as e:
            logger.error(f"Error in LLM generation: {str(e)}")
            return self._rule_based_answer(query, context, query_intent)
    
    def _create_answer_prompt(self, query: str, context: str, query_intent: QueryIntent) -> str:
        """Create a prompt for answer generation"""
        prompt_templates = {
            "coverage_check": f"""Based on the policy information below, answer whether the requested item is covered.

Policy Context:
{context[:1000]}

Question: {query}

Answer: """,
            
            "condition_lookup": f"""Based on the document information below, list the conditions and requirements.

Document Context:
{context[:1000]}

Question: {query}

Answer: """,
            
            "general_query": f"""Based on the document information below, provide a helpful answer.

Document Context:
{context[:1000]}

Question: {query}

Answer: """
        }
        
        template_key = query_intent.intent_type if query_intent.intent_type in prompt_templates else "general_query"
        return prompt_templates[template_key]
    
    def _rule_based_answer(self, query: str, context: str, query_intent: QueryIntent) -> str:
        """Generate answer using rule-based approach as fallback"""
        
        # Simple pattern-based answer generation
        query_lower = query.lower()
        context_lower = context.lower()
        
        if query_intent.intent_type == "coverage_check":
            # Check for coverage patterns
            if any(word in context_lower for word in ["covered", "includes", "benefit"]):
                if any(negative in context_lower for negative in ["not covered", "excluded", "limitation"]):
                    return "Coverage may be limited or excluded. Please review the specific terms and conditions."
                else:
                    return "Yes, this appears to be covered based on the policy terms found."
            else:
                return "Coverage information is not clearly specified in the available documents."
        
        elif query_intent.intent_type == "condition_lookup":
            # Look for condition patterns
            conditions_found = []
            condition_keywords = ["required", "must", "shall", "condition", "requirement", "prerequisite"]
            
            sentences = context.split('.')
            for sentence in sentences:
                if any(keyword in sentence.lower() for keyword in condition_keywords):
                    conditions_found.append(sentence.strip())
            
            if conditions_found:
                return f"The following conditions apply: {'; '.join(conditions_found[:3])}"
            else:
                return "No specific conditions were found in the available documents."
        
        else:
            # General answer based on context similarity
            if context.strip():
                return "Based on the available information, please refer to the relevant sections in the document for detailed information."
            else:
                return "I couldn't find specific information to answer your question in the available documents."
    
    def _clean_generated_answer(self, answer: str) -> str:
        """Clean up the generated answer"""
        # Remove common artifacts
        answer = answer.replace("Answer:", "").strip()
        
        # Stop at first complete sentence if answer is too long
        sentences = answer.split('.')
        if len(sentences) > 3:
            answer = '.'.join(sentences[:3]) + '.'
        
        # Remove incomplete sentences at the end
        if answer and not answer.endswith('.'):
            last_period = answer.rfind('.')
            if last_period > 0:
                answer = answer[:last_period + 1]
        
        return answer.strip()
    
    def _extract_conditions(self, context: str, query_intent: QueryIntent) -> List[str]:
        """Extract conditions and requirements from context"""
        conditions = []
        
        # Pattern-based condition extraction
        condition_patterns = [
            r"must\s+([^.]+)",
            r"required\s+to\s+([^.]+)",
            r"shall\s+([^.]+)",
            r"prerequisite[s]?\s*:?\s*([^.]+)",
            r"condition[s]?\s*:?\s*([^.]+)"
        ]
        
        for pattern in condition_patterns:
            matches = re.finditer(pattern, context, re.IGNORECASE)
            for match in matches:
                condition = match.group(1).strip()
                if len(condition) > 10 and len(condition) < 200:  # Reasonable length
                    conditions.append(condition)
        
        # Remove duplicates and limit results
        return list(set(conditions))[:5]
    
    def _generate_rationale(self, query: str, context: str, answer: str, query_intent: QueryIntent) -> str:
        """Generate explanation for the decision"""
        
        if query_intent.intent_type == "coverage_check":
            return f"This determination is based on the policy language found in the documents, which addresses {query_intent.domain} coverage requirements."
        
        elif query_intent.intent_type == "condition_lookup":
            return f"The conditions listed are extracted from the relevant policy sections that govern {query_intent.domain} requirements."
        
        else:
            return f"This answer is based on information found in the {query_intent.domain} documents that most closely match your query."
    
    def _prepare_relevant_clauses(self, search_results: List[SearchResult]) -> List[Dict]:
        """Prepare relevant clauses from search results"""
        clauses = []
        
        for i, result in enumerate(search_results[:5]):
            clause = {
                "clause_id": f"CLAUSE_{i+1}",
                "text": result.chunk.text[:500] + "..." if len(result.chunk.text) > 500 else result.chunk.text,
                "relevance_score": result.similarity_score,
                "source": result.chunk.document_id
            }
            clauses.append(clause)
        
        return clauses
    
    def _calculate_answer_confidence(self, search_results: List[SearchResult], query_intent: QueryIntent, answer: str) -> float:
        """Calculate overall confidence in the answer"""
        base_confidence = 0.3
        
        # Factor in search result quality
        if search_results:
            avg_similarity = sum(r.similarity_score for r in search_results[:3]) / min(len(search_results), 3)
            base_confidence += 0.4 * avg_similarity
        
        # Factor in intent confidence
        base_confidence += 0.2 * query_intent.confidence
        
        # Factor in answer quality (simple heuristics)
        if len(answer) > 20 and not answer.startswith("I couldn't"):
            base_confidence += 0.1
        
        return min(base_confidence, 1.0)
    
    def _generate_no_results_answer(self, query: str, query_intent: QueryIntent) -> DecisionResult:
        """Generate answer when no search results are found"""
        return DecisionResult(
            answer="I couldn't find relevant information in the available documents to answer your question.",
            confidence=0.1,
            relevant_clauses=[],
            conditions=[],
            rationale="No relevant documents were found in the knowledge base.",
            document_sources=[],
            decision_type=query_intent.intent_type
        )