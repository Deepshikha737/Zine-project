# LLM-Powered Intelligent Query-Retrieval System

An advanced document processing and intelligent query system designed for insurance, legal, HR, and compliance domains. This system uses free LLMs and semantic search to provide contextual decisions with explainable rationale.

## Features

- **Multi-format Document Processing**: Supports PDF, DOCX, and email documents
- **Semantic Search**: FAISS-based vector search with sentence transformers
- **Free LLM Integration**: Uses open-source models for natural language understanding
- **Explainable AI**: Provides clear rationale for decisions and clause matching
- **Domain-Specific**: Optimized for insurance, legal, HR, and compliance use cases
- **Structured Responses**: Returns JSON with confidence scores and explanations

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Document      │    │   Embedding      │    │   Query         │
│   Processors    │───▶│   Generation     │───▶│   Engine        │
│                 │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Text          │    │   FAISS Vector   │    │   Decision      │
│   Extraction    │    │   Database       │    │   Engine        │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download required models (automatic on first run):
- sentence-transformers/all-MiniLM-L6-v2 (embeddings)
- microsoft/DialoGPT-medium (free LLM)

## Quick Start

1. Start the API server:
```bash
python app.py
```

2. Upload documents and run queries through the web interface at `http://localhost:8000`

3. Or use the API directly:
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "Does this policy cover knee surgery, and what are the conditions?"}'
```

## Sample Query Response

```json
{
  "query": "Does this policy cover knee surgery, and what are the conditions?",
  "answer": "Yes, knee surgery is covered under the surgical benefits section.",
  "confidence": 0.92,
  "relevant_clauses": [
    {
      "clause_id": "SEC_4_2_1",
      "text": "Surgical procedures including orthopedic surgeries are covered...",
      "relevance_score": 0.95
    }
  ],
  "conditions": [
    "Prior authorization required",
    "Must be medically necessary",
    "Network provider preferred"
  ],
  "rationale": "The policy explicitly covers orthopedic surgeries under Section 4.2.1...",
  "document_sources": ["policy_doc_123.pdf"]
}
```

## Usage Examples

### Insurance Domain
- "What is the coverage limit for dental procedures?"
- "Does this policy cover pre-existing conditions?"
- "What are the deductible requirements for emergency care?"

### Legal Domain
- "What are the termination clauses in this contract?"
- "Are there any liability limitations mentioned?"
- "What are the dispute resolution procedures?"

### HR Domain
- "What is the leave policy for medical emergencies?"
- "Are remote work arrangements permitted?"
- "What are the performance review criteria?"

### Compliance Domain
- "Does this process meet GDPR requirements?"
- "What are the data retention policies?"
- "Are there any regulatory compliance gaps?"

## API Endpoints

- `POST /upload` - Upload documents (PDF, DOCX, email)
- `POST /query` - Submit natural language queries
- `GET /documents` - List processed documents
- `GET /health` - System health check

## Configuration

Edit `config.yaml` to customize:
- Embedding model selection
- FAISS index parameters
- LLM model configuration
- Domain-specific settings

## Development

Run tests:
```bash
python -m pytest tests/
```

Start development server:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## License

MIT License - See LICENSE file for details.