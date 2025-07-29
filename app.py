from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
import tempfile
import os
import shutil
from pathlib import Path
import logging

from src.intelligent_query_system import IntelligentQuerySystem

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the intelligent query system
query_system = None

def get_query_system():
    """Get the global query system instance"""
    global query_system
    if query_system is None:
        query_system = IntelligentQuerySystem()
    return query_system

# FastAPI app
app = FastAPI(
    title="LLM-Powered Intelligent Query-Retrieval System",
    description="Advanced document processing and intelligent query system for insurance, legal, HR, and compliance domains",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class QueryRequest(BaseModel):
    query: str
    domain: Optional[str] = None

class QueryResponse(BaseModel):
    success: bool
    query: str
    answer: str
    confidence: float
    query_analysis: Dict[str, Any]
    relevant_clauses: List[Dict]
    conditions: List[str]
    rationale: str
    document_sources: List[Dict]
    decision_type: str
    search_results_count: int
    timestamp: str

class DocumentUploadResponse(BaseModel):
    success: bool
    document_id: str
    chunks_created: int
    domain: str
    metadata: Dict[str, Any]
    message: str

class SystemStats(BaseModel):
    total_chunks: int
    total_documents: int
    index_size: int
    model_name: str
    document_types: Dict[str, int]
    domain_breakdown: Dict[str, int]

# Static files for web interface
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main web interface"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Intelligent Query-Retrieval System</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f5f7fa; }
            .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
            .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 12px; margin-bottom: 30px; }
            .header h1 { font-size: 2.5rem; margin-bottom: 10px; }
            .header p { font-size: 1.1rem; opacity: 0.9; }
            .card { background: white; border-radius: 12px; padding: 30px; margin-bottom: 30px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
            .section-title { font-size: 1.5rem; color: #333; margin-bottom: 20px; border-bottom: 2px solid #667eea; padding-bottom: 10px; }
            .form-group { margin-bottom: 20px; }
            label { display: block; margin-bottom: 8px; font-weight: 600; color: #555; }
            input, select, textarea { width: 100%; padding: 12px; border: 2px solid #e1e5e9; border-radius: 8px; font-size: 16px; }
            input:focus, select:focus, textarea:focus { outline: none; border-color: #667eea; }
            .btn { background: #667eea; color: white; padding: 12px 24px; border: none; border-radius: 8px; cursor: pointer; font-size: 16px; font-weight: 600; }
            .btn:hover { background: #5a6fd8; }
            .btn:disabled { background: #ccc; cursor: not-allowed; }
            .result-container { background: #f8f9fa; border-radius: 8px; padding: 20px; margin-top: 20px; }
            .confidence-bar { background: #e9ecef; height: 20px; border-radius: 10px; overflow: hidden; margin: 10px 0; }
            .confidence-fill { background: linear-gradient(90deg, #28a745, #ffc107, #dc3545); height: 100%; transition: width 0.3s ease; }
            .clause { background: white; border-left: 4px solid #667eea; padding: 15px; margin: 10px 0; border-radius: 4px; }
            .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; }
            .stat-card { background: white; padding: 20px; border-radius: 8px; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            .stat-value { font-size: 2rem; font-weight: bold; color: #667eea; }
            .stat-label { color: #666; margin-top: 5px; }
            .loading { display: none; text-align: center; margin: 20px 0; }
            .spinner { border: 4px solid #f3f3f3; border-top: 4px solid #667eea; border-radius: 50%; width: 40px; height: 40px; animation: spin 1s linear infinite; margin: 0 auto; }
            @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🧠 Intelligent Query-Retrieval System</h1>
                <p>LLM-powered document analysis for Insurance, Legal, HR, and Compliance domains</p>
            </div>

            <div class="card">
                <h2 class="section-title">📄 Document Upload</h2>
                <form id="uploadForm">
                    <div class="form-group">
                        <label for="document">Select Document (PDF, DOCX, TXT)</label>
                        <input type="file" id="document" accept=".pdf,.docx,.doc,.txt,.eml" required>
                    </div>
                    <div class="form-group">
                        <label for="domain">Domain</label>
                        <select id="domain" required>
                            <option value="insurance">Insurance</option>
                            <option value="legal">Legal</option>
                            <option value="hr">HR</option>
                            <option value="compliance">Compliance</option>
                        </select>
                    </div>
                    <button type="submit" class="btn">📤 Upload Document</button>
                </form>
                <div id="uploadResult" class="result-container" style="display: none;"></div>
            </div>

            <div class="card">
                <h2 class="section-title">❓ Query Interface</h2>
                <form id="queryForm">
                    <div class="form-group">
                        <label for="query">Enter your question</label>
                        <textarea id="query" rows="3" placeholder="e.g., Does this policy cover knee surgery, and what are the conditions?" required></textarea>
                    </div>
                    <div class="form-group">
                        <label for="queryDomain">Domain Filter (optional)</label>
                        <select id="queryDomain">
                            <option value="">Auto-detect</option>
                            <option value="insurance">Insurance</option>
                            <option value="legal">Legal</option>
                            <option value="hr">HR</option>
                            <option value="compliance">Compliance</option>
                        </select>
                    </div>
                    <button type="submit" class="btn">🔍 Search</button>
                </form>
                
                <div class="loading" id="queryLoading">
                    <div class="spinner"></div>
                    <p>Processing your query...</p>
                </div>
                
                <div id="queryResult" class="result-container" style="display: none;"></div>
            </div>

            <div class="card">
                <h2 class="section-title">📊 System Statistics</h2>
                <div id="statsContainer">
                    <button class="btn" onclick="loadStats()">🔄 Load Statistics</button>
                </div>
            </div>
        </div>

        <script>
            // Document upload
            document.getElementById('uploadForm').addEventListener('submit', async (e) => {
                e.preventDefault();
                
                const formData = new FormData();
                const fileInput = document.getElementById('document');
                const domainSelect = document.getElementById('domain');
                
                formData.append('file', fileInput.files[0]);
                formData.append('domain', domainSelect.value);
                
                try {
                    const response = await fetch('/upload', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const result = await response.json();
                    const resultDiv = document.getElementById('uploadResult');
                    
                    if (result.success) {
                        resultDiv.innerHTML = `
                            <h3>✅ Upload Successful!</h3>
                            <p><strong>Document:</strong> ${result.metadata.file_path}</p>
                            <p><strong>Chunks Created:</strong> ${result.chunks_created}</p>
                            <p><strong>Domain:</strong> ${result.domain}</p>
                            <p><strong>File Type:</strong> ${result.metadata.file_type}</p>
                        `;
                    } else {
                        resultDiv.innerHTML = `<h3>❌ Upload Failed</h3><p>${result.error}</p>`;
                    }
                    
                    resultDiv.style.display = 'block';
                } catch (error) {
                    console.error('Upload error:', error);
                    document.getElementById('uploadResult').innerHTML = `<h3>❌ Error</h3><p>${error.message}</p>`;
                    document.getElementById('uploadResult').style.display = 'block';
                }
            });

            // Query processing
            document.getElementById('queryForm').addEventListener('submit', async (e) => {
                e.preventDefault();
                
                const query = document.getElementById('query').value;
                const domain = document.getElementById('queryDomain').value;
                
                document.getElementById('queryLoading').style.display = 'block';
                document.getElementById('queryResult').style.display = 'none';
                
                try {
                    const response = await fetch('/query', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ query, domain: domain || null })
                    });
                    
                    const result = await response.json();
                    const resultDiv = document.getElementById('queryResult');
                    
                    if (result.success) {
                        const confidencePercent = (result.confidence * 100).toFixed(1);
                        
                        resultDiv.innerHTML = `
                            <h3>💡 Answer</h3>
                            <p style="font-size: 1.1rem; line-height: 1.6; margin-bottom: 20px;">${result.answer}</p>
                            
                            <h4>📊 Confidence Score</h4>
                            <div class="confidence-bar">
                                <div class="confidence-fill" style="width: ${confidencePercent}%"></div>
                            </div>
                            <p>${confidencePercent}% confident</p>
                            
                            <h4>🔍 Query Analysis</h4>
                            <p><strong>Intent:</strong> ${result.query_analysis.intent_type}</p>
                            <p><strong>Domain:</strong> ${result.query_analysis.detected_domain}</p>
                            <p><strong>Intent Confidence:</strong> ${(result.query_analysis.intent_confidence * 100).toFixed(1)}%</p>
                            
                            ${result.conditions.length > 0 ? `
                                <h4>📋 Conditions</h4>
                                <ul>
                                    ${result.conditions.map(cond => `<li>${cond}</li>`).join('')}
                                </ul>
                            ` : ''}
                            
                            <h4>📝 Relevant Clauses</h4>
                            ${result.relevant_clauses.map(clause => `
                                <div class="clause">
                                    <p><strong>Relevance:</strong> ${(clause.relevance_score * 100).toFixed(1)}%</p>
                                    <p>${clause.text}</p>
                                </div>
                            `).join('')}
                            
                            <h4>💭 Rationale</h4>
                            <p>${result.rationale}</p>
                            
                            <h4>📚 Sources</h4>
                            <ul>
                                ${result.document_sources.map(source => `
                                    <li>${source.file_name} (${source.file_type})</li>
                                `).join('')}
                            </ul>
                        `;
                    } else {
                        resultDiv.innerHTML = `<h3>❌ Query Failed</h3><p>${result.error}</p>`;
                    }
                    
                    resultDiv.style.display = 'block';
                } catch (error) {
                    console.error('Query error:', error);
                    document.getElementById('queryResult').innerHTML = `<h3>❌ Error</h3><p>${error.message}</p>`;
                    document.getElementById('queryResult').style.display = 'block';
                } finally {
                    document.getElementById('queryLoading').style.display = 'none';
                }
            });

            // Load statistics
            async function loadStats() {
                try {
                    const response = await fetch('/stats');
                    const stats = await response.json();
                    
                    const container = document.getElementById('statsContainer');
                    container.innerHTML = `
                        <div class="stats-grid">
                            <div class="stat-card">
                                <div class="stat-value">${stats.total_documents}</div>
                                <div class="stat-label">Documents</div>
                            </div>
                            <div class="stat-card">
                                <div class="stat-value">${stats.total_chunks}</div>
                                <div class="stat-label">Text Chunks</div>
                            </div>
                            <div class="stat-card">
                                <div class="stat-value">${stats.index_size}</div>
                                <div class="stat-label">Index Size</div>
                            </div>
                        </div>
                        
                        <h4 style="margin-top: 20px;">📂 Document Types</h4>
                        ${Object.entries(stats.document_types).map(([type, count]) => `
                            <p><strong>${type.toUpperCase()}:</strong> ${count} documents</p>
                        `).join('')}
                        
                        <h4 style="margin-top: 20px;">🏢 Domain Breakdown</h4>
                        ${Object.entries(stats.domain_breakdown).map(([domain, count]) => `
                            <p><strong>${domain.charAt(0).toUpperCase() + domain.slice(1)}:</strong> ${count} documents</p>
                        `).join('')}
                        
                        <button class="btn" onclick="loadStats()" style="margin-top: 20px;">🔄 Refresh</button>
                    `;
                } catch (error) {
                    console.error('Stats error:', error);
                }
            }

            // Load stats on page load
            loadStats();
        </script>
    </body>
    </html>
    """

@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    domain: str = "insurance",
    system: IntelligentQuerySystem = Depends(get_query_system)
):
    """Upload and process a document"""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Check file type
    allowed_extensions = {'.pdf', '.docx', '.doc', '.txt', '.eml'}
    file_extension = Path(file.filename).suffix.lower()
    
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
        )
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        # Process the document
        result = system.add_document(temp_file_path, domain)
        
        # Clean up temporary file
        os.unlink(temp_file_path)
        
        if result["success"]:
            result["message"] = "Document processed successfully"
            return DocumentUploadResponse(**result)
        else:
            raise HTTPException(status_code=500, detail=result["error"])
            
    except Exception as e:
        # Clean up temporary file if it exists
        if 'temp_file_path' in locals():
            try:
                os.unlink(temp_file_path)
            except:
                pass
        
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def process_query(
    request: QueryRequest,
    system: IntelligentQuerySystem = Depends(get_query_system)
):
    """Process a natural language query"""
    try:
        result = system.process_query(request.query, request.domain)
        
        if result["success"]:
            return result
        else:
            raise HTTPException(status_code=500, detail=result["error"])
            
    except Exception as e:
        logger.error(f"Query error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_statistics(system: IntelligentQuerySystem = Depends(get_query_system)):
    """Get system statistics"""
    try:
        return system.get_document_statistics()
    except Exception as e:
        logger.error(f"Stats error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check(system: IntelligentQuerySystem = Depends(get_query_system)):
    """System health check"""
    try:
        return system.health_check()
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents")
async def list_documents(system: IntelligentQuerySystem = Depends(get_query_system)):
    """List all processed documents"""
    try:
        documents = []
        for doc_path, metadata in system.embedding_engine.document_metadata.items():
            documents.append({
                "document_id": doc_path,
                "file_name": Path(doc_path).name,
                "file_type": metadata.file_type,
                "processed_date": metadata.processed_date,
                "page_count": metadata.page_count,
                "file_size": metadata.file_size
            })
        
        return {"documents": documents, "total_count": len(documents)}
        
    except Exception as e:
        logger.error(f"List documents error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents/{document_id}")
async def get_document_content(
    document_id: str,
    system: IntelligentQuerySystem = Depends(get_query_system)
):
    """Get content for a specific document"""
    try:
        content = system.get_document_content(document_id)
        if content is None:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return content
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get document content error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search/similar")
async def search_similar_content(
    content: str,
    top_k: int = 5,
    system: IntelligentQuerySystem = Depends(get_query_system)
):
    """Search for similar content"""
    try:
        results = system.search_similar_content(content, top_k)
        return {"results": results, "query": content}
        
    except Exception as e:
        logger.error(f"Similar search error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/index")
async def clear_index(system: IntelligentQuerySystem = Depends(get_query_system)):
    """Clear the search index"""
    try:
        result = system.clear_index()
        return result
        
    except Exception as e:
        logger.error(f"Clear index error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/export")
async def export_knowledge_base(
    export_path: str = "./exports/knowledge_base",
    system: IntelligentQuerySystem = Depends(get_query_system)
):
    """Export the knowledge base"""
    try:
        result = system.export_knowledge_base(export_path)
        return result
        
    except Exception as e:
        logger.error(f"Export error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Create necessary directories
    Path("data").mkdir(exist_ok=True)
    Path("static").mkdir(exist_ok=True)
    
    # Run the server
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info"
    )