import os
import json
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import email
from email.parser import BytesParser
from email.policy import default

import PyPDF2
import pdfplumber
from docx import Document
import magic
import re
from dataclasses import dataclass
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DocumentMetadata:
    """Metadata for processed documents"""
    file_path: str
    file_type: str
    file_size: int
    processed_date: str
    page_count: Optional[int] = None
    author: Optional[str] = None
    title: Optional[str] = None
    subject: Optional[str] = None
    creation_date: Optional[str] = None

@dataclass
class DocumentChunk:
    """Represents a chunk of text from a document"""
    text: str
    chunk_id: str
    document_id: str
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    metadata: Optional[Dict] = None

class DocumentProcessor:
    """Main document processor class for handling multiple file formats"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.supported_formats = ['.pdf', '.docx', '.doc', '.eml', '.msg', '.txt']
        
    def process_document(self, file_path: str) -> Tuple[List[DocumentChunk], DocumentMetadata]:
        """Process a document and return chunks with metadata"""
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Detect file type
            file_type = self._detect_file_type(file_path)
            logger.info(f"Processing {file_type} document: {file_path}")
            
            # Extract text based on file type
            if file_type == 'pdf':
                text, metadata = self._process_pdf(file_path)
            elif file_type in ['docx', 'doc']:
                text, metadata = self._process_docx(file_path)
            elif file_type in ['eml', 'msg', 'email']:
                text, metadata = self._process_email(file_path)
            elif file_type == 'txt':
                text, metadata = self._process_text(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            
            # Create chunks
            chunks = self._create_chunks(text, str(file_path))
            
            return chunks, metadata
            
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {str(e)}")
            raise
    
    def _detect_file_type(self, file_path: Path) -> str:
        """Detect file type using magic numbers and extension"""
        try:
            mime_type = magic.from_file(str(file_path), mime=True)
            extension = file_path.suffix.lower()
            
            if 'pdf' in mime_type or extension == '.pdf':
                return 'pdf'
            elif 'word' in mime_type or extension in ['.docx', '.doc']:
                return 'docx'
            elif 'text' in mime_type or extension == '.txt':
                return 'txt'
            elif extension in ['.eml', '.msg'] or 'message' in mime_type:
                return 'email'
            else:
                return extension[1:] if extension else 'unknown'
        except:
            # Fallback to extension
            return file_path.suffix.lower()[1:] if file_path.suffix else 'unknown'
    
    def _process_pdf(self, file_path: Path) -> Tuple[str, DocumentMetadata]:
        """Process PDF files"""
        text = ""
        metadata = None
        
        try:
            # Try with pdfplumber first (better for complex layouts)
            with pdfplumber.open(file_path) as pdf:
                pages = []
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        pages.append(page_text)
                text = "\n\n".join(pages)
                
                # Extract metadata
                metadata = DocumentMetadata(
                    file_path=str(file_path),
                    file_type='pdf',
                    file_size=file_path.stat().st_size,
                    processed_date=datetime.now().isoformat(),
                    page_count=len(pdf.pages),
                    title=pdf.metadata.get('Title', ''),
                    author=pdf.metadata.get('Author', ''),
                    subject=pdf.metadata.get('Subject', ''),
                    creation_date=str(pdf.metadata.get('CreationDate', ''))
                )
        except Exception as e:
            logger.warning(f"pdfplumber failed, trying PyPDF2: {e}")
            # Fallback to PyPDF2
            try:
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    pages = []
                    for page in pdf_reader.pages:
                        pages.append(page.extract_text())
                    text = "\n\n".join(pages)
                    
                    metadata = DocumentMetadata(
                        file_path=str(file_path),
                        file_type='pdf',
                        file_size=file_path.stat().st_size,
                        processed_date=datetime.now().isoformat(),
                        page_count=len(pdf_reader.pages),
                        title=pdf_reader.metadata.get('/Title', ''),
                        author=pdf_reader.metadata.get('/Author', ''),
                        subject=pdf_reader.metadata.get('/Subject', '')
                    )
            except Exception as e2:
                logger.error(f"Both PDF processors failed: {e2}")
                raise
        
        return text, metadata
    
    def _process_docx(self, file_path: Path) -> Tuple[str, DocumentMetadata]:
        """Process DOCX files"""
        try:
            doc = Document(file_path)
            
            # Extract text from paragraphs
            paragraphs = []
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    paragraphs.append(paragraph.text)
            
            # Extract text from tables
            tables_text = []
            for table in doc.tables:
                for row in table.rows:
                    row_text = " | ".join([cell.text.strip() for cell in row.cells])
                    if row_text.strip():
                        tables_text.append(row_text)
            
            text = "\n\n".join(paragraphs)
            if tables_text:
                text += "\n\nTables:\n" + "\n".join(tables_text)
            
            # Extract metadata
            props = doc.core_properties
            metadata = DocumentMetadata(
                file_path=str(file_path),
                file_type='docx',
                file_size=file_path.stat().st_size,
                processed_date=datetime.now().isoformat(),
                title=props.title or '',
                author=props.author or '',
                subject=props.subject or '',
                creation_date=str(props.created) if props.created else ''
            )
            
            return text, metadata
            
        except Exception as e:
            logger.error(f"Error processing DOCX: {e}")
            raise
    
    def _process_email(self, file_path: Path) -> Tuple[str, DocumentMetadata]:
        """Process email files (.eml, .msg)"""
        try:
            with open(file_path, 'rb') as f:
                msg = BytesParser(policy=default).parse(f)
            
            # Extract email content
            subject = msg.get('Subject', '')
            sender = msg.get('From', '')
            recipient = msg.get('To', '')
            date = msg.get('Date', '')
            
            # Get email body
            body = ""
            if msg.is_multipart():
                for part in msg.walk():
                    content_type = part.get_content_type()
                    if content_type == "text/plain":
                        body += part.get_content()
                    elif content_type == "text/html":
                        # Simple HTML tag removal
                        html_content = part.get_content()
                        body += re.sub('<[^<]+?>', '', html_content)
            else:
                body = msg.get_content()
            
            # Combine email parts
            text = f"Subject: {subject}\nFrom: {sender}\nTo: {recipient}\nDate: {date}\n\n{body}"
            
            metadata = DocumentMetadata(
                file_path=str(file_path),
                file_type='email',
                file_size=file_path.stat().st_size,
                processed_date=datetime.now().isoformat(),
                title=subject,
                author=sender,
                subject=subject,
                creation_date=date
            )
            
            return text, metadata
            
        except Exception as e:
            logger.error(f"Error processing email: {e}")
            raise
    
    def _process_text(self, file_path: Path) -> Tuple[str, DocumentMetadata]:
        """Process plain text files"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
            
            metadata = DocumentMetadata(
                file_path=str(file_path),
                file_type='txt',
                file_size=file_path.stat().st_size,
                processed_date=datetime.now().isoformat()
            )
            
            return text, metadata
            
        except Exception as e:
            logger.error(f"Error processing text file: {e}")
            raise
    
    def _create_chunks(self, text: str, document_id: str) -> List[DocumentChunk]:
        """Split text into overlapping chunks"""
        chunks = []
        words = text.split()
        
        if len(words) <= self.chunk_size:
            # Document is small enough to be one chunk
            chunks.append(DocumentChunk(
                text=text,
                chunk_id=f"{document_id}_chunk_0",
                document_id=document_id
            ))
            return chunks
        
        # Create overlapping chunks
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = " ".join(chunk_words)
            
            chunks.append(DocumentChunk(
                text=chunk_text,
                chunk_id=f"{document_id}_chunk_{len(chunks)}",
                document_id=document_id
            ))
            
            # Stop if we've reached the end
            if i + self.chunk_size >= len(words):
                break
        
        return chunks
    
    def extract_clauses(self, text: str, domain: str = "insurance") -> List[Dict]:
        """Extract specific clauses based on domain"""
        clauses = []
        
        # Domain-specific patterns
        patterns = {
            "insurance": [
                r"coverage\s+(?:includes?|covers?|extends?\s+to)[^.]*",
                r"deductible[^.]*",
                r"exclusion[s]?[^.]*",
                r"benefit[s]?[^.]*",
                r"premium[s]?[^.]*",
                r"claim[s]?[^.]*procedure[s]?[^.]*"
            ],
            "legal": [
                r"term[s]?\s+(?:and\s+)?condition[s]?[^.]*",
                r"liability[^.]*",
                r"indemnif[y|ication][^.]*",
                r"dispute[s]?\s+resolution[^.]*",
                r"termination[^.]*",
                r"breach[^.]*"
            ],
            "hr": [
                r"employee[s]?\s+(?:benefits?|rights?|responsibilities?)[^.]*",
                r"leave\s+policy[^.]*",
                r"performance[^.]*",
                r"disciplinary[^.]*",
                r"vacation[^.]*",
                r"sick\s+leave[^.]*"
            ],
            "compliance": [
                r"regulation[s]?[^.]*",
                r"compliance[^.]*",
                r"audit[^.]*",
                r"data\s+protection[^.]*",
                r"privacy[^.]*",
                r"gdpr[^.]*"
            ]
        }
        
        domain_patterns = patterns.get(domain.lower(), patterns["insurance"])
        
        for i, pattern in enumerate(domain_patterns):
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                clauses.append({
                    "clause_id": f"{domain}_{i}_{len(clauses)}",
                    "text": match.group(),
                    "pattern_type": pattern,
                    "start_pos": match.start(),
                    "end_pos": match.end()
                })
        
        return clauses