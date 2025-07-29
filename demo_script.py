#!/usr/bin/env python3
"""
Demo Script for LLM-Powered Intelligent Query-Retrieval System
Hackathon Demonstration

This script demonstrates the system's capabilities across different domains:
- Insurance policy analysis
- HR document queries
- Legal contract review
- Compliance document processing
"""

import requests
import json
import time
import os
from pathlib import Path
from typing import Dict, List

class HackathonDemo:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        
    def print_header(self, title: str):
        """Print a formatted header"""
        print("\n" + "="*60)
        print(f"  {title}")
        print("="*60)
    
    def print_subheader(self, title: str):
        """Print a formatted subheader"""
        print(f"\n--- {title} ---")
    
    def check_system_health(self) -> bool:
        """Check if the system is running"""
        try:
            response = self.session.get(f"{self.base_url}/health")
            if response.status_code == 200:
                health = response.json()
                print(f"✅ System Status: {health['status']}")
                print(f"📊 Index Size: {health['index_size']} documents")
                return True
            else:
                print(f"❌ System health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Cannot connect to system: {e}")
            return False
    
    def upload_document(self, file_path: str, domain: str) -> Dict:
        """Upload a document to the system"""
        try:
            with open(file_path, 'rb') as f:
                files = {'file': f}
                data = {'domain': domain}
                response = self.session.post(f"{self.base_url}/upload", files=files, data=data)
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Uploaded: {Path(file_path).name}")
                print(f"   📄 Domain: {domain}")
                print(f"   🔢 Chunks: {result['chunks_created']}")
                return result
            else:
                print(f"❌ Upload failed: {response.status_code} - {response.text}")
                return {}
        except Exception as e:
            print(f"❌ Upload error: {e}")
            return {}
    
    def query_system(self, query: str, domain: str = None) -> Dict:
        """Send a query to the system"""
        try:
            payload = {"query": query}
            if domain:
                payload["domain"] = domain
            
            response = self.session.post(
                f"{self.base_url}/query",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"❌ Query failed: {response.status_code} - {response.text}")
                return {}
        except Exception as e:
            print(f"❌ Query error: {e}")
            return {}
    
    def display_query_result(self, result: Dict):
        """Display query results in a formatted way"""
        if not result or not result.get('success'):
            print("❌ No results to display")
            return
        
        print(f"\n💡 ANSWER:")
        print(f"   {result['answer']}")
        
        print(f"\n📊 CONFIDENCE: {result['confidence']:.1%}")
        
        print(f"\n🔍 QUERY ANALYSIS:")
        analysis = result['query_analysis']
        print(f"   Intent: {analysis['intent_type']}")
        print(f"   Domain: {analysis['detected_domain']}")
        print(f"   Intent Confidence: {analysis['intent_confidence']:.1%}")
        
        if result.get('conditions'):
            print(f"\n📋 CONDITIONS:")
            for condition in result['conditions'][:3]:  # Show first 3
                print(f"   • {condition}")
        
        print(f"\n📝 RELEVANT CLAUSES:")
        for i, clause in enumerate(result['relevant_clauses'][:2], 1):  # Show first 2
            print(f"   {i}. Relevance: {clause['relevance_score']:.1%}")
            text = clause['text'][:150] + "..." if len(clause['text']) > 150 else clause['text']
            print(f"      {text}")
        
        print(f"\n💭 RATIONALE:")
        print(f"   {result['rationale']}")
        
        print(f"\n📚 SOURCES:")
        for source in result['document_sources']:
            print(f"   • {source['file_name']} ({source['file_type']})")
    
    def run_insurance_demo(self):
        """Demonstrate insurance domain capabilities"""
        self.print_header("INSURANCE DOMAIN DEMONSTRATION")
        
        # Upload insurance policy
        self.print_subheader("Uploading Insurance Policy")
        self.upload_document("data/sample_documents/insurance_policy.txt", "insurance")
        
        # Insurance queries
        insurance_queries = [
            "Does this policy cover knee surgery, and what are the conditions?",
            "What is the deductible for emergency room visits?",
            "What are the coverage limits for dental procedures?",
            "Does the policy cover mental health services?",
            "What is required for prescription drug coverage?"
        ]
        
        for i, query in enumerate(insurance_queries, 1):
            self.print_subheader(f"Insurance Query {i}")
            print(f"❓ QUESTION: {query}")
            
            result = self.query_system(query, "insurance")
            self.display_query_result(result)
            
            if i < len(insurance_queries):
                input("\n⏸️  Press Enter to continue to next query...")
    
    def run_hr_demo(self):
        """Demonstrate HR domain capabilities"""
        self.print_header("HR DOMAIN DEMONSTRATION")
        
        # Upload HR handbook
        self.print_subheader("Uploading Employee Handbook")
        self.upload_document("data/sample_documents/hr_employee_handbook.txt", "hr")
        
        # HR queries
        hr_queries = [
            "What is the remote work policy for employees?",
            "How many vacation days do employees get?",
            "What are the performance review requirements?",
            "What is the policy for sick leave?",
            "What benefits are available for employee development?"
        ]
        
        for i, query in enumerate(hr_queries, 1):
            self.print_subheader(f"HR Query {i}")
            print(f"❓ QUESTION: {query}")
            
            result = self.query_system(query, "hr")
            self.display_query_result(result)
            
            if i < len(hr_queries):
                input("\n⏸️  Press Enter to continue to next query...")
    
    def demonstrate_cross_domain_intelligence(self):
        """Demonstrate the system's ability to handle cross-domain queries"""
        self.print_header("CROSS-DOMAIN INTELLIGENCE DEMONSTRATION")
        
        cross_domain_queries = [
            "Compare the leave policies between insurance coverage and employee benefits",
            "What are the privacy and confidentiality requirements across all documents?",
            "How do emergency procedures differ between insurance and workplace policies?",
            "What documentation is required for both medical claims and HR requests?"
        ]
        
        for i, query in enumerate(cross_domain_queries, 1):
            self.print_subheader(f"Cross-Domain Query {i}")
            print(f"❓ QUESTION: {query}")
            
            result = self.query_system(query)  # No domain filter for cross-domain
            self.display_query_result(result)
            
            if i < len(cross_domain_queries):
                input("\n⏸️  Press Enter to continue to next query...")
    
    def show_system_statistics(self):
        """Display system statistics"""
        self.print_header("SYSTEM STATISTICS")
        
        try:
            response = self.session.get(f"{self.base_url}/stats")
            if response.status_code == 200:
                stats = response.json()
                
                print(f"📊 KNOWLEDGE BASE STATISTICS:")
                print(f"   Total Documents: {stats['total_documents']}")
                print(f"   Total Text Chunks: {stats['total_chunks']}")
                print(f"   Index Size: {stats['index_size']}")
                print(f"   Embedding Model: {stats['model_name']}")
                
                print(f"\n📂 DOCUMENT TYPES:")
                for doc_type, count in stats['document_types'].items():
                    print(f"   {doc_type.upper()}: {count} documents")
                
                print(f"\n🏢 DOMAIN BREAKDOWN:")
                for domain, count in stats['domain_breakdown'].items():
                    print(f"   {domain.title()}: {count} documents")
                
            else:
                print(f"❌ Failed to get statistics: {response.status_code}")
        except Exception as e:
            print(f"❌ Statistics error: {e}")
    
    def create_sample_legal_document(self):
        """Create a sample legal contract document"""
        legal_content = """
# SOFTWARE DEVELOPMENT AGREEMENT

## Agreement Date: January 15, 2024
## Parties: TechCorp Industries and ClientCorp LLC

---

## SECTION 1: SCOPE OF WORK

This Software Development Agreement ("Agreement") is entered into between TechCorp Industries ("Developer") and ClientCorp LLC ("Client") for the development of custom software solutions.

### 1.1 Deliverables
The Developer agrees to deliver:
- Custom web application with user authentication
- Database design and implementation
- API development and documentation
- User interface design and development
- Testing and quality assurance

### 1.2 Timeline
Project timeline: 6 months from contract signing
- Phase 1 (Requirements): 2 weeks
- Phase 2 (Design): 4 weeks  
- Phase 3 (Development): 16 weeks
- Phase 4 (Testing): 4 weeks

## SECTION 2: PAYMENT TERMS

### 2.1 Total Contract Value
Total project cost: $150,000 USD

### 2.2 Payment Schedule
- 25% upon contract signing ($37,500)
- 25% upon completion of Phase 2 ($37,500)
- 25% upon completion of Phase 3 ($37,500)
- 25% upon final delivery and acceptance ($37,500)

### 2.3 Late Payment
Late payments subject to 1.5% monthly interest charge

## SECTION 3: INTELLECTUAL PROPERTY

### 3.1 Ownership
Client shall own all custom-developed software upon final payment
Developer retains rights to:
- Pre-existing code and frameworks
- General methodologies and techniques
- Non-client-specific improvements

### 3.2 License Grant
Developer grants Client perpetual, non-exclusive license to use, modify, and distribute the software

## SECTION 4: LIABILITY AND WARRANTIES

### 4.1 Limitation of Liability
Developer's liability shall not exceed the total contract amount
No liability for indirect, consequential, or special damages

### 4.2 Warranties
Developer warrants:
- Software will perform substantially as specified
- No infringement of third-party intellectual property
- 90-day warranty period for defects

### 4.3 Indemnification
Each party indemnifies the other against claims arising from:
- Breach of this agreement
- Negligent acts or omissions
- Violation of applicable laws

## SECTION 5: CONFIDENTIALITY

### 5.1 Confidential Information
Both parties agree to maintain confidentiality of:
- Proprietary business information
- Technical specifications and designs
- Financial information
- Customer data and personal information

### 5.2 Data Protection
Compliance with applicable data protection regulations including GDPR
Appropriate technical and organizational measures for data security

## SECTION 6: TERMINATION

### 6.1 Termination for Cause
Either party may terminate for material breach with 30-day written notice

### 6.2 Termination for Convenience
Client may terminate for convenience with 60-day notice
Payment due for work completed through termination date

### 6.3 Effect of Termination
Upon termination:
- Return of confidential information
- Payment for work completed
- Transition assistance as reasonably requested

## SECTION 7: DISPUTE RESOLUTION

### 7.1 Mediation
Disputes shall first be addressed through mediation
Mediation to be conducted by qualified neutral mediator

### 7.2 Arbitration
If mediation fails, binding arbitration under rules of American Arbitration Association
Arbitration location: State of Delaware

### 7.3 Governing Law
Agreement governed by laws of State of Delaware
Jurisdiction in Delaware state and federal courts

## SECTION 8: GENERAL PROVISIONS

### 8.1 Force Majeure
Neither party liable for delays due to circumstances beyond reasonable control

### 8.2 Amendment
Agreement may only be amended in writing signed by both parties

### 8.3 Assignment
Rights and obligations may not be assigned without written consent

### 8.4 Entire Agreement
This Agreement constitutes the entire agreement between parties
Supersedes all prior negotiations and understandings

---

**SIGNATURES:**

TechCorp Industries: _________________________ Date: _________
By: John Smith, President

ClientCorp LLC: _________________________ Date: _________
By: Jane Doe, CEO
"""
        
        # Save to file
        os.makedirs("data/sample_documents", exist_ok=True)
        with open("data/sample_documents/software_contract.txt", "w") as f:
            f.write(legal_content)
        
        return "data/sample_documents/software_contract.txt"
    
    def run_legal_demo(self):
        """Demonstrate legal domain capabilities"""
        self.print_header("LEGAL DOMAIN DEMONSTRATION")
        
        # Create and upload legal contract
        self.print_subheader("Creating and Uploading Legal Contract")
        contract_path = self.create_sample_legal_document()
        self.upload_document(contract_path, "legal")
        
        # Legal queries
        legal_queries = [
            "What are the payment terms and schedule in this contract?",
            "What are the termination clauses and conditions?",
            "Who owns the intellectual property rights?",
            "What are the liability limitations?",
            "What is the dispute resolution process?"
        ]
        
        for i, query in enumerate(legal_queries, 1):
            self.print_subheader(f"Legal Query {i}")
            print(f"❓ QUESTION: {query}")
            
            result = self.query_system(query, "legal")
            self.display_query_result(result)
            
            if i < len(legal_queries):
                input("\n⏸️  Press Enter to continue to next query...")
    
    def run_full_demo(self):
        """Run the complete hackathon demonstration"""
        self.print_header("🚀 LLM-POWERED INTELLIGENT QUERY-RETRIEVAL SYSTEM")
        print("🎯 Hackathon Demonstration")
        print("🏢 Domains: Insurance, Legal, HR, and Compliance")
        print("🤖 Features: Free LLMs, FAISS Search, Explainable AI")
        
        # Check system health
        if not self.check_system_health():
            print("❌ System not available. Please start the application first.")
            return
        
        print("\n🎬 Starting comprehensive demonstration...")
        input("⏸️  Press Enter to begin...")
        
        # Run domain demonstrations
        self.run_insurance_demo()
        input("\n⏸️  Press Enter to continue to HR demo...")
        
        self.run_hr_demo()
        input("\n⏸️  Press Enter to continue to Legal demo...")
        
        self.run_legal_demo()
        input("\n⏸️  Press Enter to continue to Cross-Domain demo...")
        
        self.demonstrate_cross_domain_intelligence()
        
        # Show final statistics
        self.show_system_statistics()
        
        self.print_header("🎉 DEMONSTRATION COMPLETE")
        print("✅ Successfully demonstrated:")
        print("   • Multi-format document processing (PDF, DOCX, TXT)")
        print("   • Cross-domain intelligent query handling")
        print("   • Semantic search with FAISS embeddings")
        print("   • Free LLM-powered natural language understanding")
        print("   • Explainable AI with confidence scores")
        print("   • Structured JSON responses with rationale")
        print("   • Real-world scenarios in Insurance, Legal, HR domains")
        
        print(f"\n🌐 System running at: {self.base_url}")
        print("📚 API Documentation: /docs")
        print("🔍 Interactive Interface: /")

def main():
    """Main demo function"""
    print("🎯 LLM-Powered Intelligent Query-Retrieval System - Hackathon Demo")
    print("="*60)
    
    # Create demo instance
    demo = HackathonDemo()
    
    # Create sample documents directory
    os.makedirs("data/sample_documents", exist_ok=True)
    
    # Check if we should run full demo or specific parts
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == "insurance":
            demo.run_insurance_demo()
        elif sys.argv[1] == "hr":
            demo.run_hr_demo()
        elif sys.argv[1] == "legal":
            demo.run_legal_demo()
        elif sys.argv[1] == "cross-domain":
            demo.demonstrate_cross_domain_intelligence()
        elif sys.argv[1] == "stats":
            demo.show_system_statistics()
        else:
            print("Usage: python demo_script.py [insurance|hr|legal|cross-domain|stats]")
    else:
        # Run full demonstration
        demo.run_full_demo()

if __name__ == "__main__":
    main()