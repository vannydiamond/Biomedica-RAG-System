#!/usr/bin/env python
"""
QUICK-START: Test your RAG setup and run Phase 1 diagnostics

Usage:
    python quick_start.py                    # Run with sample file
    python quick_start.py your_document.pdf  # Run with your file

This script:
1. Checks dependencies
2. Finds test documents
3. Runs Phase 1 diagnostics
4. Shows next steps
"""

import os
import sys
from pathlib import Path

def check_dependencies():
    """Verify all required packages are installed."""
    print("\n" + "=" * 60)
    print("  CHECKING DEPENDENCIES")
    print("=" * 60 + "\n")
    
    required = [
        "langchain",
        "langchain_core",
        "langchain_community",
        "langchain_huggingface",
        "sentence_transformers",
        "faiss",
        "pypdf",
        "docx",
        "pydantic",
    ]
    
    all_ok = True
    for package in required:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - MISSING")
            all_ok = False
    
    if not all_ok:
        print("\n⚠️  Some packages are missing.")
        print("Install with: pip install -r requirements.txt")
        return False
    
    print("\n✓ All dependencies installed!")
    return True


def find_test_documents():
    """Find available test documents."""
    print("\n" + "=" * 60)
    print("  AVAILABLE TEST DOCUMENTS")
    print("=" * 60 + "\n")
    
    extensions = [".pdf", ".docx", ".txt"]
    documents = []
    
    for ext in extensions:
        for doc in Path(".").glob(f"*{ext}"):
            if doc.is_file():
                size_mb = doc.stat().st_size / (1024 * 1024)
                documents.append((str(doc), size_mb))
    
    if documents:
        print("Documents found:")
        for doc, size in sorted(documents):
            print(f"  • {doc} ({size:.1f} MB)")
        return documents
    else:
        print("⚠️  No documents found in current directory")
        print("Place a .pdf, .docx, or .txt file here and try again")
        return None


def run_phase1(document_path):
    """Run Phase 1 diagnostics."""
    if not os.path.exists(document_path):
        print(f"\n✗ File not found: {document_path}")
        return False
    
    print(f"\n✓ Running Phase 1 diagnostics on: {document_path}\n")
    
    # Import and run diagnostics
    try:
        # Run the diagnostics script
        import subprocess
        result = subprocess.run(
            [sys.executable, "phase1_diagnostics.py", document_path],
            capture_output=False
        )
        return result.returncode == 0
    except Exception as e:
        print(f"\n✗ Error running diagnostics: {e}")
        return False


def show_next_steps():
    """Show what to do next."""
    print("\n" + "=" * 60)
    print("  NEXT STEPS")
    print("=" * 60)
    
    steps = """
1. ✓ Phase 1 diagnostics complete

2. Review results above for issues:
   • Empty/corrupted text?
   • Duplicate chunks?
   • Zero retrieval scores?

3. If issues found, fix them first:
   • Update extraction logic (app/utils.py)
   • Adjust chunk size (phase4_retrieval_improvements.py)
   • Check embeddings model (app/services.py)

4. No issues? Proceed to Phase 2:
   • Update your API endpoints with query modes
   • Add retrieval and grounding validation
   • See: phase2_query_modes.py

5. Test the complete pipeline:
   • Upload document to your app
   • Ask questions in Document mode
   • Verify answers are grounded

6. Add transparency (Phase 5):
   • Show retrieved evidence
   • Display confidence scores
   • Visualize retrieval pipeline

For detailed guidance, see:
- phase0_7_implementation_guide.py (complete walkthrough)
- Individual phase files (specific implementations)

Questions? Check the phase files for examples.
"""
    print(steps)


def main():
    """Main entry point."""
    print("\n" + "=" * 60)
    print("  BIOMEDICAL RAG - QUICK START")
    print("=" * 60)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Find test documents
    documents = find_test_documents()
    
    # Determine which document to use
    if len(sys.argv) > 1:
        # User specified document
        document = sys.argv[1]
    elif documents:
        # Use first found document
        document = documents[0][0]
        print(f"\nUsing: {document}")
    else:
        print("\n⚠️  No documents to test. Please provide a .pdf, .docx, or .txt file")
        print("\nUsage: python quick_start.py your_document.pdf")
        sys.exit(1)
    
    # Run Phase 1
    success = run_phase1(document)
    
    if success:
        show_next_steps()
        print("\n✓ Quick-start complete!")
        print("  Next: Review Phase 1 results and proceed to Phase 2")
    else:
        print("\n✗ Phase 1 diagnostics failed")
        print("  Check the output above for errors")
        sys.exit(1)


if __name__ == "__main__":
    main()
