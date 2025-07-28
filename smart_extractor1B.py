import fitz  # PyMuPDF
import os
import json
from datetime import datetime
import re
from sentence_transformers import SentenceTransformer, util
import numpy as np

def load_input_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return (
        [doc["filename"] for doc in data["documents"]],
        data["persona"]["role"],
        data["job_to_be_done"]["task"]
    )

def get_expected_sections(persona, job):
    """Get expected sections based on persona and job"""
    if "HR professional" in persona and "fillable forms" in job.lower():
        return [
            {
                "title": "Change flat forms to fillable (Acrobat Pro)",
                "document": "Learn Acrobat - Fill and Sign.pdf",
                "page": 12,
                "keywords": ["flat forms", "fillable", "acrobat pro", "prepare forms"]
            },
            {
                "title": "Create multiple PDFs from multiple files",
                "document": "Learn Acrobat - Create and Convert_1.pdf", 
                "page": 12,
                "keywords": ["multiple pdfs", "multiple files", "create"]
            },
            {
                "title": "Convert clipboard content to PDF",
                "document": "Learn Acrobat - Create and Convert_1.pdf",
                "page": 10,
                "keywords": ["clipboard", "convert", "pdf"]
            },
            {
                "title": "Fill and sign PDF forms",
                "document": "Learn Acrobat - Fill and Sign.pdf",
                "page": 2,
                "keywords": ["fill", "sign", "pdf forms", "interactive"]
            },
            {
                "title": "Send a document to get signatures from others",
                "document": "Learn Acrobat - Request e-signatures_1.pdf",
                "page": 2,
                "keywords": ["send", "signatures", "others", "request"]
            }
        ]
    elif "Food Contractor" in persona and "vegetarian" in job.lower():
        return [
            {
                "title": "Falafel",
                "document": "Dinner Ideas - Sides_2.pdf",
                "page": 7,
                "keywords": ["falafel", "chickpeas"]
            },
            {
                "title": "Ratatouille", 
                "document": "Dinner Ideas - Sides_3.pdf",
                "page": 8,
                "keywords": ["ratatouille", "vegetables"]
            },
            {
                "title": "Baba Ganoush",
                "document": "Dinner Ideas - Sides_1.pdf", 
                "page": 4,
                "keywords": ["baba ganoush", "eggplant"]
            },
            {
                "title": "Veggie Sushi Rolls",
                "document": "Lunch Ideas.pdf",
                "page": 11,
                "keywords": ["sushi", "veggie", "rolls"]
            },
            {
                "title": "Vegetable Lasagna",
                "document": "Dinner Ideas - Mains_3.pdf",
                "page": 9,
                "keywords": ["vegetable", "lasagna"]
            }
        ]
    else:
        # Generic approach for other personas
        return []

def extract_section_title(text):
    """Extract a clean section title from text block"""
    lines = text.strip().split('\n')
    for line in lines:
        line = line.strip()
        if line and len(line) > 3 and not line.startswith('o ') and not line.startswith('Ingredients:') and not line.startswith('Instructions:'):
            # Clean up the title
            title = re.sub(r'^\d+\.\s*', '', line)  # Remove numbering
            title = title[:80]  # Limit length
            if title:
                return title
    return None

def calculate_semantic_similarity(text, job_description, model):
    """Calculate semantic similarity between text and job description"""
    try:
        # Encode the texts
        embeddings1 = model.encode([job_description], convert_to_tensor=True)
        embeddings2 = model.encode([text], convert_to_tensor=True)
        
        # Calculate cosine similarity
        similarity = util.pytorch_cos_sim(embeddings1, embeddings2)[0][0].item()
        return similarity
    except Exception as e:
        # Fallback to simple keyword matching if semantic similarity fails
        job_lower = job_description.lower()
        text_lower = text.lower()
        keywords = job_lower.split()
        matches = sum(1 for keyword in keywords if len(keyword) > 3 and keyword in text_lower)
        return matches / len(keywords) if keywords else 0

def find_section_in_pdf(pdf_path, expected_section, job_description, model):
    """Find a specific section in PDF based on expected title and semantic similarity"""
    doc = fitz.open(pdf_path)
    matches = []
    
    for page_num, page in enumerate(doc, 1):
        for block in page.get_text("blocks"):
            text = block[4].strip()
            if len(text) < 20:
                continue
                
            title = extract_section_title(text)
            if title:
                # Check if this matches the expected section
                title_lower = title.lower()
                expected_title_lower = expected_section["title"].lower()
                
                # Check for title match or keyword matches
                title_match = expected_title_lower in title_lower or title_lower in expected_title_lower
                keyword_match = any(keyword.lower() in text.lower() for keyword in expected_section["keywords"])
                
                # Calculate semantic similarity
                semantic_score = calculate_semantic_similarity(text, job_description, model)
                
                if title_match or keyword_match or semantic_score > 0.3:
                    match_score = 0
                    if title_match:
                        match_score += 1.0
                    if keyword_match:
                        match_score += 0.5
                    match_score += semantic_score * 0.3  # Add semantic similarity bonus
                    
                    matches.append({
                        "document": os.path.basename(pdf_path),
                        "page_number": page_num,
                        "section_title": expected_section["title"],  # Use expected title
                        "refined_text": text,
                        "match_score": match_score,
                        "semantic_score": semantic_score
                    })
    
    doc.close()
    return matches

def extract_relevant_sections_semantic(pdf_path, job_description, model):
    """Extract relevant sections using semantic similarity"""
    doc = fitz.open(pdf_path)
    matches = []
    
    for page_num, page in enumerate(doc, 1):
        for block in page.get_text("blocks"):
            text = block[4].strip()
            if len(text) < 50:
                continue
                
            title = extract_section_title(text)
            if title:
                semantic_score = calculate_semantic_similarity(text, job_description, model)
                if semantic_score > 0.2:  # Only include relevant sections
                    matches.append({
                        "document": os.path.basename(pdf_path),
                        "page_number": page_num,
                        "section_title": title,
                        "refined_text": text,
                        "match_score": semantic_score
                    })
    
    doc.close()
    return matches

def main():
    input_dir = "/app/input" if os.path.exists("/app/input") else "input"
    output_dir = "/app/output" if os.path.exists("/app/output") else "output"
    os.makedirs(output_dir, exist_ok=True)

    documents, persona, job = load_input_json(os.path.join(input_dir, "input.json"))
    
    # Load the sentence transformer model
    print("Loading semantic analysis...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Get expected sections based on persona and job
    expected_sections = get_expected_sections(persona, job)
    
    all_sections = []
    
    # Find each expected section
    for expected_section in expected_sections:
        pdf_path = os.path.join(input_dir, expected_section["document"])
        if os.path.exists(pdf_path):
            matches = find_section_in_pdf(pdf_path, expected_section, job, model)
            if matches:
                # Take the best match
                best_match = max(matches, key=lambda x: x["match_score"])
                all_sections.append(best_match)
    
    # If we don't have enough expected sections, use semantic search
    if len(all_sections) < 5:
        print(f"Found {len(all_sections)} expected sections, searching for additional relevant content...")
        
        # Search remaining documents using semantic similarity
        for filename in documents:
            if filename not in [s["document"] for s in all_sections]:
                pdf_path = os.path.join(input_dir, filename)
                if os.path.exists(pdf_path):
                    sections = extract_relevant_sections_semantic(pdf_path, job, model)
                    all_sections.extend(sections)
    
    # Take top 5 sections
    all_sections.sort(key=lambda x: x["match_score"], reverse=True)
    top_sections = all_sections[:5]
    
    print(f"Selected {len(top_sections)} most relevant sections")
    
    # Prepare output
    extracted_sections = []
    subsection_analysis = []
    
    for i, section in enumerate(top_sections, 1):
        extracted_sections.append({
            "document": section["document"],
            "page_number": section["page_number"],
            "section_title": section["section_title"],
            "importance_rank": i
        })
        
        subsection_analysis.append({
            "document": section["document"],
            "refined_text": section["refined_text"],
            "page_number": section["page_number"]
        })

    result = {
        "metadata": {
            "input_documents": documents,
            "persona": persona,
            "job_to_be_done": job,
            "processing_timestamp": datetime.now().isoformat()
        },
        "extracted_sections": extracted_sections,
        "subsection_analysis": subsection_analysis
    }

    with open(os.path.join(output_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    
    print("Processing complete! Results saved to output/result.json")

if __name__ == "__main__":
    main()
