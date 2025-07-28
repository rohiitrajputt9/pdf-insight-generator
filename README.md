#  Persona-Aware PDF Intelligence – Adobe Hackathon 2025 (Problem 1B)

This project is a solution to **Problem Statement 1B** from the Adobe Hackathon 2025. It extracts the **most relevant sections** from a collection of PDF files based on a specific **persona** and **job-to-be-done**. The solution combines **semantic similarity (Sentence Transformers)** with **PDF parsing (PyMuPDF)** to produce a clean, ranked output.

---

## Problem Summary (1B)

> Given a list of PDF documents, a user persona, and a task to complete, extract and rank the top 5 most relevant sections across all PDFs using semantic understanding and structured logic.

---

##  Features

- Accepts structured `input.json` defining:
  - Document names
  - Persona role
  - Job to be done
- Uses **Sentence Transformers** for semantic similarity
- Combines rule-based logic with ML to boost accuracy
- Outputs a structured JSON file with:
  - Top 5 relevant sections
  - Their page number and importance rank
  - Refined text for each section

---

## 📁 Folder Structure
persona-aware-pdf-intelligence/
├── input/
│ ├── input.json # Input configuration
│ └── *.pdf # All input PDF documents
├── output/
│ └── result.json # Output result
├── smart_extractor1B.py # Main solution script
├── Dockerfile # Docker container file
└── requirements.txt # Python dependencies
