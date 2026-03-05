import io
import re
import json
import nltk
import pandas as pd
import numpy as np
from typing import Dict, List, Any
from PyPDF2 import PdfReader
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from groq import Groq

# Step 10: Ensure NLTK resources are available
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

class ResumeRankingEngine:
    def __init__(self, groq_api_key: str):
        self.client = Groq(api_key=groq_api_key)
        self.model_id = "llama-3.1-8b-instant"
        # Step 6: Mini-LLM for Embeddings
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Step 13: Dynamic Weight Profiles
        self.dynamic_weights = {
            "internship": {"experience": 0.10, "projects": 0.35, "skills": 0.35, "education": 0.15, "certifications": 0.05},
            "entry": {"experience": 0.25, "projects": 0.25, "skills": 0.30, "education": 0.15, "certifications": 0.05},
            "experienced": {"experience": 0.50, "projects": 0.10, "skills": 0.25, "education": 0.10, "certifications": 0.05},
        }
        
        # Load Dataset for Accurate Role Prediction
        try:
            self.dataset = pd.read_csv("job_database.csv")
            self.role_list = self.dataset.to_dict(orient='records')
        except:
            self.role_list = []

    def clean_text(self, text: str) -> str: # Step 3
        text = re.sub(r"\n\s*\d+\s*\n", "\n", text)
        text = re.sub(r"\n{2,}", "\n", text)
        text = re.sub(r"[•●▪]", "-", text)
        return text.strip()

    def extract_sections(self, resume_text: str) -> Dict[str, str]: # Step 4
        # MANDATORY MAPPING: Internship -> Experience
        prompt = f"""
        Extract these sections from the resume into a JSON format: experience, projects, skills, education, certifications.
        
        RULES:
        1. If the candidate has an 'INTERNSHIP' or 'WORK HISTORY', you MUST put that text into the 'experience' key.
        2. Do NOT summarize. Copy text exactly.
        3. Return ONLY a valid JSON object.
        
        Resume: {resume_text[:5000]}
        """
        try:
            res = self.client.chat.completions.create(
                model=self.model_id,
                response_format={"type": "json_object"},
                messages=[{"role": "system", "content": "You are a data extraction tool that outputs JSON. You treat Internship as Experience."},
                          {"role": "user", "content": prompt}],
                temperature=0
            )
            data = json.loads(res.choices[0].message.content)
            return {k.lower(): v for k, v in data.items()}
        except: 
            return {"skills": resume_text[:1000]}

    def score_single_resume(self, pdf_stream: io.BytesIO, jd_text: str, lvl: str) -> Dict[str, Any]:
        filename = getattr(pdf_stream, "filename", "unknown.pdf")
        reader = PdfReader(pdf_stream)
        raw_text = "".join(page.extract_text() or "" for page in reader.pages)
        resume_text = self.clean_text(raw_text) # Step 3
        sections = self.extract_sections(resume_text) # Step 4
        
        jd_emb = self.embedder.encode([jd_text]) # Step 6
        normalized_section_scores = {}

        for sec, content in sections.items():
            content_str = str(content).strip()
            if len(content_str) < 15: continue
            
            # Step 7: Semantic
            emb = self.embedder.encode([content_str])
            sem = cosine_similarity(emb, jd_emb)[0][0]
            
            # Step 8: Lexical (Step-by-step TF-IDF)
            try:
                tfidf = TfidfVectorizer(stop_words='english').fit_transform([content_str, jd_text])
                lex = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
            except: lex = 0.0

            # Step 9: Hybrid (0.6 / 0.4)
            hybrid = (0.6 * sem) + (0.4 * lex)
            
            # Step 10 & 11: Penalty
            penalty = 1 + (len(word_tokenize(content_str)) / 1000)
            
            # Step 12: Normalized Section Score
            normalized_section_scores[sec.lower()] = hybrid / penalty

        # Step 13: Profile Selection
        base_w = self.dynamic_weights.get(lvl, self.dynamic_weights['entry'])
        
        # Step 14 & 15: Re-normalization (Crucial for 100% accuracy)
        avail = [s for s in normalized_section_scores if s in base_w]
        if not avail: return {"score": 0.0, "category": "Poor Fit", "filename": filename}
        
        total_avail_weight = sum(base_w[s] for s in avail)
        
        # Step 16 & 17: weighted contribution & Raw Final Score
        raw_final = 0
        for s in avail:
            final_weight_s = base_w[s] / total_avail_weight # Re-normalized
            raw_final += normalized_section_scores[s] * final_weight_s
            
        # Step 18: Normalization Boost (1.55)
        final_score = min(raw_final * 1.55, 1.0)
        
        # Step 19: Classification
        category = "Excellent Fit" if final_score >= 0.70 else \
                   "Moderate Fit" if final_score >= 0.50 else \
                   "Weak Fit" if final_score >= 0.35 else "Poor Fit"
                   
        return {"score": round(float(final_score), 4), "category": category, "filename": filename}

    def generate_deep_report(self, pdf_stream, jd_text):
        """EXTENSION: Accurate Role Prediction using JSON Dataset"""
        reader = PdfReader(pdf_stream)
        raw_text = "".join(page.extract_text() or "" for page in reader.pages)
        resume_text = self.clean_text(raw_text)
        dataset_context = json.dumps(self.role_list[:15])

        prompt = f"""
        Act as a Senior HR Tech Lead. Predict the most accurate role from this JSON database: {dataset_context}.
        Resume: {resume_text[:4000]}
        
        INSTRUCTIONS:
        1. Compare candidate's specific technologies (e.g., Docker, Django, React) to the database.
        2. Pick the title ONLY from the database. 
        3. Do NOT include any scores.
        
        Return ONLY JSON:
        {{
          "best_match": "ACCURATE_TITLE",
          "matched_skills": ["s1", "s2"],
          "reasoning": "Technical justification.",
          "skill_gaps": ["g1", "g2"],
          "recommended": "Certifications for this role",
          "alternatives": ["Alt 1", "Alt 2"]
        }}
        """
        response = self.client.chat.completions.create(
            model=self.model_id,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        return json.loads(response.choices[0].message.content)

    def rank_resumes(self, streams, jd, lvl): # Step 20
        results = [self.score_single_resume(s, jd, lvl) for s in streams]
        return sorted(results, key=lambda x: x['score'], reverse=True)