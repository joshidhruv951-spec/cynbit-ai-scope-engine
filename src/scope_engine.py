import json
import re
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "cynbit_service_catalog.json"

FEATURE_MODULES = {
    "Authentication & RBAC": {"patterns": ["auth", "login", "sso", "oauth", "role", "rbac", "user management"], "weeks": 1.0, "cost": 900},
    "Payment Gateway Integration": {"patterns": ["payment", "upi", "stripe", "razorpay", "billing", "subscription", "checkout"], "weeks": 1.5, "cost": 1400},
    "AI Chatbot / RAG Knowledge Engine": {"patterns": ["ai", "chatbot", "llm", "rag", "gpt", "gemini", "assistant", "automation", "nlp"], "weeks": 2.0, "cost": 2200},
    "Admin Analytics & Reporting": {"patterns": ["admin", "dashboard", "analytics", "report", "metrics", "kpi", "export"], "weeks": 1.5, "cost": 1200},
    "Third-Party API & CRM Sync": {"patterns": ["api", "crm", "hubspot", "salesforce", "webhook", "integration", "slack", "whatsapp"], "weeks": 1.0, "cost": 1000},
    "Real-Time Notifications": {"patterns": ["notification", "email", "sms", "push", "alert", "websocket"], "weeks": 0.5, "cost": 600},
}


class CynbitAIScopeEngine:
    """Hybrid TF-IDF Semantic Retrieval + Rule-Grounded Structured SOW Generator."""

    def __init__(self):
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            self.catalog = json.load(f)

        corpus = [f"{item['domain']} {item['keywords']}" for item in self.catalog]
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        self.catalog_vectors = self.vectorizer.fit_transform(corpus)

    def analyze_brief(self, brief_text: str, urgency: str = "Standard") -> dict:
        cleaned = re.sub(r"\s+", " ", brief_text.strip().lower())
        query_vec = self.vectorizer.transform([cleaned])
        similarities = cosine_similarity(query_vec, self.catalog_vectors).flatten()

        ranked_indices = np.argsort(similarities)[::-1]
        primary_idx = int(ranked_indices[0])
        primary_service = self.catalog[primary_idx]
        confidence = round(float(min(0.98, max(0.42, similarities[primary_idx] * 2.4 + 0.35))), 2)

        detected_modules = []
        extra_weeks = 0.0
        extra_cost = 0
        for mod_name, meta in FEATURE_MODULES.items():
            if any(pat in cleaned for pat in meta["patterns"]):
                detected_modules.append(mod_name)
                extra_weeks += meta["weeks"]
                extra_cost += meta["cost"]

        if not detected_modules:
            detected_modules = ["Core Architecture & REST API Setup", "Admin Analytics & Reporting"]
            extra_weeks = 1.5
            extra_cost = 1200

        urgency_multiplier = 1.25 if urgency == "Accelerated (Fast-Track)" else 1.0
        time_divisor = 0.8 if urgency == "Accelerated (Fast-Track)" else 1.0

        est_weeks_min = max(3, int(round((primary_service["base_weeks"] + extra_weeks) * time_divisor)))
        est_weeks_max = est_weeks_min + 2
        est_cost_min = int(round((primary_service["base_cost_usd"] + extra_cost) * urgency_multiplier, -2))
        est_cost_max = int(round(est_cost_min * 1.22, -2))

        milestones = [
            f"Phase 1 (Weeks 1-{max(1, est_weeks_min // 3)}): Architecture Blueprint, UI/UX Wireframes & Data Schema",
            f"Phase 2 (Weeks {max(2, est_weeks_min // 3 + 1)}-{max(2, (2 * est_weeks_min) // 3)}): Core Engineering ({', '.join(detected_modules[:2])})",
            f"Phase 3 (Weeks {max(3, (2 * est_weeks_min) // 3 + 1)}-{est_weeks_max}): QA Regression, Security Hardening & Production Deployment",
        ]

        return {
            "matched_practice": primary_service["domain"],
            "service_id": primary_service["id"],
            "retrieval_confidence": confidence,
            "recommended_stack": primary_service["recommended_stack"],
            "detected_modules": detected_modules,
            "estimated_timeline_weeks": f"{est_weeks_min}-{est_weeks_max} Weeks",
            "estimated_budget_usd": f"${est_cost_min:,} -${est_cost_max:,}",
            "technical_risks": primary_service["key_risks"],
            "delivery_roadmap": milestones,
        }


if __name__ == "__main__":
    engine = CynbitAIScopeEngine()
    sample = "We need a B2B SaaS platform with user authentication, Razorpay subscription billing, and an AI chatbot trained on our PDF manuals."
    print(json.dumps(engine.analyze_brief(sample), indent=2))