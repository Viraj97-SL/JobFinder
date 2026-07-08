"""
JobForge AI — Role Category Classifier Tests.

Labelled test set of real-world-style UK AI/ML/DS job titles. Acceptance
target from the upgrade plan: >=85% accuracy on a >=40-title labelled set.
"""

from __future__ import annotations

from src.jobforge.analytics.role_classifier import classify_role_category

# (title, expected_category)
LABELLED_TITLES: list[tuple[str, str]] = [
    ("MLOps Engineer", "MLOps"),
    ("Senior MLOps Engineer", "MLOps"),
    ("ML Platform Engineer", "MLOps"),
    ("Machine Learning Infrastructure Engineer", "MLOps"),
    ("Computer Vision Engineer", "Computer Vision"),
    ("Senior Computer Vision Engineer - Autonomous Driving", "Computer Vision"),
    ("CV Engineer (Image Recognition)", "Computer Vision"),
    ("NLP Engineer", "NLP"),
    ("Natural Language Processing Scientist", "NLP"),
    ("Senior NLP Engineer - Chatbots", "NLP"),
    ("Research Scientist - Reinforcement Learning", "Research"),
    ("Research Engineer, Foundation Models", "Research"),
    ("Applied Scientist", "Research"),
    ("AI Engineer", "AI/LLM Engineer"),
    ("Senior AI Engineer", "AI/LLM Engineer"),
    ("LLM Engineer", "AI/LLM Engineer"),
    ("Generative AI Engineer", "AI/LLM Engineer"),
    ("GenAI Solutions Engineer", "AI/LLM Engineer"),
    ("Prompt Engineer", "AI/LLM Engineer"),
    ("AI/ML Engineer", "AI/LLM Engineer"),
    ("Artificial Intelligence Engineer", "AI/LLM Engineer"),
    ("Machine Learning Engineer", "ML Engineer"),
    ("Senior Machine Learning Engineer", "ML Engineer"),
    ("ML Engineer", "ML Engineer"),
    ("Staff ML Engineer", "ML Engineer"),
    ("Machine Learning Engineer - Content Intelligence", "ML Engineer"),
    ("Data Scientist", "Data Scientist"),
    ("Senior Data Scientist", "Data Scientist"),
    ("Lead Data Scientist - Pricing", "Data Scientist"),
    ("Data Scientist (NLP)", "Data Scientist"),
    ("Data Engineer", "Data Engineer"),
    ("Senior Data Engineer", "Data Engineer"),
    ("Data Engineer - Analytics Platform", "Data Engineer"),
    ("Principal Data Engineer", "Data Engineer"),
    ("Backend Software Engineer", "Other"),
    ("Product Manager - AI Products", "Other"),
    ("Solutions Architect", "Other"),
    ("Frontend Developer", "Other"),
    ("Engineering Manager, ML Platform", "MLOps"),
    ("Head of Data Science", "Data Scientist"),
    ("DevOps Engineer", "Other"),
    ("QA Engineer", "Other"),
    ("Business Analyst", "Other"),
]


def test_labelled_set_has_minimum_size():
    assert len(LABELLED_TITLES) >= 40


def test_classifier_accuracy_meets_threshold():
    correct = sum(
        1 for title, expected in LABELLED_TITLES if classify_role_category(title) == expected
    )
    accuracy = correct / len(LABELLED_TITLES)
    assert accuracy >= 0.85, f"accuracy {accuracy:.2%} below 85% threshold"


def test_mlops_takes_priority_over_generic_ml_engineer():
    assert classify_role_category("MLOps Engineer") == "MLOps"


def test_computer_vision_takes_priority_over_generic_engineer():
    assert classify_role_category("Computer Vision Engineer") == "Computer Vision"


def test_unrelated_role_is_other():
    assert classify_role_category("Frontend Developer") == "Other"


def test_falls_back_to_description_when_title_has_no_signal():
    category = classify_role_category(
        "Engineer II",
        description="You'll build and deploy machine learning models as a machine learning engineer.",
    )
    assert category == "ML Engineer"
