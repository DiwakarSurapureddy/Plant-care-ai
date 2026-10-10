"""Evaluation dataset and automated benchmark test suite for Plant Care AI.

Evaluates 10 representative queries covering:
- Indoor & outdoor plant care
- Flowering, medicinal, succulent plants
- Watering, sunlight, soil, fertilizer, propagation
- Symptom diagnosis (yellow leaves)
- Nursery pricing & unverified price handling
- Related plants & comparison
- Out-of-domain / ungrounded questions
- Missing Ollama service and missing model resilience
"""
import pytest
from src.chatbot import PlantCareChatbot
from src.ollama_client import OllamaClient


EVALUATION_DATASET = [
    {
        "id": "Q1",
        "question": "How do I care for a Money Plant?",
        "expected_plant": "money-plant",
        "expected_keywords": ["water", "sunlight"],
        "expect_grounded": True
    },
    {
        "id": "Q2",
        "question": "How much sunlight does a Rose need?",
        "expected_plant": "rose-plant",
        "expected_keywords": ["sunlight"],
        "expect_grounded": True
    },
    {
        "id": "Q3",
        "question": "How can I grow Tulsi at home?",
        "expected_plant": "tulsi-plant",
        "expected_keywords": ["cuttings", "seed"],
        "expect_grounded": True
    },
    {
        "id": "Q4",
        "question": "Why are my Hibiscus leaves turning yellow?",
        "expected_plant": "hibiscus-plant",
        "expected_keywords": ["yellow", "water"],
        "expect_grounded": True
    },
    {
        "id": "Q5",
        "question": "What is the price of an Aloe Vera plant?",
        "expected_plant": "aloe-vera",
        "expected_keywords": ["price", "not available"],
        "expect_grounded": True
    },
    {
        "id": "Q6",
        "question": "Which plants are suitable for indoor spaces?",
        "expected_plant": "money-plant",
        "expected_keywords": ["indoor"],
        "expect_grounded": True
    },
    {
        "id": "Q7",
        "question": "Suggest plants similar to Money Plant.",
        "expected_plant": "money-plant",
        "expected_keywords": ["pothos", "philodendron", "syngonium"],
        "expect_grounded": True
    },
    {
        "id": "Q8",
        "question": "Compare Rose and Hibiscus care requirements.",
        "expected_plant": "rose-plant",
        "expected_keywords": ["sunlight"],
        "expect_grounded": True
    },
    {
        "id": "Q9",
        "question": "How to calibrate a quantum reactor?",
        "expected_plant": None,
        "expected_keywords": ["could not find reliable information"],
        "expect_grounded": False
    },
    {
        "id": "Q10",
        "question": "Can I propagate Aloe Vera using offsets or pups?",
        "expected_plant": "aloe-vera",
        "expected_keywords": ["pups", "offset"],
        "expect_grounded": True
    }
]


@pytest.fixture(scope="module")
def bot():
    return PlantCareChatbot()


def test_missing_ollama_service_resilience():
    """Verify system handles disconnected Ollama gracefully without crash."""
    offline_client = OllamaClient(base_url="http://127.0.0.1:99999", model_name="nonexistent")
    status = offline_client.check_status()

    assert status.is_running is False
    assert "Connection" in status.status_message or "Error" in status.status_message

    success, err_msg = offline_client.generate("Hello")
    assert success is False
    assert "Connection refused" in err_msg or "error" in err_msg.lower()


def test_missing_model_resilience():
    """Verify system diagnostics when Ollama service exists but requested model is missing."""
    client = OllamaClient(model_name="fictional-nonexistent-model-xyz")
    status = client.check_status()
    if status.is_running:
        assert status.is_configured_model_available is False
        assert any("ollama pull" in cmd for cmd in status.recommended_commands)


@pytest.mark.parametrize("item", EVALUATION_DATASET)
def test_evaluation_dataset_retrieval_and_grounding(bot, item):
    """Run full evaluation suite verifying retrieval quality and grounding."""
    resp = bot.chat(item["question"], retrieval_mode="hybrid", top_k=4)

    # 1. Unrelated / Out-of-domain handling
    if not item["expect_grounded"]:
        assert resp.raw_evidence_count == 0 or "could not find reliable information" in resp.answer.lower()
        return

    # 2. Retrieval Quality
    assert resp.raw_evidence_count > 0, f"No evidence retrieved for {item['id']}: {item['question']}"
    assert len(resp.source_documents) > 0

    # Ensure expected plant file is present in retrieved sources
    expected_plant_key = item["expected_plant"].lower()
    sources_text = " ".join(resp.source_documents).lower().replace("-", " ")
    expected_normalized = expected_plant_key.replace("-", " ")
    assert expected_normalized in sources_text, (
        f"Expected plant '{item['expected_plant']}' not in sources: {resp.source_documents}"
    )

    # 3. Answer Grounding Quality
    answer_lower = resp.answer.lower()
    # At least one of the expected keywords should be covered in the grounded answer
    matched_keywords = [kw for kw in item["expected_keywords"] if kw.lower() in answer_lower]
    assert len(matched_keywords) > 0, (
        f"For question {item['id']} ('{item['question']}'), none of keywords {item['expected_keywords']} "
        f"were found in answer:\n{resp.answer}"
    )
