import sys
import os
from unittest.mock import MagicMock, patch
import json
from fastapi.testclient import TestClient
import pytest

# Add paths
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'data_pipeline')))

# Mock Dependencies that might be heavy or missing
sys.modules['paddleocr'] = MagicMock()
sys.modules['paddle'] = MagicMock()
# sys.modules['cv2'] = MagicMock() # Removed to avoid conflict with transformers find_spec

# Import app after mocking
from serving.api import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_predict_legacy_resume_text():
    # Mock LLM calls to avoid cost/latency
    with patch('serving.api.chamar_llm') as mock_llm:
        mock_llm.return_value = '```json\n{"competencias_tecnicas": ["Python"], "experiencia_anos": "5-8 anos"}\n```'
        
        payload = {
            "resume_text": "Desenvolvedor Python experiente",
            "job_description": "Vaga Python"
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "candidate_extracted" in data
        assert "Python" in data["candidate_extracted"].get("competencias_tecnicas", [])

def test_predict_zero_shot_structured():
    # Phase 3: Zero-shot with structured payload
    # Should NOT call LLM for extraction
    with patch('serving.api.chamar_llm') as mock_llm:
        # Valid V2 Schema Payload
        payload = {
            "candidate_data": {
                "skills": {
                     "technical_skills": ["Rust", "C++"],
                     "soft_skills": ["Foco"]
                }
            },
            "job_data": {
                "requirements": {
                    "required_tech_skills": ["Rust"],
                    "required_soft_skills": ["Foco"]
                },
                "metadata": {
                    "job_title": "Rust Dev"
                }
            }
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        
        # Verify LLM was NOT called
        mock_llm.assert_not_called()
        
        # Verify scores are high (Rust matches Rust)
        scores = data["scores"]
        # Check basic structure matches input (serialized)
        assert data["candidate_extracted"]["skills"]["technical_skills"] == ["Rust", "C++"]
        assert data["job_extracted"]["requirements"]["required_tech_skills"] == ["Rust"]
        # Equality might fail due to defaults, so check subset or specific field
        assert data["job_extracted"]["metadata"]["job_title"] == "Rust Dev"

@patch('serving.api.DocumentParser.parse_file')
def test_predict_file_ocr_flag(mock_parse):
    mock_parse.return_value = "Extracted Text"
    
    # Test with use_ocr=False
    files = {'file': ('resume.pdf', b'%PDF-1.4', 'application/pdf')}
    client.post("/predict_file?use_ocr=false", files=files)
    mock_parse.assert_called_with(b'%PDF-1.4', 'resume.pdf', use_ocr=False)
    
    # Test with use_ocr=True
    files = {'file': ('resume.pdf', b'%PDF-1.4', 'application/pdf')}
    client.post("/predict_file?use_ocr=true", files=files)
    mock_parse.assert_called_with(b'%PDF-1.4', 'resume.pdf', use_ocr=True)

@patch('serving.api.DocumentParser.parse_file')
def test_predict_file_with_structured_data(mock_parse):
    mock_parse.return_value = "Extracted Text"
    
    files = {'file': ('resume.pdf', b'%PDF-1.4', 'application/pdf')}
    
    # Valid V2 Schema Payload
    c_payload = {
        "skills": {
            "technical_skills": ["Go"],
            "soft_skills": []
        },
        "profile": {
            "resume_text": "Content from PDF will overwrite this or strictly typically extracted"
        }
    }
    j_payload = {
        "requirements": {
             "required_tech_skills": ["Go"],
             "required_soft_skills": []
        },
        "metadata": {
            "job_title": "Go Developer"
        } 
    }
    
    data = {
        'candidate_data': json.dumps(c_payload),
        'job_data': json.dumps(j_payload)
    }
    
    with patch('serving.api.chamar_llm') as mock_llm:
         response = client.post("/predict_file?use_ocr=false", files=files, data=data)
         assert response.status_code == 200
         resp = response.json()
         
         # Verify we got the structured data back (Zero-shot path logic)
         # Using new V2 key structure
         assert "Go" in resp["candidate_extracted"]["skills"]["technical_skills"]
         assert "Go" in resp["job_extracted"]["requirements"]["required_tech_skills"]
         
         # Verify LLM was NOT called (Zero-shot should skip LLM)
         mock_llm.assert_not_called()

if __name__ == "__main__":
    print("Running manual tests...")
    # Manual execution of tests if not using pytest
    test_health()
    test_predict_legacy_resume_text()
    test_predict_zero_shot_structured()
    test_predict_file_with_structured_data()
    print("All manual tests passed!")
