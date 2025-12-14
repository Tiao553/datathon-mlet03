import polars as pl
import json
import os
import concurrent.futures
from tqdm import tqdm
from pipe.features.prompts import chamar_llm, prompt_candidato
from pipe.features.free_text_transform import CandidatoEstruturado, extrair_json_limpo
import logging

# Setup Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("batch_extraction")

# Configuration
INPUT_FILE = "data/curated/applicants.parquet"
OUTPUT_FILE = "data/feature_store/extracted_resumes_batch.jsonl"
MAX_WORKERS = 4  # Adjust based on system resources

def process_single_candidate(row):
    """
    Process a single candidate row: generate prompt -> call LLM -> parse JSON.
    """
    candidate_id = row['codigo_candidato']
    
    # Generate Prompt
    try:
        prompt = prompt_candidato(row)
    except Exception as e:
        logger.error(f"Error generating prompt for {candidate_id}: {e}")
        return None

    # Call LLM
    try:
        # Using the simple chamar_llm from prompts.py
        # You might want to implement retry logic here similar to free_text_transform if reliability is low
        response_text = chamar_llm(prompt, model_name="gemma3:1b")
        
        # Parse and Validate
        try:
            data_dict = extrair_json_limpo(response_text)
        except Exception as e:
            logger.error(f"JSON parsing failed. Raw text: {response_text[:200]}...")
            return None
        
        # Validate with Pydantic (optional but good for consistency)
        try:
            CandidatoEstruturado.model_validate(data_dict)
        except Exception as validation_error:
            logger.warning(f"Validation failed for {candidate_id}: {validation_error}. Saving raw anyway.")
        
        # Add ID
        data_dict['codigo_candidato'] = candidate_id
        return data_dict

    except Exception as e:
        logger.error(f"LLM call failed for {candidate_id}: {e}")
        return None

def run_batch_extraction():
    if not os.path.exists(INPUT_FILE):
        print(f"Input file not found: {INPUT_FILE}")
        return

    print(f"Reading {INPUT_FILE}...")
    df = pl.read_parquet(INPUT_FILE)
    
    # Rename columns to match what prompt_candidato expects if needed
    # prompt_candidato uses 'app_cv_pt'. 
    # Check if we need to rename.
    if 'cv_pt' in df.columns and 'app_cv_pt' not in df.columns:
         df = df.rename({'cv_pt': 'app_cv_pt'})
    
    # Filter already processed
    processed_ids = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r') as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    processed_ids.add(rec['codigo_candidato'])
                except:
                    pass
    
    print(f"Already processed: {len(processed_ids)} records")
    
    # Candidates to process
    candidates_to_process = df.filter(~pl.col('codigo_candidato').is_in(processed_ids))
    
    # For testing, assume we process a chunk or all. 
    # Warning: Running all might take a long time on CPU only.
    candidates_to_process = candidates_to_process.head(5) # Comment this out for full run
    
    print(f"Starting processing for {len(candidates_to_process)} records with {MAX_WORKERS} workers...")
    
    futures_map = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for row in candidates_to_process.to_dicts():
            future = executor.submit(process_single_candidate, row)
            futures_map[future] = row['codigo_candidato']
        
        with open(OUTPUT_FILE, 'a') as f:
            for future in tqdm(concurrent.futures.as_completed(futures_map), total=len(futures_map)):
                result = future.result()
                if result:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    f.flush()

    print(f"Batch extraction complete. Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    run_batch_extraction()
