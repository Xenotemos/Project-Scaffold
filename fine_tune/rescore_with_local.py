import json
import os
import time
import urllib.request
import urllib.error
from pathlib import Path

# Configuration
# Default to standard llama.cpp server port
API_URL = os.environ.get("LOCAL_LLM_URL", "http://localhost:8080/v1/chat/completions")
INPUT_FILE = Path("fine_tune/harvest/run003_external_subset.label.jsonl")
OUTPUT_FILE = Path("fine_tune/harvest/run003_external_rescored.jsonl")
BATCH_SIZE = 1  # Single item per request (model returns one object).

def create_prompt(batch):
    intro = """You are an expert in affective computing.
For each input, infer the affect scores based on the user utterance and archetype hint.

Definitions:
- valence: pleasant(+1) <-> painful(-1)
- intimacy: closeness/relational pull (+1) <-> distance (-1)
- tension: bodily arousal/urgency (+1) <-> relaxed (-1)
- confidence: overall certainty (0-1)

Output a single JSON object per input, in the exact order provided.
Each object must have: {"valence": float, "intimacy": float, "tension": float, "confidence": float, "tags": [str], "reason": str}
- Keep "reason" to a single concise sentence (<=120 characters).
- Keep 3 or fewer short tags per item.

Inputs:
"""
    item = batch[0]
    item_str = f"""
Archetype hint: {item['archetype']}
User text: ```{item['text']}```
"""
    return intro + item_str + "\nOutput JSON object:"

def call_local_llm(prompt):
    payload = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that outputs strict JSON."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0,
        "max_tokens": 4096, # Ensure enough space for the JSON response
        "stream": False,
        "response_format": {"type": "json_object"} # generic hint, some servers support strict schema
    }
    
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        API_URL, 
        data=data, 
        headers={"Content-Type": "application/json"}
    )
    
    try:
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode("utf-8"))
            content = result["choices"][0]["message"]["content"]
            return content
    except urllib.error.URLError as e:
        print(f"API Request failed: {e}")
        if hasattr(e, 'read'):
            print(e.read().decode('utf-8'))
        raise

def process_file():
    print(f"Connecting to Local LLM at {API_URL}...")
    print(f"Input: {INPUT_FILE}")
    print(f"Output: {OUTPUT_FILE}")
    
    # Read existing progress
    processed_count = 0
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            processed_count = sum(1 for _ in f)
        print(f"Resuming from line {processed_count}...")

    # Read input lines
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()

    total_lines = len(lines)

    # Process line by line
    for i in range(processed_count, total_lines):
        batch_lines = lines[i : i + 1]
        batch_data = [json.loads(line) for line in batch_lines]

        print(f"Processing entry {i+1}/{total_lines}...")
        
        prompt = create_prompt(batch_data)
        
        retries = 3
        while retries > 0:
            try:
                json_str = call_local_llm(prompt)
                
                # Clean up potential markdown code blocks if the model adds them
                if "```json" in json_str:
                    json_str = json_str.split("```json")[1].split("```")[0].strip()
                elif "```" in json_str:
                    json_str = json_str.split("```")[1].split("```")[0].strip()
                
                # Parse response
                try:
                    result_obj = json.loads(json_str)
                except json.JSONDecodeError:
                    import re
                    match = re.search(r'\{.*\}', json_str, re.DOTALL)
                    if match:
                        result_obj = json.loads(match.group(0))
                    else:
                        print("Model response preview:", json_str[:500])
                        raise ValueError("Could not extract JSON object from model response")

                if not isinstance(result_obj, dict):
                    print("Raw model response preview:", json_str[:500])
                    raise ValueError("Model response was not a JSON object.")
                batch_data_subset = batch_data

                # Merge and write
                with open(OUTPUT_FILE, "a", encoding="utf-8") as out_f:
                    original = batch_data_subset[0]
                    merged = original.copy()
                    merged.update(result_obj)
                    # Ensure clean float format
                    merged['valence'] = float(merged.get('valence', 0.0))
                    merged['intimacy'] = float(merged.get('intimacy', 0.0))
                    merged['tension'] = float(merged.get('tension', 0.0))
                    merged['confidence'] = float(merged.get('confidence', 0.0))
                    # Limit tags/reason length
                    tags = merged.get('tags') or []
                    if isinstance(tags, list):
                        merged['tags'] = [str(tag)[:40] for tag in tags[:3]]
                    reason = merged.get('reason')
                    if isinstance(reason, str):
                        merged['reason'] = reason.strip()[:120]
                    out_f.write(json.dumps(merged) + "\n")
                
                break # Success
                
            except Exception as e:
                print(f"Error: {e}")
                retries -= 1
                if retries == 0:
                    print(f"Failed to process batch starting at {i}. Skipping...")
                time.sleep(2) # Backoff

    print("Rescoring complete.")

if __name__ == "__main__":
    process_file()
