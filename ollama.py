import requests

def send_ollama_prompt(prompt: str, model: str = "mistral") -> str:
    try:
        response = requests.post("http://localhost:11434/api/generate", json={"model": model, "prompt": prompt, "stream": False}, timeout=10)
        #print(f"Response: {response.json()}")
        data = response.json()

        if "response" not in data:
            # no response
            print("Ollama error:", data.get("error", "Unknown error"))
            return None

        return data["response"]
    
    # catch
    except requests.exceptions.ConnectionError:
        print("Exception: Connection error")
    except requests.exceptions.Timeout:
        print("Exception: Ollama timed out.")
    except requests.exceptions.RequestException as e:
        print(f"Exception: HTTP request error: {e}")
    except Exception as e:
        print(f"Exception: Unexpected error: {e}")
    
    return None

def get_components(object_name: str):
    prompt = (
        f"List only the physical parts of a {object_name} that are visible in a photo. "
        f"Answer concisely as a comma-separated list. No descriptions or categories. No parentheses. I repeat do NOT include parentheses in response."
        )
    
    # send prompt
    response = send_ollama_prompt(prompt=prompt)
    
    if response:
        return clean_component_list(response)
    else:
        return []

def clean_component_list(response_text):
    # Lowercase and remove trailing punctuation
    text = response_text.lower().replace("\n", " ").replace(".", "")
    
    # If the whole list came back as one string, flatten it
    if isinstance(response_text, list) and len(response_text) == 1:
        text = response_text[0].lower().replace("\n", " ").replace(".", "")
    
    # Split on commas and clean each item
    parts = [part.strip() for part in text.split(",") if part.strip()]
    
    return parts