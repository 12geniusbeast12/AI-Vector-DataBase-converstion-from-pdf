import urllib.request
import json
import ssl

api_key = ""
url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"

context = ssl._create_unverified_context()

try:
    with urllib.request.urlopen(url, context=context) as response:
        data = json.loads(response.read().decode())
        print("Available Embedding Models:")
        for model in data.get('models', []):
            name = model.get('name', '')
            methods = model.get('supportedGenerationMethods', [])
            if 'embedContent' in methods:
                print(f"- {name}")
except Exception as e:
    print(f"Error: {e}")
