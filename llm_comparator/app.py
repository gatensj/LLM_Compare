import os
import asyncio
import time
from flask import Flask, render_template, request, flash, session
from dotenv import load_dotenv
import httpx  # For async HTTP requests (Ollama and potentially others)

# Import LLM SDKs
from openai import OpenAI, AzureOpenAI, APIError, APITimeoutError, RateLimitError
from anthropic import Anthropic, APIError as AnthropicAPIError, RateLimitError as AnthropicRateLimitError
import google.generativeai as genai
from google.api_core import exceptions as GoogleAPIErrors

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
# Secret key for session management (e.g., flash messages)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'default-secret-key-for-dev')
# Configure HTTPX timeout (seconds)
HTTPX_TIMEOUT = 30.0

# --- Configuration & Client Initialization ---

# Model Identifiers (used in forms and logic)
MODEL_IDS = {
    "openai": "OpenAI (GPT)",
    "claude": "Anthropic (Claude)",
    "gemini": "Google (Gemini)",
}

'''
# TODO: Add the local model

MODEL_IDS = {
    "openai": "OpenAI (GPT)",
    "azure": "Azure OpenAI",
    "claude": "Anthropic (Claude)",
    "gemini": "Google (Gemini)",
    "ollama": "Ollama (Local)",
}
'''

# Store config errors to display to the user
config_errors = []
config_errors_detail = {} # Store specific errors per model ID

# --- OpenAI Client ---
openai_api_key = os.getenv("OPENAI_API_KEY")
openai_model = os.getenv("OPENAI_MODEL", "gpt-4.1-2025-04-14") # Default to Haiku for speed/cost

openai_client = None
if openai_api_key:
    try:
        openai_client = OpenAI(api_key=openai_api_key, timeout=HTTPX_TIMEOUT)
    except Exception as e:
        err_msg = f"OpenAI: Failed to initialize client - {e}"
        config_errors.append(err_msg)
        config_errors_detail["openai"] = err_msg
else:
    config_errors_detail["openai"] = "OpenAI API Key not found in .env"


# --- Azure OpenAI Client ---
azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
azure_client = None
if azure_api_key and azure_endpoint and azure_deployment:
    try:
        azure_client = AzureOpenAI(
            api_key=azure_api_key,
            api_version="2024-02-01", # Use a recent, stable API version
            azure_endpoint=azure_endpoint,
            timeout=HTTPX_TIMEOUT,
        )
    except Exception as e:
        err_msg = f"Azure OpenAI: Failed to initialize client - {e}"
        config_errors.append(err_msg)
        config_errors_detail["azure"] = err_msg
elif not (not azure_api_key and not azure_endpoint and not azure_deployment): # Only add error if some Azure vars are set
    config_errors_detail["azure"] = "Azure OpenAI config incomplete (Key, Endpoint, Deployment Name required)"

# --- Anthropic Client ---
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
anthropic_model = os.getenv("ANTHROPIC_MODEL", "claude-3-haiku-20240307") # Default to Haiku for speed/cost
anthropic_client = None
if anthropic_api_key:
    try:
        anthropic_client = Anthropic(api_key=anthropic_api_key, timeout=HTTPX_TIMEOUT)
    except Exception as e:
        err_msg = f"Anthropic: Failed to initialize client - {e}"
        config_errors.append(err_msg)
        config_errors_detail["claude"] = err_msg
else:
    config_errors_detail["claude"] = "Anthropic API Key not found in .env"

# --- Google Gemini Client ---
google_api_key = os.getenv("GOOGLE_API_KEY")
gemini_model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash-latest") # Default to Flash for speed/cost
gemini_client = None
if google_api_key:
    try:
        genai.configure(api_key=google_api_key)
        # Check if the model exists/is accessible during init or first call
        # For simplicity, we'll assume config is okay and catch errors during the call
        gemini_client = genai.GenerativeModel(gemini_model_name)
    except Exception as e:
        err_msg = f"Google Gemini: Failed to configure - {e}"
        config_errors.append(err_msg)
        config_errors_detail["gemini"] = err_msg
else:
    config_errors_detail["gemini"] = "Google API Key not found in .env"

# --- Ollama Config ---
ollama_base_url = os.getenv("OLLAMA_BASE_URL")
ollama_model = os.getenv("OLLAMA_MODEL")
ollama_configured = bool(ollama_base_url and ollama_model)
if not ollama_configured and (ollama_base_url or ollama_model):
     config_errors_detail["ollama"] = "Ollama config incomplete (Base URL and Model required)"
elif not ollama_configured:
     config_errors_detail["ollama"] = "Ollama config not found in .env"


# --- Async Helper Functions for API Calls ---

async def get_openai_response(prompt: str) -> dict:
    """Gets response from OpenAI API."""
    if not openai_client:
        return {"success": False, "error": "OpenAI client not configured."}
    try:
        # Using chat completions endpoint
        response = await asyncio.to_thread(
            openai_client.chat.completions.create,
            model=openai_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000, # Limit response length
            temperature=0.7 # Adjust creativity
        )
        # Check if response is valid and has choices
        if response and response.choices and response.choices[0].message:
             return {"success": True, "response": response.choices[0].message.content.strip()}
        else:
             return {"success": False, "error": "Invalid response structure from OpenAI."}
    except (APIError, APITimeoutError, RateLimitError) as e:
        return {"success": False, "error": f"OpenAI API Error: {type(e).__name__} - {e}"}
    except Exception as e:
        return {"success": False, "error": f"OpenAI Unexpected Error: {e}"}

async def get_azure_openai_response(prompt: str) -> dict:
    """Gets response from Azure OpenAI API."""
    if not azure_client:
        return {"success": False, "error": "Azure OpenAI client not configured."}
    if not azure_deployment: # Should be caught by client init, but double check
        return {"success": False, "error": "Azure OpenAI deployment name not set."}
    try:
        response = await asyncio.to_thread(
             azure_client.chat.completions.create,
            model=azure_deployment, # Use deployment name here
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0.7
        )
        if response and response.choices and response.choices[0].message:
            return {"success": True, "response": response.choices[0].message.content.strip()}
        else:
            return {"success": False, "error": "Invalid response structure from Azure OpenAI."}

    except (APIError, APITimeoutError, RateLimitError) as e:
         # Specific OpenAI SDK errors likely apply here too
        return {"success": False, "error": f"Azure API Error: {type(e).__name__} - {e}"}
    except Exception as e:
        # Catch potential generic errors like connection issues
        return {"success": False, "error": f"Azure Unexpected Error: {e}"}


async def get_claude_response(prompt: str) -> dict:
    """Gets response from Anthropic Claude API."""
    if not anthropic_client:
        return {"success": False, "error": "Anthropic client not configured."}
    try:
        # Use the messages API
        response = await asyncio.to_thread(
            anthropic_client.messages.create,
            model=anthropic_model,
            max_tokens=1000,
            temperature=0.7,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        # Check response structure
        if response and response.content and isinstance(response.content, list) and response.content[0].text:
            return {"success": True, "response": response.content[0].text.strip()}
        else:
            return {"success": False, "error": "Invalid response structure from Anthropic."}
    except (AnthropicAPIError, AnthropicRateLimitError) as e:
        return {"success": False, "error": f"Anthropic API Error: {type(e).__name__} - {e}"}
    except Exception as e:
        return {"success": False, "error": f"Anthropic Unexpected Error: {e}"}

async def get_gemini_response(prompt: str) -> dict:
    """Gets response from Google Gemini API."""
    if not gemini_client:
        return {"success": False, "error": "Google Gemini client not configured."}
    try:
        # Use generate_content_async for non-blocking call
        response = await gemini_client.generate_content_async(
            prompt,
            generation_config=genai.types.GenerationConfig(
                # candidate_count=1, # Default is 1
                max_output_tokens=1000,
                temperature=0.7,
            )
        )
        # Accessing the text content, handling potential blocks/safety issues
        if response and response.parts:
             return {"success": True, "response": response.text.strip()}
        elif response.prompt_feedback.block_reason:
             return {"success": False, "error": f"Blocked due to: {response.prompt_feedback.block_reason.name}"}
        else:
             # Attempt to get text even if parts might be structured differently,
             # or provide a generic error if text is unavailable.
             try:
                 return {"success": True, "response": response.text.strip()}
             except ValueError: # If response.text raises ValueError (e.g. function call)
                 return {"success": False, "error": "Gemini response format not text or blocked."}
             except Exception: # Catch other potential issues accessing .text
                 return {"success": False, "error": "Could not extract text from Gemini response."}

    except GoogleAPIErrors.GoogleAPIError as e:
         # Handle specific Google API errors
        return {"success": False, "error": f"Google API Error: {e}"}
    except GoogleAPIErrors.RetryError as e:
        return {"success": False, "error": f"Google API Retry Error: {e}"}
    except Exception as e:
        return {"success": False, "error": f"Google Gemini Unexpected Error: {e}"}


async def get_ollama_response(prompt: str) -> dict:
    """Gets response from local Ollama API."""
    if not ollama_configured:
        return {"success": False, "error": "Ollama not configured (Base URL or Model missing)."}

    ollama_api_url = f"{ollama_base_url.rstrip('/')}/api/generate"
    payload = {
        "model": ollama_model,
        "prompt": prompt,
        "stream": False # Get the full response at once
    }
    try:
        async with httpx.AsyncClient(timeout=HTTPX_TIMEOUT) as client:
            response = await client.post(ollama_api_url, json=payload)
            response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
            response_data = response.json()

            if response_data and "response" in response_data:
                return {"success": True, "response": response_data["response"].strip()}
            else:
                # Handle cases where the response might be missing the expected key
                 err_detail = response_data.get('error', 'Unknown format')
                 return {"success": False, "error": f"Ollama response missing 'response' key. Detail: {err_detail}"}

    except httpx.HTTPStatusError as e:
        # More specific error for HTTP errors
        error_body = ""
        try:
             # Try to get error details from response body if available
             error_body = e.response.json().get('error', e.response.text)
        except Exception:
             error_body = e.response.text # Fallback to raw text

        # Check for common Ollama errors
        if "model not found" in error_body.lower():
             return {"success": False, "error": f"Ollama Error: Model '{ollama_model}' not found. Make sure it's pulled."}

        return {"success": False, "error": f"Ollama HTTP Error: {e.response.status_code} - {error_body}"}
    except httpx.RequestError as e:
        # Handles connection errors, timeouts, etc.
        return {"success": False, "error": f"Ollama Connection Error: Could not connect to {ollama_api_url}. Is Ollama running? Details: {e}"}
    except Exception as e:
        return {"success": False, "error": f"Ollama Unexpected Error: {e}"}


# Mapping model IDs to their async handler functions
MODEL_HANDLERS = {
    "openai": get_openai_response,
    "azure": get_azure_openai_response,
    "claude": get_claude_response,
    "gemini": get_gemini_response,
    "ollama": get_ollama_response,
}

# --- Flask Routes ---

@app.route('/', methods=['GET'])
def index():
    """Renders the main page."""
    # Pass available models and any config errors to the template
    return render_template('index.html',
                           available_models=MODEL_IDS,
                           selected_models=session.get('selected_models', []), # Remember selections
                           prompt_text=session.get('prompt_text', ''),     # Remember prompt
                           results=None,
                           config_errors=config_errors,
                           config_errors_detail=config_errors_detail)

@app.route('/compare', methods=['POST'])
async def compare():
    """Handles form submission, runs comparisons, and renders results."""
    start_time = time.time()
    prompt = request.form.get('prompt')
    selected_models = request.form.getlist('models') # Gets list of checked values

    # Store prompt and selections in session to repopulate form
    session['prompt_text'] = prompt
    session['selected_models'] = selected_models

    results = {}
    duration = None

    if not prompt:
        flash("Please enter a prompt.", "error")
        return render_template('index.html',
                               available_models=MODEL_IDS,
                               selected_models=selected_models,
                               prompt_text=prompt,
                               results=None,
                               config_errors=config_errors,
                               config_errors_detail=config_errors_detail)

    if not selected_models:
        flash("Please select at least one model.", "error")
        return render_template('index.html',
                               available_models=MODEL_IDS,
                               selected_models=selected_models,
                               prompt_text=prompt,
                               results=None,
                               config_errors=config_errors,
                               config_errors_detail=config_errors_detail)

    # Create asyncio tasks for selected models
    tasks = []
    valid_selected_models = [] # Track models we actually attempt to call
    for model_id in selected_models:
        handler = MODEL_HANDLERS.get(model_id)
        # Only create a task if the handler exists and no config error disables it
        if handler and not config_errors_detail.get(model_id):
            tasks.append(asyncio.create_task(handler(prompt)))
            valid_selected_models.append(model_id)
        elif config_errors_detail.get(model_id):
            # If it was selected but had a config error, show the error directly
             results[model_id] = {"success": False, "error": f"Configuration Error: {config_errors_detail[model_id]}"}


    # Run tasks concurrently and gather results
    if tasks:
        gathered_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results, mapping back to model IDs
        for i, result in enumerate(gathered_results):
            model_id = valid_selected_models[i] # Get corresponding model ID
            if isinstance(result, Exception):
                # If asyncio.gather caught an unexpected exception in the handler itself
                 results[model_id] = {"success": False, "error": f"Internal Task Error: {result}"}
                 print(f"Internal error for {model_id}: {result}") # Log internal errors
            else:
                 # Result should be the dictionary {"success": bool, "response/error": str}
                 results[model_id] = result

    duration = time.time() - start_time

    # Render the template with the results
    return render_template('index.html',
                           available_models=MODEL_IDS,
                           selected_models=selected_models, # Pass back original selection
                           prompt_text=prompt,
                           results=results,
                           duration=duration,
                           config_errors=config_errors, # Still show persistent config errors
                           config_errors_detail=config_errors_detail)

if __name__ == '__main__':
    # Check for essential config errors on startup
    if not app.secret_key or app.secret_key == 'default-secret-key-for-dev':
        print("WARNING: FLASK_SECRET_KEY is not set or is insecure in .env. Please set a strong secret key.")
    if config_errors:
        print("\n--- Configuration Errors Detected ---")
        for error in config_errors:
            print(f"- {error}")
        print("-----------------------------------\n")

    # Use development server. For production, use a proper WSGI server like Gunicorn or Waitress.
    # `debug=True` enables auto-reloading and detailed error pages (disable in production).
    app.run(debug=True, host='0.0.0.0', port=5001) # Run on port 5001 to avoid conflicts