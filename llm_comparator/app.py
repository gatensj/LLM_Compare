import os
import asyncio
import time
import logging

from flask import Flask, render_template, request, flash, session
from dotenv import load_dotenv
import httpx  # For async HTTP requests (Ollama and potentially others)
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
    #"azure": "Azure OpenAI", # Assuming you might add Azure back
    "claude": "Anthropic (Claude)",
    "gemini": "Google (Gemini)",
    "grok": "xAI (Grok)",
    #"ollama": "Ollama (Local)", # Assuming you might add Ollama back
}


# Store config errors to display to the user
config_errors = []
config_errors_detail = {} # Store specific errors per model ID

# --- OpenAI Client (Standard) ---
openai_api_key = os.getenv("OPENAI_API_KEY")
openai_model = os.getenv("OPENAI_MODEL", "gpt-4o") # Default to a capable model

openai_client = None
if openai_api_key:
    try:
        openai_client = OpenAI(api_key=openai_api_key, timeout=HTTPX_TIMEOUT)
    except Exception as e:
        err_msg = f"OpenAI: Failed to initialize client - {e}"
        config_errors.append(err_msg)
        config_errors_detail["openai"] = err_msg
else:
    config_errors_detail["openai"] = "OpenAI API Key (OPENAI_API_KEY) not found in .env"


# --- Azure OpenAI Client ---
# (Keeping the Azure config structure in case you add it back)
azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
azure_client = None
if azure_api_key and azure_endpoint and azure_deployment:
    try:
        azure_client = AzureOpenAI(
            api_key=azure_api_key,
            api_version="2024-02-01",
            azure_endpoint=azure_endpoint,
            timeout=HTTPX_TIMEOUT,
        )
    except Exception as e:
        err_msg = f"Azure OpenAI: Failed to initialize client - {e}"
        config_errors.append(err_msg)
        config_errors_detail["azure"] = err_msg
elif not (not azure_api_key and not azure_endpoint and not azure_deployment):
    config_errors_detail["azure"] = "Azure OpenAI config incomplete (Key, Endpoint, Deployment Name required)"
else:
     config_errors_detail["azure"] = "Azure OpenAI config not found in .env (optional)"


# --- Anthropic Client ---
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
anthropic_model = os.getenv("ANTHROPIC_MODEL", "claude-3-haiku-20240307")
anthropic_client = None
if anthropic_api_key:
    try:
        anthropic_client = Anthropic(api_key=anthropic_api_key, timeout=HTTPX_TIMEOUT)
    except Exception as e:
        err_msg = f"Anthropic: Failed to initialize client - {e}"
        config_errors.append(err_msg)
        config_errors_detail["claude"] = err_msg
else:
    config_errors_detail["claude"] = "Anthropic API Key (ANTHROPIC_API_KEY) not found in .env"

# --- Google Gemini Client ---
google_api_key = os.getenv("GOOGLE_API_KEY")
gemini_model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash-latest")
gemini_client = None
if google_api_key:
    try:
        genai.configure(api_key=google_api_key)
        gemini_client = genai.GenerativeModel(gemini_model_name)
    except Exception as e:
        err_msg = f"Google Gemini: Failed to configure - {e}"
        config_errors.append(err_msg)
        config_errors_detail["gemini"] = err_msg
else:
    config_errors_detail["gemini"] = "Google API Key (GOOGLE_API_KEY) not found in .env"

# --- Grok (xAI) Client (Using OpenAI Library) --- # <<< === MODIFIED GROK CONFIG ===
grok_api_key = os.getenv("XAI_API_KEY")
# Use a valid Grok model name. Check xAI docs. 'grok-1' is common, example used 'grok-3-beta'
grok_model = os.getenv("GROK_MODEL", "grok-1")
grok_base_url = "https://api.x.ai/v1" # Grok API endpoint

grok_openai_client = None # Use a distinct variable name
if grok_api_key:
    try:
        # Initialize an OpenAI client instance configured for Grok
        grok_openai_client = OpenAI(
            api_key=grok_api_key,
            base_url=grok_base_url,
            timeout=HTTPX_TIMEOUT
            )
        # Optional: Add a test call here if desired, e.g., list models
        # try:
        #     grok_openai_client.models.list()
        # except Exception as test_e:
        #     raise Exception(f"Failed Grok API test call: {test_e}")

    except Exception as e:
        err_msg = f"Grok (xAI): Failed to initialize OpenAI client for Grok - {e}"
        config_errors.append(err_msg)
        config_errors_detail["grok"] = err_msg
else:
    config_errors_detail["grok"] = "Grok API Key (XAI_API_KEY) not found in .env"
# <<< === END OF MODIFIED GROK CONFIG ===

# --- Ollama Config ---
# (Keeping the Ollama config structure in case you add it back)
ollama_base_url = os.getenv("OLLAMA_BASE_URL")
ollama_model = os.getenv("OLLAMA_MODEL")
ollama_configured = bool(ollama_base_url and ollama_model)
if not ollama_configured and (ollama_base_url or ollama_model):
     config_errors_detail["ollama"] = "Ollama config incomplete (Base URL and Model required)"
elif not ollama_configured:
     config_errors_detail["ollama"] = "Ollama config not found in .env (optional)"


# --- Async Helper Functions for API Calls ---

async def get_openai_response(prompt: str) -> dict:
    """Gets response from OpenAI API."""
    if not openai_client:
        return {"success": False, "error": "OpenAI client not configured."}
    try:
        response = await asyncio.to_thread(
            openai_client.chat.completions.create,
            model=openai_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1500,
            temperature=0.7
        )
        if response and response.choices and response.choices[0].message:
             return {"success": True, "response": response.choices[0].message.content.strip()}
        else:
             return {"success": False, "error": "Invalid response structure from OpenAI."}
    except (APIError, APITimeoutError, RateLimitError) as e:
        return {"success": False, "error": f"OpenAI API Error: {type(e).__name__} - {e}"}
    except Exception as e:
        app.logger.error(f"OpenAI Unexpected Error: {e}", exc_info=True)
        return {"success": False, "error": f"OpenAI Unexpected Error: {e}"}

async def get_azure_openai_response(prompt: str) -> dict:
    """Gets response from Azure OpenAI API."""
    if not azure_client:
        return {"success": False, "error": "Azure OpenAI client not configured."}
    if not azure_deployment:
        return {"success": False, "error": "Azure OpenAI deployment name not set."}
    try:
        response = await asyncio.to_thread(
             azure_client.chat.completions.create,
            model=azure_deployment,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1500,
            temperature=0.7
        )
        if response and response.choices and response.choices[0].message:
            return {"success": True, "response": response.choices[0].message.content.strip()}
        else:
            return {"success": False, "error": "Invalid response structure from Azure OpenAI."}
    except (APIError, APITimeoutError, RateLimitError) as e:
        return {"success": False, "error": f"Azure API Error: {type(e).__name__} - {e}"}
    except Exception as e:
        app.logger.error(f"Azure Unexpected Error: {e}", exc_info=True)
        return {"success": False, "error": f"Azure Unexpected Error: {e}"}


async def get_claude_response(prompt: str) -> dict:
    """Gets response from Anthropic Claude API."""
    if not anthropic_client:
        return {"success": False, "error": "Anthropic client not configured."}
    try:
        response = await asyncio.to_thread(
            anthropic_client.messages.create,
            model=anthropic_model,
            max_tokens=1500,
            temperature=0.7,
            messages=[{"role": "user","content": prompt}]
        )
        if response and response.content and isinstance(response.content, list) and response.content[0].text:
            return {"success": True, "response": response.content[0].text.strip()}
        else:
            return {"success": False, "error": "Invalid response structure from Anthropic."}
    except (AnthropicAPIError, AnthropicRateLimitError) as e:
        return {"success": False, "error": f"Anthropic API Error: {type(e).__name__} - {e}"}
    except Exception as e:
        app.logger.error(f"Anthropic Unexpected Error: {e}", exc_info=True)
        return {"success": False, "error": f"Anthropic Unexpected Error: {e}"}

# In get_gemini_response function:
# In get_gemini_response function:
async def get_gemini_response(prompt: str) -> dict:
    """Gets response from Google Gemini API."""
    if not gemini_client:
        return {"success": False, "error": "Google Gemini client not configured."}
    try:
        # Define the generation config first
        # --- MODIFICATION START ---
        # Remove max_output_tokens to use the default
        generation_config = genai.types.GenerationConfig(
            # max_output_tokens=1500, # REMOVED
            temperature=0.7,
        )
        # --- MODIFICATION END ---

        # Wrap the SYNCHRONOUS call in asyncio.to_thread
        app.logger.info(f"Calling Gemini (via to_thread): model='{gemini_model_name}', prompt='{prompt[:50]}...'")
        response = await asyncio.to_thread(
            gemini_client.generate_content, # Use the synchronous method here
            prompt,                         # Pass arguments to the sync method
            generation_config=generation_config
        )
        app.logger.info(f"Gemini raw response received: {response}") # Log raw response

        # (Parsing logic remains the same)
        if response and response.parts:
             return {"success": True, "response": response.text.strip()}
        elif hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
             return {"success": False, "error": f"Blocked due to: {response.prompt_feedback.block_reason.name}"}
        else:
             try:
                 if hasattr(response, 'text') and response.text:
                    return {"success": True, "response": response.text.strip()}
                 else:
                    app.logger.warning(f"Gemini response has no text/parts and is not blocked: {response}")
                    return {"success": False, "error": "Gemini response format unexpected (empty or no text)."}
             except ValueError:
                 return {"success": False, "error": "Gemini response format not text (e.g., function call)."}
             except Exception as e:
                 app.logger.error(f"Error extracting Gemini text: {e}\nResponse: {response}", exc_info=True)
                 return {"success": False, "error": "Could not extract text from Gemini response."}

    # (Error handling remains the same)
    except GoogleAPIErrors.GoogleAPIError as e:
         return {"success": False, "error": f"Google API Error: {e}"}
    except GoogleAPIErrors.RetryError as e:
        return {"success": False, "error": f"Google API Retry Error: {e}"}
    except Exception as e:
        if isinstance(e, RuntimeError) and 'Event loop is closed' in str(e):
             app.logger.error(f"Gemini encountered 'Event loop closed' error even with to_thread: {e}", exc_info=True)
             return {"success": False, "error": "Gemini failed due to event loop issue."}
        app.logger.error(f"Google Gemini Unexpected Error: {e}", exc_info=True)
        return {"success": False, "error": f"Google Gemini Unexpected Error: {e}"}
# <<< === REWRITTEN GROK HANDLER FUNCTION (Using OpenAI Library) ===
async def get_grok_response(prompt: str) -> dict:
    """Gets response from xAI Grok API using the OpenAI library."""
    if not grok_openai_client: # Check the correctly named client
        return {"success": False, "error": "Grok client (via OpenAI library) not configured."}
    try:
        app.logger.info(f"Calling Grok (via OpenAI lib): model='{grok_model}', prompt='{prompt[:50]}...'")
        # Use the standard OpenAI chat completions method
        response = await asyncio.to_thread(
            grok_openai_client.chat.completions.create,
            model=grok_model, # Pass the Grok model name
            messages=[
                # Optional: Add a system prompt if desired for Grok
                # {"role": "system", "content": "You are Grok."},
                {"role": "user", "content": prompt}
                ],
            max_tokens=1500,  # These parameters should work if API is OpenAI compatible
            temperature=0.7
        )
        app.logger.info(f"Grok (via OpenAI lib) raw response received: {response}")

        # Use standard OpenAI response structure
        if response and response.choices and response.choices[0].message and response.choices[0].message.content:
            assistant_response = response.choices[0].message.content.strip()
            app.logger.info(f"Grok success. Response: '{assistant_response[:100]}...'")
            return {"success": True, "response": assistant_response}
        else:
            app.logger.warning(f"Unexpected/Invalid response structure received from Grok (via OpenAI lib): {response}")
            return {"success": False, "error": "Invalid response structure from Grok."}

    # Use the same error types as the standard OpenAI client
    except (APIError, APITimeoutError, RateLimitError) as e:
        # These errors might be raised by the client even when hitting the Grok URL
        app.logger.error(f"Grok API Error (via OpenAI lib): {type(e).__name__} - {e}", exc_info=True)
        return {"success": False, "error": f"Grok API Error: {type(e).__name__} - {e}"}
    except Exception as e:
        # Catch any other unexpected errors
        app.logger.error(f"Grok Unexpected Error (via OpenAI lib): {e}", exc_info=True)
        return {"success": False, "error": f"Grok Unexpected Error: {e}"}
# <<< === END OF REWRITTEN GROK HANDLER FUNCTION ===


async def get_ollama_response(prompt: str) -> dict:
    """Gets response from local Ollama API."""
    # (Keeping the Ollama handler structure in case you add it back)
    if not ollama_configured:
        return {"success": False, "error": "Ollama not configured (Base URL or Model missing)."}

    ollama_api_url = f"{ollama_base_url.rstrip('/')}/api/generate"
    payload = {"model": ollama_model,"prompt": prompt,"stream": False}
    try:
        async with httpx.AsyncClient(timeout=HTTPX_TIMEOUT) as client:
            response = await client.post(ollama_api_url, json=payload)
            response.raise_for_status()
            response_data = response.json()
            if response_data and "response" in response_data:
                return {"success": True, "response": response_data["response"].strip()}
            else:
                 err_detail = response_data.get('error', 'Unknown format')
                 return {"success": False, "error": f"Ollama response missing 'response' key. Detail: {err_detail}"}
    except httpx.HTTPStatusError as e:
        error_body = ""
        try: error_body = e.response.json().get('error', e.response.text)
        except Exception: error_body = e.response.text
        if "model not found" in error_body.lower():
             return {"success": False, "error": f"Ollama Error: Model '{ollama_model}' not found. Make sure it's pulled."}
        return {"success": False, "error": f"Ollama HTTP Error: {e.response.status_code} - {error_body}"}
    except httpx.RequestError as e:
        return {"success": False, "error": f"Ollama Connection Error: Could not connect to {ollama_api_url}. Is Ollama running? Details: {e}"}
    except Exception as e:
        app.logger.error(f"Ollama Unexpected Error: {e}", exc_info=True)
        return {"success": False, "error": f"Ollama Unexpected Error: {e}"}


# Mapping model IDs to their async handler functions
MODEL_HANDLERS = {
    "openai": get_openai_response,
    "azure": get_azure_openai_response,
    "claude": get_claude_response,
    "gemini": get_gemini_response,
    "grok": get_grok_response, # Maps to the rewritten Grok handler
    "ollama": get_ollama_response,
}

# --- Flask Routes ---
# ... (Flask routes remain the same) ...

@app.route('/', methods=['GET'])
def index():
    """Renders the main page."""
    # Filter available models based on successful client initialization
    truly_available_models = {
        model_id: name for model_id, name in MODEL_IDS.items()
        if not config_errors_detail.get(model_id) # Only include if no config error
        # Or, more explicitly check if corresponding client exists:
        # if (model_id == "openai" and openai_client) or \
        #    (model_id == "azure" and azure_client) or \
        #    (model_id == "claude" and anthropic_client) or \
        #    (model_id == "gemini" and gemini_client) or \
        #    (model_id == "grok" and grok_openai_client) or \
        #    (model_id == "ollama" and ollama_configured)
    }
    # Use session to remember, but default to *actually* available models
    selected_models = session.get('selected_models', list(truly_available_models.keys()))

    return render_template('index.html',
                           available_models=MODEL_IDS, # Pass all defined models for display logic
                           truly_available_models=truly_available_models, # Pass only usable models
                           selected_models=selected_models,
                           prompt_text=session.get('prompt_text', ''),
                           results=None,
                           config_errors=config_errors,
                           config_errors_detail=config_errors_detail)

@app.route('/compare', methods=['POST'])
async def compare():
    """Handles form submission, runs comparisons, and renders results."""
    start_time = time.time()
    prompt = request.form.get('prompt')
    selected_models = request.form.getlist('models') # Gets list of checked values

    session['prompt_text'] = prompt
    session['selected_models'] = selected_models

    results = {}
    duration = None

    if not prompt:
        flash("Please enter a prompt.", "error")
        # Redirect or render? Render is simpler here.
        truly_available_models = { mid: name for mid, name in MODEL_IDS.items() if not config_errors_detail.get(mid) }
        return render_template('index.html',
                                available_models=MODEL_IDS,
                                truly_available_models=truly_available_models,
                                selected_models=selected_models, prompt_text=prompt, results=None,
                                config_errors=config_errors, config_errors_detail=config_errors_detail)

    if not selected_models:
        flash("Please select at least one model.", "error")
        truly_available_models = { mid: name for mid, name in MODEL_IDS.items() if not config_errors_detail.get(mid) }
        return render_template('index.html',
                                available_models=MODEL_IDS,
                                truly_available_models=truly_available_models,
                                selected_models=selected_models, prompt_text=prompt, results=None,
                                config_errors=config_errors, config_errors_detail=config_errors_detail)

    # Create asyncio tasks for selected models
    tasks = []
    valid_selected_models = []
    for model_id in selected_models:
        handler = MODEL_HANDLERS.get(model_id)
        # Check configuration status before creating task
        if config_errors_detail.get(model_id):
             results[model_id] = {"success": False, "error": f"Configuration Error: {config_errors_detail[model_id]}"}
             app.logger.warning(f"Skipping {model_id} due to config error: {config_errors_detail[model_id]}")
        elif handler:
            tasks.append(asyncio.create_task(handler(prompt), name=model_id))
            valid_selected_models.append(model_id)
        else:
            # Should not happen if MODEL_HANDLERS is correct
            results[model_id] = {"success": False, "error": f"Internal Error: No handler defined for model '{model_id}'."}
            app.logger.error(f"No handler found for selected model: {model_id}")


    # Run tasks concurrently and gather results
    if tasks:
        gathered_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results, mapping back to model IDs
        for i, result in enumerate(gathered_results):
            model_id = valid_selected_models[i]
            if isinstance(result, Exception):
                 results[model_id] = {"success": False, "error": f"Internal Task Error: {result}"}
                 app.logger.error(f"Internal error during task execution for {model_id}: {result}", exc_info=True)
            elif isinstance(result, dict):
                results[model_id] = result
            else:
                 results[model_id] = {"success": False, "error": f"Internal Error: Unexpected result type '{type(result)}' from handler."}
                 app.logger.error(f"Unexpected result type from handler {model_id}: {type(result)} - {result}")

    duration = time.time() - start_time
    app.logger.info(f"Comparison for {len(selected_models)} models took {duration:.2f} seconds.")

    # Ensure results dict includes entries for all *selected* models (incl. skipped ones)
    final_results = {model_id: results.get(model_id) for model_id in selected_models}

    truly_available_models = { mid: name for mid, name in MODEL_IDS.items() if not config_errors_detail.get(mid) }
    return render_template('index.html',
                           available_models=MODEL_IDS,
                           truly_available_models=truly_available_models,
                           selected_models=selected_models,
                           prompt_text=prompt,
                           results=final_results,
                           duration=duration,
                           config_errors=config_errors,
                           config_errors_detail=config_errors_detail)

# --- Main Execution ---

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s') # Added funcName

    if not app.secret_key or app.secret_key == 'default-secret-key-for-dev':
        app.logger.warning("FLASK_SECRET_KEY is not set or is insecure in .env. Please set a strong secret key.")
    if config_errors:
        app.logger.warning("\n--- Configuration Errors Detected ---")
        for error in config_errors:
            app.logger.warning(f"- {error}")
        app.logger.warning("-----------------------------------\n")

    app.logger.info("--- Model Configuration Status ---")
    for mid, name in MODEL_IDS.items():
        status = "[ OK ]" if not config_errors_detail.get(mid) else "[FAIL]"
        details = "" if not config_errors_detail.get(mid) else f": {config_errors_detail[mid]}"
        app.logger.info(f"{status} {name} ({mid}){details}")
    app.logger.info("----------------------------------\n")

    app.run(debug=True, host='0.0.0.0', port=5001, use_reloader=True)