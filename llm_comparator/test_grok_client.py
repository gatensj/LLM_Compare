# test_grok_client.py
import os
import logging
from dotenv import load_dotenv
from xai_sdk.v1 import Client # Assuming this import is correct now

logging.basicConfig(level=logging.INFO)
load_dotenv()

grok_api_key = os.getenv("XAI_API_KEY")

if grok_api_key:
    try:
        print("Attempting to initialize xai_sdk.v1 Client...")
        grok_client = Client(api_key=grok_api_key)
        print("Client initialized successfully.")

        print("\n--- Attributes/Methods available on grok_client (dir(grok_client)) ---")
        # Filter out private/dunder methods for clarity
        public_attrs = [attr for attr in dir(grok_client) if not attr.startswith('_')]
        print(public_attrs)

        # If 'chat' exists, inspect it too, although it was problematic before
        if 'chat' in public_attrs:
             try:
                 print("\n--- Attributes/Methods available on grok_client.chat (dir(grok_client.chat)) ---")
                 chat_attrs = [attr for attr in dir(grok_client.chat) if not attr.startswith('_')]
                 print(chat_attrs)
             except Exception as e:
                 print(f"Could not inspect grok_client.chat: {e}")

        # If 'chat_completions' exists, inspect it
        if 'chat_completions' in public_attrs:
             try:
                 print("\n--- Attributes/Methods available on grok_client.chat_completions (dir(grok_client.chat_completions)) ---")
                 cc_attrs = [attr for attr in dir(grok_client.chat_completions) if not attr.startswith('_')]
                 print(cc_attrs)
             except Exception as e:
                 print(f"Could not inspect grok_client.chat_completions: {e}")


    except Exception as e:
        logging.error(f"Failed to initialize or inspect Grok client: {e}", exc_info=True)
else:
    print("Grok API Key (XAI_API_KEY) not found in .env file.")