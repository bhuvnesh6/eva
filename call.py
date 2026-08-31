"""
Tata Tele Smartflo - Click to Call Support
Docs: https://docs.smartflo.tatatelebusiness.com/reference/v1click_to_call_support

How this endpoint behaves:
  1. It first calls `customer_number`.
  2. Once that call is picked up, it connects to the destination tied to
     your `api_key` (agent / extension / department).
  3. The response only confirms the call was *initiated* (async). Use
     webhooks if you want real-time status (ringing/answered/ended):
     https://docs.smartflo.tatatelebusiness.com/docs/webhook
"""

import requests

# ---- Fill these in ----
API_KEY = "[api key]a"   # never hardcode/commit real keys
TARGET_NUMBER = "918920419130"             # number to be called (customer_number)
DID_NUMBER = "918065251635"                # your DID, used as caller_id

ENDPOINT = "https://api-smartflo.tatateleservices.com/v1/click_to_call_support"


def make_call():
    payload = {
        "api_key": API_KEY,
        "customer_number": TARGET_NUMBER,
        "caller_id": DID_NUMBER,
        "async": 1,                # 1 = don't wait for answer before responding
        "customer_ring_timeout": 30,  # seconds (10-30)
        # "call_timeout": 120,     # optional: auto-hangup after N seconds
        # "custom_identifier": "order_12345",  # optional: returned in webhook
    }

    headers = {
        "accept": "application/json",
        "content-type": "application/json",
    }

    try:
        response = requests.post(ENDPOINT, json=payload, headers=headers, timeout=15)
        response.raise_for_status()
        data = response.json()
        print("Status Code:", response.status_code)
        print("Response:", data)
        return data
    except requests.exceptions.HTTPError as e:
        print("HTTP error:", e)
        print("Response body:", response.text)
    except requests.exceptions.RequestException as e:
        print("Request failed:", e)


if __name__ == "__main__":
    make_call()
