import os
import requests
from urllib.parse import quote


BASE_URL = "https://api.clashroyale.com/v1"


def _headers():
    token = os.getenv("CLASH_API_TOKEN")

    if not token:
        raise ValueError("Missing CLASH_API_TOKEN environment variable")

    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json"
    }


def encode_player_tag(player_tag: str) -> str:
    return quote(player_tag, safe="")


def get_player(player_tag: str) -> dict:
    encoded_tag = encode_player_tag(player_tag)
    url = f"{BASE_URL}/players/{encoded_tag}"

    response = requests.get(url, headers=_headers())

    if response.status_code != 200:
        raise Exception(f"API error {response.status_code}: {response.text}")

    return response.json()


def get_battlelog(player_tag: str) -> list:
    encoded_tag = encode_player_tag(player_tag)
    url = f"{BASE_URL}/players/{encoded_tag}/battlelog"

    response = requests.get(url, headers=_headers())

    if response.status_code != 200:
        raise Exception(f"API error {response.status_code}: {response.text}")

    return response.json()