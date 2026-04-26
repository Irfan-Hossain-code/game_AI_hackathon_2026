from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

import requests
from requests import RequestException


@dataclass
class ActionPlan:
    should_play: bool
    card: str
    slot: int
    x: int
    y: int
    where: str
    why: str
    confidence: float


class OllamaBrain:
    """
    Single-model brain for both tactical decisions and spoken narration.
    """

    def __init__(
        self,
        model: str = "gemma2:2b",
        host: str = "http://127.0.0.1:11434",
        timeout_s: float = 8.0,
    ) -> None:
        self.model = model
        self.url = host.rstrip("/") + "/api/generate"
        self.timeout_s = timeout_s
        self.last_voice_command = ""
        self.tick = 0
        self.recent_actions: list[dict[str, Any]] = []
        self.role_prompt = (
            "You are ClashMind, an autonomous Clash Royale coach/player brain. "
            "Your role: pick safe, useful card plays using game state memory. "
            "You must be decisive, concise, and output strict machine-readable data when asked. "
            "Never output markdown."
        )

    def set_voice_command(self, text: str) -> None:
        self.last_voice_command = (text or "").strip()

    def _generate(self, system_prompt: str, user_prompt: str, max_tokens: int = 96) -> str:
        payload = {
            "model": self.model,
            "prompt": f"{system_prompt}\n\n{user_prompt}",
            "stream": False,
            "options": {
                "temperature": 0.15,
                "num_predict": max_tokens,
            },
        }
        t0 = time.perf_counter()
        try:
            resp = requests.post(self.url, json=payload, timeout=self.timeout_s)
            resp.raise_for_status()
            out = resp.json().get("response", "").strip()
            print(f"[LAT] llm={((time.perf_counter() - t0) * 1000):.1f}ms", flush=True)
            print(f"[BRAIN][RAW] {out}", flush=True)
            return out
        except RequestException as exc:
            print(f"[LLM] request failed: {exc}", flush=True)
            return ""

    def decide_action(self, state: dict[str, Any], extra_context: dict[str, Any] | None = None) -> ActionPlan:
        self.tick += 1
        extra_context = extra_context or {}
        system_prompt = (
            f"{self.role_prompt} Return strict JSON only. "
            "No markdown. No extra text."
        )
        user_prompt = (
            f"tick={self.tick}\n"
            f"state={json.dumps(state, ensure_ascii=True)}\n"
            f"extra_context={json.dumps(extra_context, ensure_ascii=True)}\n"
            f"voice_command={self.last_voice_command!r}\n"
            f"recent_actions={json.dumps(self.recent_actions[-5:], ensure_ascii=True)}\n"
            "Output JSON schema exactly:\n"
            '{"should_play":bool,"card":"str","slot":0,"x":640,"y":430,'
            '"where":"short","why":"short","confidence":0.0}\n'
            "Use the enemy memory from state. Keep where/why <= 8 words each."
        )
        raw = self._generate(system_prompt, user_prompt, max_tokens=120)
        try:
            obj = json.loads(raw)
        except Exception:
            obj = {
                "should_play": False,
                "card": "none",
                "slot": 0,
                "x": 640,
                "y": 430,
                "where": "center",
                "why": "waiting cycle",
                "confidence": 0.0,
            }

        plan = ActionPlan(
            should_play=bool(obj.get("should_play", False)),
            card=str(obj.get("card", "none")),
            slot=int(obj.get("slot", 0)),
            x=int(obj.get("x", 640)),
            y=int(obj.get("y", 430)),
            where=str(obj.get("where", "center")),
            why=str(obj.get("why", "pressure")),
            confidence=float(obj.get("confidence", 0.0)),
        )
        self.recent_actions.append(
            {
                "tick": self.tick,
                "card": plan.card,
                "slot": plan.slot,
                "x": plan.x,
                "y": plan.y,
                "where": plan.where,
                "why": plan.why,
                "confidence": plan.confidence,
                "should_play": plan.should_play,
            }
        )
        if len(self.recent_actions) > 24:
            self.recent_actions = self.recent_actions[-24:]
        return plan

    def narrate_play(self, plan: ActionPlan, state: dict[str, Any], extra_context: dict[str, Any] | None = None) -> str:
        extra_context = extra_context or {}
        system_prompt = (
            f"{self.role_prompt} Return one short spoken sentence for voice output. "
            "No markdown."
        )
        user_prompt = (
            f"Say what+where+why for this play: card={plan.card}, where={plan.where}, why={plan.why}. "
            f"enemy_log_available={state.get('enemy_log_available', True)} "
            f"enemy_cards_seen={state.get('enemy_cards_seen', [])} "
            f"teammate_context={state.get('teammate_context', '')}. "
            f"extra_context={json.dumps(extra_context, ensure_ascii=True)}."
        )
        line = self._generate(system_prompt, user_prompt, max_tokens=36)
        if not line:
            line = f"Placing {plan.card} {plan.where} to {plan.why}."

        if (not state.get("enemy_log_available", True)) and ("skeleton" in plan.card.lower()):
            line = "They used Log already, so Skeleton Army is safe now."
        return line

    def narrate_status(self, state: dict[str, Any], extra_context: dict[str, Any] | None = None) -> str:
        extra_context = extra_context or {}
        system_prompt = (
            f"{self.role_prompt} Return one short live coaching sentence (max 14 words). "
            "No markdown."
        )
        user_prompt = (
            f"Give a quick battle update from state: "
            f"enemy_cards_seen={state.get('enemy_cards_seen', [])}, "
            f"enemy_elixir_est={state.get('enemy_elixir_est', 5)}, "
            f"enemy_log_available={state.get('enemy_log_available', True)}, "
            f"enemy_lane={state.get('signals', {}).get('enemy_lane', 'unknown')}. "
            f"extra_context={json.dumps(extra_context, ensure_ascii=True)}."
        )
        line = self._generate(system_prompt, user_prompt, max_tokens=28)
        return line or "Holding for value. Watching their elixir and next commit."
