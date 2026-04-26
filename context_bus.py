from __future__ import annotations

import re
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ContextSnapshot:
    ts: float
    focused_app: str = ""
    focused_activity: str = ""
    ui_text: list[str] = field(default_factory=list)
    ui_nodes: list[dict[str, Any]] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)


class ContextBus:
    """
    Lightweight ADB-side context collector.
    Uses dumpsys + uiautomator dump text to provide non-vision hints.
    """

    def __init__(self, refresh_interval_s: float = 1.5) -> None:
        self.refresh_interval_s = refresh_interval_s
        self._last_refresh = 0.0
        self._last_snapshot = ContextSnapshot(ts=time.time())

    def _run_adb(self, *args: str, timeout_s: float = 2.0) -> str:
        try:
            r = subprocess.run(
                ["adb", *args],
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
            if r.returncode != 0:
                return ""
            return (r.stdout or "").strip()
        except Exception:
            return ""

    def _focused_activity(self) -> tuple[str, str]:
        txt = self._run_adb("shell", "dumpsys", "window", "windows", timeout_s=2.0)
        if not txt:
            return "", ""
        # Examples include "... mCurrentFocus=Window{... com.supercell.clashroyale/... }"
        m = re.search(r"mCurrentFocus=.*?\s([A-Za-z0-9._]+)/(?:[A-Za-z0-9._$]+)", txt)
        app = m.group(1) if m else ""
        m2 = re.search(r"mCurrentFocus=.*?\s([A-Za-z0-9._]+/[A-Za-z0-9._$]+)", txt)
        activity = m2.group(1) if m2 else ""
        return app, activity

    def _ui_nodes(self) -> list[dict[str, Any]]:
        # Dump UI tree to /sdcard then cat; works on many emulator UIs.
        _ = self._run_adb("shell", "uiautomator", "dump", "/sdcard/view.xml", timeout_s=2.5)
        xml = self._run_adb("shell", "cat", "/sdcard/view.xml", timeout_s=2.0)
        if not xml:
            return []
        # Parse text + bounds.
        pattern = r'text="([^"]*)".*?resource-id="([^"]*)".*?bounds="\[(\d+),(\d+)\]\[(\d+),(\d+)\]"'
        nodes: list[dict[str, Any]] = []
        for m in re.finditer(pattern, xml):
            text = (m.group(1) or "").strip()
            rid = (m.group(2) or "").strip()
            x1, y1, x2, y2 = int(m.group(3)), int(m.group(4)), int(m.group(5)), int(m.group(6))
            if not text and not rid:
                continue
            nodes.append(
                {
                    "text": text,
                    "resource_id": rid,
                    "bounds": [x1, y1, x2, y2],
                    "center": [(x1 + x2) // 2, (y1 + y2) // 2],
                }
            )
        return nodes[:120]

    def snapshot(self, force: bool = False) -> ContextSnapshot:
        now = time.time()
        if not force and (now - self._last_refresh) < self.refresh_interval_s:
            return self._last_snapshot

        app, activity = self._focused_activity()
        ui_nodes = self._ui_nodes()
        ui_text = []
        seen = set()
        for n in ui_nodes:
            t = (n.get("text") or "").strip()
            if not t:
                continue
            tl = t.lower()
            if tl in seen:
                continue
            seen.add(tl)
            ui_text.append(t)
            if len(ui_text) >= 40:
                break
        snap = ContextSnapshot(
            ts=now,
            focused_app=app,
            focused_activity=activity,
            ui_text=ui_text,
            ui_nodes=ui_nodes,
            raw={
                "focused_app": app,
                "focused_activity": activity,
                "ui_text_count": len(ui_text),
                "ui_nodes_count": len(ui_nodes),
            },
        )
        self._last_snapshot = snap
        self._last_refresh = now
        return snap
