import json
import hashlib
from datetime import datetime
from pathlib import Path
from threading import Lock
import os

class TokenLedger:
    _lock = Lock()  
    def __init__(self, request_id: str, session_id: str, log_dir: str = "token_ledger/"):
        self.session_id = session_id
        self.request_id = request_id
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.file_path = self.log_dir / f"{self.session_id}.jsonl"
    
    def _make_event_id(self, entry: dict) -> str:
        raw = json.dumps(entry, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()

    def record(self, *, user_id, agent_id, model,
                prompt_tokens, completion_tokens,
                latency_ms=0, cache_hit=False,
                retry_count=0, tool_calls=0):

        total_tokens = prompt_tokens + completion_tokens
        cost = self._calculate_cost(model, prompt_tokens, completion_tokens)

        entry = {
            "request_id": self.request_id,
            "session_id": self.session_id,
            "user_id": user_id,
            "agent_id": agent_id,
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cost": cost,
            "timestamp": datetime.now().isoformat(),
            "latency_ms": latency_ms,
            "cache_hit": cache_hit,
            "retry_count": retry_count,
            "tool_calls": tool_calls,
        }

        entry["event_id"] = self._make_event_id(entry)
        if cost is None:
            entry["cost_error"]=True

        line = json.dumps(entry, separators=(",", ":")) + "\n"
        fd = os.open(self.file_path, os.O_APPEND | os.O_CREAT | os.O_WRONLY)
        try:
            with self._lock: 
                os.write(fd, line.encode())
        finally:
            os.close(fd)

    
    def _calculate_cost(self, model, prompt_tokens, completion_tokens):
        from litellm import completion_cost
        try:
            return completion_cost(
                model=model, 
                prompt_tokens=prompt_tokens, 
                completion_tokens=completion_tokens
            )
        except Exception:
            return None
