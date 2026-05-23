import json
import os
import hashlib
from datetime import datetime
from pathlib import Path
from threading import Lock

class TraceLogger:
    _lock = Lock()  

    def __init__(self, session_id: str, log_dir: str = "traces/"):
        self.session_id = session_id
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.file_path = self.log_dir / f"{self.session_id}.jsonl"

    def _make_event_id(self, entry: dict) -> str:
        raw = json.dumps(entry, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()

    def log(self, tool: str, intent: str, inputs: dict,
            outputs: dict, confidence: float, usage_id: int = 0):

        entry = {
            "schema_version": 1,
            "ts": datetime.utcnow().isoformat(),
            "session": self.session_id,
            "tool": tool,
            "intent": intent,
            "inputs": inputs,
            "outputs": outputs,
            "confidence": confidence,
            "usage_ref": usage_id
        }

        entry["event_id"] = self._make_event_id(entry)
        line = json.dumps(entry, separators=(",", ":")) + "\n"


        fd = os.open(self.file_path, os.O_APPEND | os.O_CREAT | os.O_WRONLY)
        try:
            with self._lock: 
                os.write(fd, line.encode())
        finally:
            os.close(fd)


    

    def stream(self):
        if not self.file_path.exists():
            return
    
        with open(self.file_path, "r") as f:
            for line in f:
                try:
                    yield json.loads(line)
                except:
                    continue