import os
import base64
import logging
import time
from pathlib import Path
from typing import Iterator, Optional
from collections import deque

import requests


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("code_ingestor")


EXTENSION_LANGUAGE_MAP: dict[str, str] = {
    ".py":    "python",
    ".js":    "javascript",
    ".ts":    "typescript",
    ".jsx":   "javascript",
    ".tsx":   "typescript",
    ".java":  "java",
    ".go":    "go",
    ".rb":    "ruby",
    ".rs":    "rust",
    ".cpp":   "cpp",
    ".cc":    "cpp",
    ".cxx":   "cpp",
    ".c":     "c",
    ".h":     "c",
    ".hpp":   "cpp",
    ".cs":    "csharp",
    ".php":   "php",
    ".swift": "swift",
    ".kt":    "kotlin",
    ".scala": "scala",
    ".sh":    "bash",
    ".bash":  "bash",
    ".zsh":   "bash",
    ".lua":   "lua",
    ".r":     "r",
    ".R":     "r",
    ".m":     "matlab",
    ".sql":   "sql",
    ".html":  "html",
    ".htm":   "html",
    ".css":   "css",
    ".yaml":  "yaml",
    ".yml":   "yaml",
    ".json":  "json",
    ".toml":  "toml",
    ".xml":   "xml",
    ".md":    "markdown",
}

SKIP_DIRS: set[str] = {
    ".git", ".github", "__pycache__", "node_modules",
    ".venv", "venv", "env", ".env", "dist", "build",
    ".idea", ".vscode", ".mypy_cache", ".pytest_cache",
}

MAX_FILE_SIZE_BYTES: int = 500_000

GITHUB_API_BASE = "https://api.github.com"

MAX_RETRIES = 5
RETRY_BACKOFF_BASE = 2

class CodeIngestor:
    def __init__(
        self,
        local_path: Optional[str] = None,
        github_repo: Optional[str] = None,
        github_token: Optional[str] = None,
        github_branch: Optional[str] = None,   # None = auto-detect, explicit value = pin
        max_file_size: int = MAX_FILE_SIZE_BYTES,
    ):
        if not local_path and not github_repo:
            raise ValueError("Provide either local_path or github_repo.")
        if local_path and github_repo:
            raise ValueError("Provide only one of local_path or github_repo, not both.")

        self.local_path = local_path
        self.github_repo = github_repo
        self.max_file_size = max_file_size

        self.github_token = github_token or os.getenv("GITHUB_TOKEN")
        if github_repo and not self.github_token:
            logger.warning(
                "No GitHub token provided. Unauthenticated requests are rate-limited "
                "to 60/hr. Set GITHUB_TOKEN env var or pass github_token=."
            )

        self._session = self._build_session() if github_repo else None

        # Resolve branch: explicit pin wins, otherwise ask the API
        if github_repo:
            if github_branch:
                self.github_branch = github_branch
                logger.info(f"Using pinned branch: {self.github_branch}")
            else:
                self.github_branch = self._get_default_branch(github_repo)
                logger.info(f"Auto-detected default branch: {self.github_branch}")
        else:
            self.github_branch = github_branch or "main"  # unused for local, but keeps the attribute defined

    def walk(self) -> Iterator[dict]:
        if self.local_path:
            yield from self._walk_local()
        else:
            yield from self._walk_github()

    def _walk_local(self) -> Iterator[dict]:
        root = Path(self.local_path).resolve()
        if not root.exists():
            raise FileNotFoundError(f"Local path not found: {root}")
        logger.info(f"Walking local repo: {root}")
        file_count = 0
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]

            for filename in filenames:
                full_path = Path(dirpath) / filename
                relative_path = str(full_path.relative_to(root))
                language = self._detect_language(filename)

                if language is None:
                    logger.debug(f"Skipping (unrecognized extension): {relative_path}")
                    continue

                size = full_path.stat().st_size
                if size == 0:
                    logger.debug(f"Skipping (empty file): {relative_path}")
                    continue
                if size > self.max_file_size:
                    logger.debug(f"Skipping (too large, {size}B): {relative_path}")
                    continue

                source_code = self._read_local_file(full_path)
                if source_code is None:
                    continue

                file_count += 1
                yield {
                    "source":      "local",
                    "repo":        str(root),
                    "path":        relative_path,
                    "language":    language,
                    "source_code": source_code,
                    "sha":         None,
                    "size":        size,
                }

        logger.info(f"Local walk complete. Files ingested: {file_count}")


    def _read_local_file(self, path: Path) -> Optional[str]:
        try:
            return path.read_text(encoding="utf-8", errors="strict")
        except UnicodeDecodeError:
            try:
                return path.read_text(encoding="latin-1")
            except Exception as e:
                logger.warning(f"Encoding error, skipping {path}: {e}")
                return None
        except Exception as e:
            logger.warning(f"Failed to read {path}: {e}")
            return None
        
    def _walk_github(self) -> Iterator[dict]:
        logger.info(f"Walking GitHub repo: {self.github_repo} (branch: {self.github_branch})")
        file_count = 0

        tree = self._fetch_git_tree()
        if not tree:
            logger.error("Failed to fetch repo tree. Aborting.")
            return

        blobs = [node for node in tree if node.get("type") == "blob"]
        logger.info(f"Total files in tree: {len(blobs)}")

        for node in blobs:
            path = node["path"]
            filename = Path(path).name
            language = self._detect_language(filename)

            if language is None:
                logger.debug(f"Skipping (unrecognized extension): {path}")
                continue

            size = node.get("size", 0)
            if size == 0:
                logger.debug(f"Skipping (empty): {path}")
                continue
            if size > self.max_file_size:
                logger.debug(f"Skipping (too large, {size}B): {path}")
                continue

            source_code = self._fetch_file_content(path, sha=node.get("sha"))
            if source_code is None:
                continue

            file_count += 1
            yield {
                "source":      "github",
                "repo":        self.github_repo,
                "path":        path,
                "language":    language,
                "source_code": source_code,
                "sha":         node.get("sha"),
                "size":        size,
            }

        logger.info(f"GitHub walk complete. Files ingested: {file_count}")

    def _fetch_git_tree(self) -> Optional[list[dict]]:
        url = (
            f"{GITHUB_API_BASE}/repos/{self.github_repo}"
            f"/git/trees/{self.github_branch}?recursive=1"
        )
        data = self._github_get(url)
        if data is None:
            return None

        if data.get("truncated"):
            logger.warning(
                "GitHub tree response was truncated (repo too large). "
                "Some files may be missed."
            )

        return data.get("tree", [])
    
    def _fetch_file_content(self, path: str, sha: Optional[str] = None) -> Optional[str]:
        raw_url = f"https://raw.githubusercontent.com/{self.github_repo}/{self.github_branch}/{path}"

        try:
            response = self._session.get(raw_url, timeout=15)
            if response.status_code == 200:
                return response.text
            logger.debug(f"Raw fetch failed ({response.status_code}) for {path}, trying blobs API")
        except requests.RequestException as e:
            logger.debug(f"Raw fetch error for {path}: {e}, trying blobs API")

        # Fallback: blobs API requires sha from the tree node
        if sha:
            return self._fetch_blob_by_sha(sha)

        logger.warning(f"No SHA available and raw fetch failed for {path}, skipping")
        return None

    def _build_session(self) -> requests.Session:
        session = requests.Session()
        if self.github_token:
            session.headers.update({"Authorization": f"token {self.github_token}"})
        session.headers.update({"Accept": "application/vnd.github.v3+json"})
        return session
    
    def _github_get(self, url: str) -> Optional[dict]:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = self._session.get(url, timeout=15)

                if response.status_code == 200:
                    return response.json()

                if response.status_code in (403, 429):
                    #
                    retry_after = int(response.headers.get("Retry-After", 0))
                    wait = retry_after if retry_after > 0 else RETRY_BACKOFF_BASE ** attempt
                    logger.warning(
                        f"Rate limited (HTTP {response.status_code}). "
                        f"Retrying in {wait}s (attempt {attempt}/{MAX_RETRIES})."
                    )
                    time.sleep(wait)
                    continue

                if response.status_code == 404:
                    logger.warning(f"Not found (404): {url}")
                    return None

                logger.error(f"GitHub API error {response.status_code} for {url}")
                return None

            except requests.RequestException as e:
                wait = RETRY_BACKOFF_BASE ** attempt
                logger.warning(f"Request failed ({e}). Retrying in {wait}s.")
                time.sleep(wait)

        logger.error(f"All {MAX_RETRIES} retries exhausted for {url}")
        return None
    
    def _get_default_branch(self, repo: str) -> str:
        url = f"{GITHUB_API_BASE}/repos/{repo}"
        data = self._github_get(url)
        if data is None:
            logger.warning("Could not fetch repo metadata, falling back to 'main'")
            return "main"
        branch = data.get("default_branch", "main")
        return branch
    
    @staticmethod
    def _detect_language(filename: str) -> Optional[str]:
        
        suffix = Path(filename).suffix.lower()
        return EXTENSION_LANGUAGE_MAP.get(suffix, None)
    
