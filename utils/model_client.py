"""Direct Qwen API client; optional legacy Flask backend. No local model imports."""
import base64
import hashlib
import json
import logging
import os
import re
import struct
import threading
import time
from collections import OrderedDict
from functools import lru_cache
from io import BytesIO
from pathlib import Path

import requests
from dotenv import load_dotenv

LOG = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]
_IMAGE_DATA_RE = re.compile(r'data:image/[^;\s]+;base64,[A-Za-z0-9+/=]+')
_PNG_SIGNATURE = b'\x89PNG\r\n\x1a\n'
_CLIENTS = []
_CLIENTS_LOCK = threading.Lock()
_TIMING_FIELDS = ("requests", "cache_hits", "seconds", "text_seconds", "vision_seconds")


def load_environment():
    load_dotenv(os.environ.get("PSGNAV_ENV_FILE", ROOT / ".env"), override=False)


class ModelRequestError(RuntimeError):
    """Infrastructure failures must not be interpreted as navigation answers."""

    def __init__(self, message, *, retry_exhausted=False):
        super().__init__(message)
        self.retry_exhausted = bool(retry_exhausted)


class ModelOutputTruncatedError(ModelRequestError):
    """The caller may reduce its batch rather than accepting partial evidence."""


class ModelClient:
    def __init__(self, server_port=None):
        load_environment()
        self.backend = os.environ.get("PSGNAV_MODEL_BACKEND", "openai").lower()
        if self.backend not in {"openai", "openrouter", "siliconflow", "local"}:
            raise ValueError("PSGNAV_MODEL_BACKEND must be openai, openrouter, siliconflow, or local")
        self.referer = ""
        self.title = ""
        if self.backend == "openrouter":
            self.model = os.environ.get("QWEN_MODEL_NAME", "google/gemini-2.5-flash")
            self.api_key = os.environ.get("OPENROUTER_API_KEY", "")
            base = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
            self.url = base.rstrip("/") + "/chat/completions"
            self.referer = os.environ.get("OPENROUTER_HTTP_REFERER", "https://localhost")
            self.title = os.environ.get("OPENROUTER_TITLE", "psg-nav-instance")
            if not self.api_key:
                raise ValueError("Set OPENROUTER_API_KEY in PSGNAV_ENV_FILE or .env")
        elif self.backend == "siliconflow":
            self.model = os.environ.get("QWEN_MODEL_NAME", "Qwen/Qwen3-VL-8B-Instruct")
            self.api_key = os.environ.get("SILICONFLOW_API_KEY", "")
            base = os.environ.get("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")
            self.url = base.rstrip("/") + "/chat/completions"
            if not self.api_key:
                raise ValueError("Set SILICONFLOW_API_KEY in PSGNAV_ENV_FILE or .env")
        else:
            self.model = os.environ.get("QWEN_MODEL_NAME", "qwen3-vl-flash")
            self.api_key = os.environ.get("DASHSCOPE_API_KEY", "")
            base = os.environ.get("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
            self.url = base.rstrip("/") + "/chat/completions"
            if self.backend == "openai" and not self.api_key:
                raise ValueError("Set DASHSCOPE_API_KEY in PSGNAV_ENV_FILE or .env")
        if self.backend == "local":
            if server_port is None:
                gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
                server_port = 5001 + (int(gpu) if gpu.isdigit() else 0)
            self.url = f"http://127.0.0.1:{server_port}/chat"
        self.timeout = float(os.environ.get("PSGNAV_API_TIMEOUT", "45"))
        self.retries = int(os.environ.get("PSGNAV_API_RETRIES", "2"))
        self.concurrency = int(os.environ.get("PSGNAV_API_CONCURRENCY", "4"))
        self.cache_size = int(os.environ.get("PSGNAV_API_CACHE_SIZE", "512"))
        self.max_tokens = int(os.environ.get("PSGNAV_API_MAX_TOKENS", "1024"))
        if self.timeout <= 0 or self.retries < 0 or self.concurrency < 1 or self.cache_size < 0 or self.max_tokens < 1:
            raise ValueError("Invalid API timeout/retry/concurrency/cache/token configuration")
        self._local = threading.local()
        self._cache = OrderedDict()
        self._image_cache = OrderedDict()
        self._cache_lock = threading.Lock()
        # The shared cap stays PSGNAV_API_CONCURRENCY. Landmark ranking may ask
        # for more workers; raise only the in-flight limit so those requests
        # are not silently queued behind the smaller cap.
        from configs.navigation_config import landmark_comparison_workers
        comparison_workers = landmark_comparison_workers()
        slot_count = self.concurrency if comparison_workers is None else max(self.concurrency, comparison_workers)
        self._slots = threading.BoundedSemaphore(slot_count)
        self._key_locks = [threading.Lock() for _ in range(64)]
        # seconds is real time with at least one network call in flight, not the
        # sum of overlapping calls. text_seconds and vision_seconds split bursts
        # that are entirely one kind.
        self.stats = {
            "requests": 0, "cache_hits": 0, "seconds": 0.0,
            "text_seconds": 0.0, "vision_seconds": 0.0,
        }
        self._inflight = 0
        self._wall_origin = None
        self._burst_network = False
        self._burst_text = 0
        self._burst_vision = 0
        with _CLIENTS_LOCK:
            _CLIENTS.append(self)

    def _session(self):
        if not hasattr(self._local, "session"):
            self._local.session = requests.Session()
        return self._local.session

    def _begin_call(self):
        with self._cache_lock:
            if self._inflight == 0:
                self._wall_origin = time.monotonic()
                self._burst_network = False
                self._burst_text = 0
                self._burst_vision = 0
            self._inflight += 1

    def _finish_call(self, network, vision):
        """Close one complete() call. Overlapping calls share one real-time interval."""
        with self._cache_lock:
            if network:
                self._burst_network = True
                self.stats["requests"] += 1
                if vision:
                    self._burst_vision += 1
                else:
                    self._burst_text += 1
            self._inflight -= 1
            if self._inflight == 0 and self._burst_network:
                elapsed = time.monotonic() - self._wall_origin
                self.stats["seconds"] += elapsed
                if self._burst_vision and not self._burst_text:
                    self.stats["vision_seconds"] += elapsed
                elif self._burst_text and not self._burst_vision:
                    self.stats["text_seconds"] += elapsed
            if self._inflight == 0:
                self._burst_network = False
                self._burst_text = 0
                self._burst_vision = 0

    def complete(self, prompt, image=None, *, images=None, response_format=None,
                 validator=None, max_tokens=None, retries=None):
        self._local.request_metadata = {}
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("A nonempty prompt is required")
        if image is not None and images is not None:
            raise ValueError('Pass image or images, not both')
        if images is not None:
            if not isinstance(images, (list, tuple)) or not images:
                raise ValueError('images must be a nonempty list or tuple')
            if self.backend != 'openai':
                raise ValueError('Multi-image requests require the openai-compatible backend')
        if retries is not None and (type(retries) is not int or retries < 0):
            raise ValueError('retries must be a non-negative integer')
        if max_tokens is not None and (type(max_tokens) is not int or max_tokens < 1):
            raise ValueError('max_tokens must be a positive integer')
        vision = image is not None or images is not None
        self._begin_call()
        network = False

        def mark_network():
            nonlocal network
            network = True

        try:
            return self._complete_uncached(prompt, image, images=images,
                                           response_format=response_format, validator=validator,
                                           mark_network=mark_network, max_tokens=max_tokens, retries=retries)
        finally:
            self._finish_call(network, vision)

    def last_request_metadata(self):
        """Metadata for this thread's latest request; contains no credentials or payload."""
        return dict(getattr(self._local, 'request_metadata', {}))

    def _encode_image(self, image):
        rgb = image.convert('RGB')
        image_key = (rgb.size, hashlib.sha256(rgb.tobytes()).digest())
        with self._cache_lock:
            encoded = self._image_cache.get(image_key)
            if encoded is not None:
                self._image_cache.move_to_end(image_key)
                return encoded
        with BytesIO() as buffer:
            rgb.save(buffer, format="PNG")
            encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        with self._cache_lock:
            self._image_cache[image_key] = encoded
            while len(self._image_cache) > 32:
                self._image_cache.popitem(last=False)
        return encoded

    def _complete_uncached(self, prompt, image, *, images, response_format, validator,
                           mark_network, max_tokens, retries):
        content = prompt
        image_b64 = None
        if image is not None:
            image_b64 = self._encode_image(image)
            content = [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + image_b64}},
                {"type": "text", "text": prompt},
            ]
        elif images is not None:
            # Keep the full ordered image set in one request and in its cache key.
            content = []
            for index, frame in enumerate(images):
                content.extend([
                    {"type": "text", "text": f"Image {index}:"},
                    {"type": "image_url", "image_url": {
                        "url": "data:image/png;base64," + self._encode_image(frame)}},
                ])
            content.append({"type": "text", "text": prompt})
        if self.backend == "local":
            payload = {"prompt": prompt}
            if image_b64:
                payload["image"] = image_b64
        else:
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": content}],
                "temperature": 0.0,
                "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
                "stream": False,
            }
            # enable_thinking is DashScope-specific. OpenRouter and SiliconFlow requests omit it.
            if self.backend == "openai":
                payload["enable_thinking"] = False
            if response_format is not None:
                payload['response_format'] = response_format
        key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        self._local.request_metadata = {'cache_key': key, 'cache_hit': False}
        with self._key_locks[int(key[:8], 16) % len(self._key_locks)]:
            with self._cache_lock:
                if key in self._cache:
                    self._local.request_metadata['cache_hit'] = True
                    if validator is not None:
                        try:
                            validator(self._cache[key])
                        except (ValueError, TypeError, KeyError, AttributeError):
                            del self._cache[key]
                            raise
                    self.stats["cache_hits"] += 1
                    self._cache.move_to_end(key)
                    return self._cache[key]
            with self._slots:
                mark_network()
                started = time.monotonic()
                try:
                    answer = self._send(payload) if retries is None else self._send(payload, retries=retries)
                finally:
                    elapsed = time.monotonic() - started
            # A well-formed HTTP reply can still violate the caller's output
            # contract. Validate before caching it as reusable evidence.
            if validator is not None:
                validator(answer)
            with self._cache_lock:
                if self.cache_size:
                    self._cache[key] = answer
                    while len(self._cache) > self.cache_size:
                        self._cache.popitem(last=False)
            LOG.info("model=%s vision=%s latency=%.2fs", self.model,
                     image is not None or images is not None, elapsed)
            return answer

    def _redact(self, text):
        if self.api_key:
            text = text.replace(self.api_key, '[redacted]')
        return _IMAGE_DATA_RE.sub('[image]', text)

    def _redact_value(self, value):
        if isinstance(value, str):
            return self._redact(value)
        if isinstance(value, dict):
            return {str(key): self._redact_value(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._redact_value(item) for item in value]
        return value

    def _provider_reason(self, body):
        """OpenAI puts code/message under error; SiliconFlow puts them at the top level."""
        if not isinstance(body, dict):
            return ''
        sources = []
        detail = body.get('error')
        if isinstance(detail, dict):
            sources.append(detail)
        sources.append(body)
        for source in sources:
            pieces = [source.get(key) for key in ('code', 'message')]
            reason = ' '.join(str(piece) for piece in pieces if piece not in (None, '')).strip()
            if reason:
                return self._redact(reason)
        return ''

    def _json_decode_error(self, response):
        """SiliconFlow sometimes returns HTTP 400 for a request it failed to parse."""
        if response.status_code != 400:
            return False
        try:
            reason = self._provider_reason(response.json())
        except (ValueError, AttributeError, TypeError):
            return False
        return 'json decode error' in reason.lower()

    def _response_error(self, response):
        """Keep useful provider reasons without dumping bodies or request images."""
        message = f"Model API HTTP {response.status_code}"
        try:
            reason = self._provider_reason(response.json())
        except (ValueError, AttributeError, TypeError):
            reason = ''
        if reason:
            message += ': ' + reason[:500]
        return message

    def _describe_image(self, encoded):
        """Record image identity and size. The base64 payload is not written to the log."""
        if not isinstance(encoded, str) or not encoded:
            return {'present': False}
        try:
            raw = base64.b64decode(encoded, validate=False)
        except (ValueError, TypeError):
            return {'present': True, 'decoded': False, 'encoded_chars': len(encoded)}
        facts = {
            'encoded_chars': len(encoded),
            'bytes': len(raw),
            'sha256': hashlib.sha256(raw).hexdigest(),
        }
        if raw.startswith(_PNG_SIGNATURE) and len(raw) >= 24:
            facts['width'], facts['height'] = struct.unpack('>II', raw[16:24])
            facts['format'] = 'PNG'
        return facts

    def _describe_content(self, content):
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            return self._redact_value(content)
        described = []
        for part in content:
            if not isinstance(part, dict):
                described.append(self._redact_value(part))
                continue
            if part.get('type') == 'image_url':
                url = (part.get('image_url') or {}).get('url', '')
                encoded = url.split('base64,', 1)[1] if isinstance(url, str) and 'base64,' in url else ''
                described.append({'type': 'image_url', 'image': self._describe_image(encoded)})
            else:
                described.append(self._redact_value(part))
        return described

    def _describe_request(self, payload):
        if not isinstance(payload, dict):
            return self._redact_value(payload)
        described = {}
        for key, value in payload.items():
            if key == 'messages' and isinstance(value, list):
                described[key] = [
                    {
                        'role': message.get('role') if isinstance(message, dict) else None,
                        'content': self._describe_content(
                            message.get('content') if isinstance(message, dict) else message),
                    }
                    for message in value
                ]
            elif key == 'image' and isinstance(value, str):
                described[key] = self._describe_image(value)
            else:
                described[key] = self._redact_value(value)
        return described

    def _response_body(self, response):
        try:
            text = response.text
        except Exception as exc:
            return {'unreadable': type(exc).__name__}
        text = self._redact(text)
        try:
            return json.loads(text)
        except ValueError:
            return text

    def _log_http_400(self, payload, response, attempt):
        """One stdout record. Evaluation logs capture stdout, not the logging module."""
        try:
            headers = {str(key): self._redact(str(value)) for key, value in response.headers.items()
                       if str(key).lower() not in {'set-cookie', 'authorization'}}
        except Exception:
            headers = {}
        record = {
            'backend': self.backend,
            'url': self.url,
            'model': self.model,
            'attempt': attempt,
            'request': self._describe_request(payload),
            'response': {
                'status': getattr(response, 'status_code', None),
                'headers': headers,
                'body': self._response_body(response),
            },
        }
        try:
            text = json.dumps(record, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            text = repr(record)
        print('[API HTTP 400] ' + text, flush=True)

    def _send(self, payload, *, retries=None):
        retries = self.retries if retries is None else retries
        headers = {"Content-Type": "application/json"}
        if self.backend in {"openai", "openrouter", "siliconflow"}:
            headers["Authorization"] = "Bearer " + self.api_key
        if self.backend == "openrouter":
            headers["HTTP-Referer"] = self.referer
            headers["X-OpenRouter-Title"] = self.title
        for attempt in range(retries + 1):
            delay = min(2 ** attempt, 5)
            try:
                with self._session().post(self.url, headers=headers, json=payload,
                                          timeout=(10, self.timeout)) as response:
                    transient_400 = self._json_decode_error(response)
                    if response.status_code in {408, 429, 500, 502, 503, 504} or transient_400:
                        if transient_400:
                            self._log_http_400(payload, response, attempt + 1)
                        error = self._response_error(response)
                        try:
                            delay = min(5, max(delay, float(response.headers.get("Retry-After", 0))))
                        except ValueError:
                            pass
                    elif not response.ok:
                        if response.status_code == 400:
                            self._log_http_400(payload, response, attempt + 1)
                        raise ModelRequestError(self._response_error(response) + '; check endpoint, key and model')
                    else:
                        try:
                            data = response.json()
                            if self.backend == "local":
                                answer = data["response"]
                            else:
                                choice = data["choices"][0]
                                if choice.get("finish_reason") not in (None, "stop"):
                                    error_type = (ModelOutputTruncatedError
                                                  if choice.get("finish_reason") == "length"
                                                  else ModelRequestError)
                                    raise error_type(
                                        f"Model response incomplete: finish_reason={choice.get('finish_reason')}, "
                                        f"completion_tokens={data.get('usage', {}).get('completion_tokens')}; "
                                        "check max tokens or provider filtering")
                                answer = choice["message"]["content"]
                            if not isinstance(answer, str) or not answer.strip():
                                raise ValueError("empty answer")
                            return answer.strip()
                        except (ValueError, KeyError, IndexError, TypeError):
                            raise ModelRequestError("Model API returned an invalid or empty response") from None
            except (requests.Timeout, requests.ConnectionError):
                error = "Model API connection or timeout failure"
            except requests.RequestException as exc:
                error = f"Model API transport failure: {type(exc).__name__}"
            if attempt == retries:
                raise ModelRequestError(
                    f"{error} after {attempt + 1} attempts", retry_exhausted=True) from None
            time.sleep(delay)


def snapshot_api_timing():
    """Sum real-time counters across every client created in this process."""
    totals = {key: 0 for key in _TIMING_FIELDS}
    with _CLIENTS_LOCK:
        clients = list(_CLIENTS)
    for client in clients:
        with client._cache_lock:
            for key in _TIMING_FIELDS:
                totals[key] += client.stats[key]
    return totals


def diff_api_timing(start, end):
    return {key: end[key] - start.get(key, 0) for key in _TIMING_FIELDS}


def format_api_timing(delta, *, gpu=None, dataset_index, episode_id):
    """One log line. seconds is real time, not the sum of overlapping calls."""
    fields = []
    if gpu is not None:
        fields.append(f"gpu={gpu}")
    fields.extend((
        f"dataset_index={dataset_index}",
        f"episode_id={episode_id}",
        f"requests={int(delta['requests'])}",
        f"cache_hits={int(delta['cache_hits'])}",
        f"seconds={float(delta['seconds']):.3f}",
        f"text_seconds={float(delta['text_seconds']):.3f}",
        f"vision_seconds={float(delta['vision_seconds']):.3f}",
    ))
    return "[API Timing] " + " ".join(fields)


@lru_cache(maxsize=16)
def get_model_client(server_port=None):
    return ModelClient(server_port)
