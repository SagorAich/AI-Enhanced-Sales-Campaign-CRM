import os
from typing import Optional

# Try importing Groq SDK. The package should be installed via requirements.txt.
try:
    from groq import Groq
except Exception as e:
    Groq = None  # type: ignore


class LLMClient:
    """Minimal LLM client using Groq's API. Falls back to raising helpful errors if Groq isn't available.

    Usage:
        llm = LLMClient(api_key="...")
        text = llm.generate(prompt)
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "openai/gpt-oss-20b"):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.model = model
        if Groq is None:
            raise ImportError(
                "groq package is not installed. Please install requirements.txt which includes 'groq'."
            )

        # If an api_key was provided, set it in environment so the Groq client can pick it up if needed.
        if self.api_key:
            # set/override the env var before creating the client
            os.environ["GROQ_API_KEY"] = self.api_key

        # Create the Groq client. The SDK reads the GROQ_API_KEY env var.
        try:
            # Prefer calling Groq() without attempting to pass api_key (some SDKs require env var)
            self.client = Groq()
        except Exception as e:
            raise RuntimeError(
                "Failed to initialize Groq client. Make sure the GROQ_API_KEY environment variable is set or pass the API key to LLMClient(api_key=...). Original error: " + str(e)
            )

    def generate(self, prompt: str, max_new_tokens: int = 128, timeout: int = 60) -> str:
        """Call Groq chat completions API in non-streaming mode and return text result.

        This tries to be resilient to slight differences in the SDK return shape.
        """
        messages = [{"role": "user", "content": prompt}]
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.2,
                max_completion_tokens=max_new_tokens,
                top_p=1,
                reasoning_effort="medium",
                stream=False,
                stop=None,
            )
        except TypeError:
            # Some older/newer SDKs may use slightly different parameter names.
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.2,
                max_tokens=max_new_tokens,
                top_p=1,
                stream=False,
            )

        # Extract text from response (supports object or dict-like responses)
        try:
            # SDK object with choices -> message -> content
            if hasattr(resp, "choices") and len(resp.choices) > 0:
                choice = resp.choices[0]
                # try message.content
                msg = getattr(choice, "message", None)
                if msg is not None:
                    content = getattr(msg, "content", None)
                    if content:
                        return content.strip()
                # try choice.text
                text = getattr(choice, "text", None)
                if text:
                    return text.strip()

            # dict-like handling
            if isinstance(resp, dict):
                choices = resp.get("choices") or []
                if len(choices) > 0:
                    first = choices[0]
                    if isinstance(first, dict):
                        # common Groq shape: {'message': {'content': '...'}}
                        msg = first.get("message") or {}
                        if isinstance(msg, dict) and msg.get("content"):
                            return msg.get("content").strip()
                        if first.get("text"):
                            return first.get("text").strip()

            # Fallback: stringify
            return str(resp)
        except Exception as e:
            return str(resp)
