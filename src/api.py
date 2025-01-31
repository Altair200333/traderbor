import io
from src.const import *
from src.utils import *
from openai import OpenAI

class GenericApiClient:
    def __init__(self, client=None, model="o1"):
        # Initializes with default OpenAI client if none provided,
        # and uses "o1" as the default model.
        self.client = client if client is not None else OpenAI()
        self.model = model

    def create(self, messages, options=None):
        """
        Create a chat completion using the OpenAI client.

        Parameters:
            messages (list): The input messages for the chat model.
            options (dict, optional): Optional parameters for the request.
                - format (str): The response format. Default is "text".
                - tokens (int): The maximum token limit. Default is DEFAULT_TOKEN_LIMIT.

        Returns:
            str: The generated content from the chat completion, or an empty string if an error occurs.
        """
        opts = options or {}
        response_format = opts.get("format", TEXT_MODE)
        tokens = opts.get("tokens", DEFAULT_TOKEN_LIMIT)
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_completion_tokens=tokens,
                response_format={"type": response_format},
            )
            return completion.choices[0].message.content
        except Exception as error:
            print("Failed to generate: " + str(error), messages)
            return ""

    def _prepare_image(self, img):
        """
        Convert various image types to API-compatible format.
        Supports:
          - io.BytesIO (converted to base64)
          - Image URLs or Base64 strings
          - plt charts (pass plt after drawing directly)
        """
        if isinstance(img, io.BytesIO):
            return buf_to_base64(img)
        if isinstance(img, str):
            return get_image_url(img)
        return plt_to_base64(img)

    def make_msg(self, text=None, img=None, role=ROLE_USER):
        # For text-only messages, use a simple format.
        if img is None:
            return {"role": role, "content": text}

        # Compose a multimodal message otherwise.
        content = []
        if text is not None:
            content.append({"type": "text", "text": text})
        if img is not None:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": self._prepare_image(img),
                        "detail": "high",
                    },
                }
            )

        return {"role": role, "content": content}


