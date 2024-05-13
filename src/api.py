from openai import OpenAI
from src.const import *
from src.utils import *
import io


class ApiClient:
    def __init__(self):
        self.client = OpenAI()
        self.model = "gpt-4o"

    def create(self, messages, format="text", tokens=DEFAULT_TOKEN_LIMIT):
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=tokens,
                response_format={"type": format},
            )

            return completion.choices[0].message.content
        except Exception as error:
            print("Failed to generate: " + str(error), messages)
            return ""

    def _prepare_image(self, img):
        """
        cook any kind data into gpt-feedable image
        supports:
        - urls with link
        - base 64 images
        - plt charts (pass plt after drawing directly)
        - file data (io.BytesIO)
        """
        if isinstance(img, io.BytesIO):
            return buf_to_base64(img)
        if isinstance(img, str):
            return get_image_url(img)
        return plt_to_base64(img)

    def make_msg(self, text=None, img=None, role=ROLE_USER):
        # if it is text only message use simple format
        if img is None:
            return {"role": role, "content": text}

        # compose multimodal message otherwise
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

        return {
            "role": role,
            "content": content,
        }


ai_client = ApiClient()
