from openai import OpenAI
from const import *


class ApiClient:
    def __init__(self):
        self.client = OpenAI()
        self.model = "gpt-4-turbo"

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
            print("Failed to generate: " + str(error))
            return ""

    def make_msg(
        self,
        role,
        text=None,
        img=None,
    ):
        if img is None:
            return {"role": role, "content": text}

        content = []
        if text is not None:
            content.append({"type": "text", "text": text})
        if img is not None:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": img,
                        "detail": "high",
                    },
                }
            )

        return {
            "role": role,
            "content": content,
        }
