from openai import OpenAI


class ApiClient:
    def __init__(self):
        self.client = OpenAI()
        self.model = "gpt-4-turbo"

    def create(self, messages):
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=4000,
            )

            return completion.choices[0].message.content
        except Exception as error:
            print("Failed to generate: " + str(error))
            return ""

    def make_msg(self, role, text=None, img=None):
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
