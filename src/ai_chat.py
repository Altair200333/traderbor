from src.api import *
from src.const import ROLE_USER, ROLE_ASSISTANT


class AiChat:
    """
    A wrapper that manages conversation history for an LLM, using the same
    message interface as the BasicApiClient.
    """

    def __init__(self, client=None):
        # Use the provided client or fallback to BasicApiClient
        self.client = client or BasicApiClient()
        self.history = []

    def message(self, text=None, img=None, role=ROLE_USER, format=TEXT_MODE):
        """
        Send message to the LLM

        :param text: The text content for the message
        :param img: The optional image or BytesIO or plot object
        :param role: The role of the message
        :return: response text.
        """
        new_msg = self.client.make_msg(text=text, img=img, role=role)
        self.history.append(new_msg)

        # If this is from the user, we get an openai response
        if role == ROLE_USER:
            assistant_text = self.client.create(
                self.history, options={"format": format})
            self.history.append({
                "role": ROLE_ASSISTANT,
                "content": assistant_text
            })
            return assistant_text
        return None

    def clear_history(self):
        """Clears the stored conversation history."""
        self.history = []

    def get_history(self):
        """
        Returns the entire conversation history in the same format passed
        into the BasicApiClient.
        """
        return self.history
