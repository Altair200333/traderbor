import re
import json


class RscObjectExtractor:
    @staticmethod
    def _extract_after_prefix(text):
        # This regex matches:
        #  - '^.*?:'    -> any characters from the start up to and including the first colon
        #  - '.?'       -> an optional single character immediately after the colon
        #  - '([[{"].*)' -> captures the rest of the string that starts with either '[', '{', or '"' followed by any characters
        pattern = r'^.*?:.*?([[{"].*)'
        match = re.match(pattern, text)
        if match:
            return match.group(1)
        return text

    @staticmethod
    def _extract_object_element(text):
        object_string = RscObjectExtractor._extract_after_prefix(text)
        if object_string.startswith("[") or object_string.startswith("{"):
            try:
                return json.loads(object_string)
            except Exception as e:
                return object_string
        return object_string

    @staticmethod
    def extract_objects(items):
        return [RscObjectExtractor._extract_object_element(item) for item in items]
