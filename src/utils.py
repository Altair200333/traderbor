import re
import base64
import requests
import io


def is_url(val: str):
    return val.startswith("http://") or val.startswith("https://")


def is_base64(val: str):
    return re.search(r"^data:image\/(.*);base64", val) is not None


def fetch_content(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36"
    }

    response = requests.get(url, headers=headers)
    return response


def fetch_image_as_base64(url):
    """Function to convert image from URL to base64 string."""
    response = fetch_content(url)

    if response.status_code == 200:
        content_type = response.headers["Content-Type"]
        image_content = response.content
        base64_string = base64.b64encode(image_content).decode("utf-8")

        return f"data:{content_type};base64,{base64_string}"
    else:
        response.raise_for_status()


def get_image_url(img: str):
    if is_url(img):
        # in case of internet image fetch it externally and pass as base64
        return fetch_image_as_base64(img)
    if is_base64(img):
        return img
    raise Exception("Invalid image")


def buf_to_base64(buf):
    """Convert buffer (i.e. BytesIO) to base64 image string. (you better make sure it is an image)"""
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8").replace("\n", "")
    return "data:image/png;base64," + encoded


def plt_to_base64(plt, close=True):
    """After calling plt.plot(), scatter() ... etc, you can pass plt instance here and it will generate base64 image of your plot"""
    stream = io.BytesIO()
    plt.savefig(stream, format="png", bbox_inches="tight")

    if close == True:
        plt.close()

    return buf_to_base64(stream)
