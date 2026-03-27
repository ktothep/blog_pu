import io

import docx
import requests
from bs4 import BeautifulSoup
import PyPDF2


def visit_link_scrap(url: str) -> str:
    '''
    This tool is used to scrape information from the webpage and present it to the user
    It takes a url as input parameter
    :param url: The URL that has to be scraped
    :return: The scraped text from webpage
    '''
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/120.0.0.0 Safari/537.36"
    }

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")

        # Remove unwanted tags (scripts, styles, metadata)
        for tag in soup(["script", "style", "meta", "noscript", "head"]):
            tag.decompose()

        # Extract plain text only
        text = soup.get_text(separator="\n", strip=True)

        # Remove blank lines
        lines = [line for line in text.splitlines() if line.strip()]
        clean_text = "\n".join(lines)

        return (clean_text)

    else:
        print(f"Failed. Status code: {response.status_code}")


def read_local_file(file_path: str) -> str:
    """Reads and returns the text content of a local file.

    Args:
        file_path: The absolute or relative path to the file.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"


def extract_text_from_file(file_bytes: bytes, filename: str) -> str:
    """Extracts text from PDF, DOCX, or TXT files in memory."""
    ext = filename.lower().split('.')[-1]

    try:
        if ext == 'pdf':
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text

        elif ext == 'docx':
            doc = docx.Document(io.BytesIO(file_bytes))
            return "\n".join([para.text for para in doc.paragraphs])

        elif ext == 'txt':
            return file_bytes.decode('utf-8')

        else:
            raise ValueError(f"Unsupported file format: .{ext}")
    except Exception as e:
        raise ValueError(f"Failed to parse file: {str(e)}")