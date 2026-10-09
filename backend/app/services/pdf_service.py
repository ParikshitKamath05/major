import base64
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional
import fitz  # PyMuPDF
from fastapi import HTTPException

from app.core.config import GROQ_VISION_MODEL
from app.core.database import groq_client


def validate_pdf_file(file_bytes: bytes, filename: str) -> None:
    """
    Validate that the uploaded file is a genuine PDF by checking magic bytes.
    """
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    # Check PDF magic bytes: %PDF- (0x25 0x50 0x44 0x46 0x2D)
    if not file_bytes.startswith(b"%PDF-"):
        raise HTTPException(
            status_code=400,
            detail="Invalid file format: File header does not match PDF specification",
        )


def extract_text_from_pdf_path(file_path: str) -> str:
    """Extract text from a PDF file on disk."""
    try:
        doc = fitz.open(file_path)
        text = ""

        for page in doc:
            page_text = page.get_text()
            if page_text.strip():
                text += page_text + "\n"

        doc.close()
        return text.strip()

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read PDF: {str(e)}")


def extract_images_from_page(page: fitz.Page, page_num: int) -> List[dict]:
    """Extract images from a PDF page as base64-encoded payloads."""
    images: List[dict] = []

    try:
        page_images = page.get_images(full=True)
    except Exception:
        return []

    for img_index, img in enumerate(page_images):
        xref = img[0]

        try:
            base_image = page.parent.extract_image(xref)
        except Exception:
            continue

        if not base_image or not base_image.get("image"):
            continue

        image_bytes = base_image["image"]
        b64 = base64.b64encode(image_bytes).decode("utf-8")

        images.append(
            {
                "page": page_num + 1,
                "img_index": img_index,
                "format": base_image.get("ext", "png"),
                "width": base_image.get("width", 0),
                "height": base_image.get("height", 0),
                "base64": b64,
            }
        )

    return images


def describe_image(b64_data: str, image_format: str) -> str:
    """Send an image to Groq's vision model and get a detailed description."""
    if not b64_data:
        return "[Image could not be described]"

    fmt = (image_format or "png").lower()
    mime_type = "image/jpeg" if fmt in {"jpg", "jpeg"} else f"image/{fmt}"
    data_url = f"data:{mime_type};base64,{b64_data}"

    last_error: Optional[Exception] = None

    for attempt in range(2):
        try:
            response = groq_client.chat.completions.create(
                model=GROQ_VISION_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "Describe this image in complete detail. "
                                    "Include all visible text, numbers, labels, "
                                    "diagrams, charts, relationships, and colors. "
                                    "Be specific enough that someone could answer "
                                    "detailed questions about the image without seeing it."
                                ),
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": data_url},
                            },
                        ],
                    }
                ],
                temperature=0.0,
                max_tokens=1024,
            )

            return response.choices[0].message.content

        except Exception as e:
            last_error = e
            time.sleep(0.4 * (attempt + 1))

    print(f"Image description failed: {last_error}")
    return "[Image could not be described]"


def process_pdf_images(temp_path: str, max_images: int) -> List[str]:
    """Extract images from PDF and describe them concurrently."""
    all_images: List[dict] = []

    doc = fitz.open(temp_path)
    try:
        for page_num in range(len(doc)):
            page = doc[page_num]
            all_images.extend(extract_images_from_page(page, page_num))
    finally:
        doc.close()

    if max_images > 0:
        all_images = all_images[:max_images]

    image_descriptions: List[str] = []

    if all_images:
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_to_img = {
                executor.submit(describe_image, img["base64"], img["format"]): img
                for img in all_images
            }

            for future in as_completed(future_to_img):
                img = future_to_img[future]
                try:
                    desc = future.result()
                except Exception as e:
                    desc = f"[Image could not be described: {e}]"

                img_chunk = (
                    f"[IMAGE DESCRIPTION — PAGE {img['page']}]\n"
                    f"This image is on page {img['page']}. "
                    f"Its dimensions are {img['width']}x{img['height']}.\n"
                    f"Visual content: {desc}\n"
                    f"End of image description for page {img['page']}."
                )

                image_descriptions.append(img_chunk)

    return image_descriptions
