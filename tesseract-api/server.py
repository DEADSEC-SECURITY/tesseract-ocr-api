from aiohttp import web
import asyncio
from concurrent.futures import ProcessPoolExecutor
import tempfile
import os
import fitz

MAX_WORKERS = int(os.environ.get("MAX_WORKERS", os.cpu_count() or 4))

_ocr = None


def _init_worker():
    global _ocr
    from paddleocr import PaddleOCR
    _ocr = PaddleOCR(use_textline_orientation=True, lang="en", device="cpu")


def _ocr_image(image_path):
    result = _ocr.ocr(image_path)
    lines = []
    if result and result[0]:
        for line in result[0]:
            lines.append(line[1][0])
    return "\n".join(lines)


executor = ProcessPoolExecutor(max_workers=MAX_WORKERS, initializer=_init_worker)


async def run_ocr(image_path):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, _ocr_image, image_path)


async def handle_ocr(request):
    reader = await request.multipart()

    file_data = None

    async for field in reader:
        if field.name == "file":
            file_data = await field.read()

    if file_data is None:
        return web.json_response({"error": "No file provided"}, status=400)

    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp.write(file_data)
    tmp_path = tmp.name
    tmp.close()

    try:
        text = await run_ocr(tmp_path)
        return web.json_response({"text": text})
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)
    finally:
        os.unlink(tmp_path)


async def handle_ocr_pdf(request):
    reader = await request.multipart()

    file_data = None
    dpi = 150

    async for field in reader:
        if field.name == "file":
            file_data = await field.read()
        elif field.name == "dpi":
            dpi = int((await field.read()).decode())

    if file_data is None:
        return web.json_response({"error": "No file provided"}, status=400)

    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    tmp.write(file_data)
    pdf_path = tmp.name
    tmp.close()

    page_images = []
    try:
        doc = fitz.open(pdf_path)
        total_pages = len(doc)

        for i in range(total_pages):
            pix = doc[i].get_pixmap(dpi=dpi, colorspace=fitz.csGRAY)
            img_path = tempfile.mktemp(suffix=".png")
            pix.save(img_path)
            page_images.append(img_path)

        doc.close()

        tasks = [run_ocr(img) for img in page_images]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        pages = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                pages.append({"page": i, "error": str(result)})
            else:
                pages.append({"page": i, "text": result})

        return web.json_response({"total_pages": total_pages, "pages": pages})
    finally:
        os.unlink(pdf_path)
        for img in page_images:
            if os.path.exists(img):
                os.unlink(img)


async def handle_health(request):
    return web.json_response({"status": "ok"})


app = web.Application()
app.router.add_post("/ocr", handle_ocr)
app.router.add_post("/ocr/pdf", handle_ocr_pdf)
app.router.add_get("/health", handle_health)

if __name__ == "__main__":
    web.run_app(app, host="0.0.0.0", port=8884)
