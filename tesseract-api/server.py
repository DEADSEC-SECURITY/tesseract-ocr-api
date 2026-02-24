from aiohttp import web
import asyncio
import tempfile
import os
import fitz

MAX_WORKERS = int(os.environ.get("MAX_WORKERS", os.cpu_count() or 4))
semaphore = asyncio.Semaphore(MAX_WORKERS)


async def run_tesseract(image_path, lang="eng"):
    async with semaphore:
        proc = await asyncio.create_subprocess_exec(
            "tesseract", image_path, "stdout", "-l", lang,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await proc.communicate()
        except asyncio.CancelledError:
            proc.kill()
            await proc.wait()
            raise
        if proc.returncode != 0:
            raise RuntimeError(stderr.decode())
        return stdout.decode().strip()


async def handle_ocr(request):
    reader = await request.multipart()

    file_data = None
    lang = "eng"

    async for field in reader:
        if field.name == "file":
            file_data = await field.read()
        elif field.name == "lang":
            lang = (await field.read()).decode()

    if file_data is None:
        return web.json_response({"error": "No file provided"}, status=400)

    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp.write(file_data)
    tmp_path = tmp.name
    tmp.close()

    task = asyncio.current_task()
    request.protocol.on_connection_lost(lambda _exc: task.cancel())

    try:
        text = await run_tesseract(tmp_path, lang)
        return web.json_response({"text": text})
    except asyncio.CancelledError:
        return web.Response(status=499)
    except RuntimeError as e:
        return web.json_response({"error": str(e)}, status=500)
    finally:
        os.unlink(tmp_path)


async def handle_ocr_pdf(request):
    reader = await request.multipart()

    file_data = None
    lang = "eng"
    dpi = 150

    async for field in reader:
        if field.name == "file":
            file_data = await field.read()
        elif field.name == "lang":
            lang = (await field.read()).decode()
        elif field.name == "dpi":
            dpi = int((await field.read()).decode())

    if file_data is None:
        return web.json_response({"error": "No file provided"}, status=400)

    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    tmp.write(file_data)
    pdf_path = tmp.name
    tmp.close()

    task = asyncio.current_task()
    request.protocol.on_connection_lost(lambda _exc: task.cancel())

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

        tasks = [asyncio.ensure_future(run_tesseract(img, lang)) for img in page_images]
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
        except asyncio.CancelledError:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

        pages = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                pages.append({"page": i, "error": str(result)})
            else:
                pages.append({"page": i, "text": result})

        return web.json_response({"total_pages": total_pages, "pages": pages})
    except asyncio.CancelledError:
        return web.Response(status=499)
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
