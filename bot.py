import os
import asyncio
import logging
import tempfile
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
from inference_sdk import InferenceHTTPClient
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext
from fastapi import FastAPI, Request


# Environment variables
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY")
ROBOFLOW_API_URL = os.environ.get("ROBOFLOW_API_URL")
MODEL_ID = os.environ.get("MODEL_ID")
WEBHOOK_URL = os.environ.get("WEBHOOK_URL")

CLIENT = InferenceHTTPClient(api_url=ROBOFLOW_API_URL, api_key=ROBOFLOW_API_KEY)

application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

app = FastAPI()


# basic logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


# telegram handlers
async def start(update: Update, context: CallbackContext):
    await update.message.reply_text(
        "Hi! Send me a photo containing euro coins and I'll tell you how much money there is. "
        "If the photo contains only coins of the same value, you can write that value in the description (e.g., 0.1 for 10 cent coins)."
    )


async def handle_photo(update: Update, context: CallbackContext):
    
    photo_file = await update.message.photo[-1].get_file()

    # custom value from caption
    caption = update.message.caption
    if caption:
        try:
            custom_value = float(caption.strip())
        except ValueError:
            custom_value = None  # ignore non-numeric captions

    # download image bytes via Telegram API and preprocess
    img_bytes = await photo_file.download_as_bytearray()
    image = Image.open(BytesIO(img_bytes)).convert("RGB")
    temp_input_path = _resize_and_save_temp(image)

    try:
        # run inference
        result = await asyncio.to_thread(CLIENT.infer, temp_input_path, model_id=MODEL_ID)
        detections = result["predictions"]
        # create a temp output path for annotated image
        out_fd, out_path = tempfile.mkstemp(suffix=".jpg")
        os.close(out_fd)
        total_coins, coin_counts, annotated_path = draw_detections(temp_input_path, detections, output_path=out_path)
        # compute total from deduplicated counts
        total_value = sum(val * cnt for val, cnt in coin_counts.items())

        with open(annotated_path, "rb") as f:
            await context.bot.send_photo(chat_id=update.effective_chat.id, photo=f)

        coin_counts_str = "\n".join(
            [f"Coins of {value:.2f} euro detected: {count}" for value, count in sorted(coin_counts.items())]
        )

        response_message = (
            f"Total detected: {total_value:.2f} euro\n"
            f"Total number of coins detected: {total_coins}\n\n"
            f"{coin_counts_str}"
        )

        if custom_value is not None:
            estimated_total = total_coins * custom_value
            response_message += f"\n\nEstimated total with value {custom_value:.2f} euro: {estimated_total:.2f} euro"

        await update.message.reply_text(response_message)

    except Exception as e:
        logger.exception("Error in coin detection")
        await update.message.reply_text("Sorry, an error occurred. Try again later.")
    finally:
        # cleanup temporary files
        try:
            if os.path.exists(temp_input_path):
                os.remove(temp_input_path)
        except Exception:
            pass
        try:
            if 'annotated_path' in locals() and os.path.exists(annotated_path):
                os.remove(annotated_path)
        except Exception:
            pass


def calculate_iou(box1, box2): # intersection over union, overlap ratio of two boxes
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2
    left = max(x1, x3)
    top = max(y1, y3)
    right = min(x2, x4)
    bottom = min(y2, y4)
    inter_area = max(0, right - left) * max(0, bottom - top)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x4 - x3) * (y4 - y3)
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area # 0 = no overlap, 1 = identical boxes


def draw_detections(image_path, detections, output_path: str | None = None):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    label_font = ImageFont.load_default()

    total_coins = 0
    detected_boxes = []
    coin_counts = {} # dict mapping coin value -> count

    for d in detections:
        # conversion from center-based detections (origin at top left) to corner-based bounding box
        left = d["x"] - d["width"] / 2 
        top = d["y"] - d["height"] / 2
        right = d["x"] + d["width"] / 2
        bottom = d["y"] + d["height"] / 2
        box = [left, top, right, bottom]

        # avoid detecting the same box twice, but allow detection of overlapping coins
        if any(calculate_iou(box, b) > 0.5 for b in detected_boxes):
            continue
        else:
            detected_boxes.append(box)

        draw.rectangle(box, outline="red", width=3)
        value = float(d["class"]) / 100 # classes are in cents
        label = f"{value:.2f} euros ({d['confidence']:.2f})"
        draw.text((left, top - 15), label, fill="red", font=label_font) # label a bit above the top left corner

        total_coins += 1
        coin_counts[value] = coin_counts.get(value, 0) + 1

    if not output_path:
        output_path = "photo_with_detections.jpg"
    image.save(output_path, format="JPEG", quality=90)
    return total_coins, coin_counts, output_path


def _resize_and_save_temp(image: Image.Image, max_side: int = 1024, quality: int = 85) -> str:
    """Resize preserving aspect ratio if needed and save to a temporary JPEG file. Returns the temp file path."""
    w, h = image.size
    if max(w, h) > max_side:
        scale = max_side / float(max(w, h))
        new_size = (int(w * scale), int(h * scale))
        image = image.resize(new_size, Image.LANCZOS)
    fd, temp_path = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)
    image.save(temp_path, format="JPEG", optimize=True, quality=quality)
    return temp_path


# fastAPI webhook endpoint
@app.post("/webhook")
async def telegram_webhook(request: Request):
    data = await request.json()
    update = Update.de_json(data, application.bot)
    await application.update_queue.put(update)
    return {"ok": True}


# startup and shutdown events
@app.on_event("startup")
async def on_startup():
    webhook_url = f"{WEBHOOK_URL}/webhook"
    await application.bot.set_webhook(webhook_url)
    logger.info(f"Webhook set to {webhook_url}")

    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))

    await application.initialize()
    await application.start()
    logger.info("Bot is ready!")


@app.on_event("shutdown")
async def on_shutdown():
    await application.stop()
    await application.shutdown()
