import logging
import httpx
from dotenv import load_dotenv
from pathlib import Path
import uvicorn
import os
from PIL import Image, ImageDraw, ImageFont
from inference_sdk import InferenceHTTPClient
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext

# load env variables
env_path = Path('.') / '.env'
if env_path.exists():
    load_dotenv(dotenv_path=env_path)

# Logging configuration
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

# env variables
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY")
ROBOFLOW_API_URL = 'https://detect.roboflow.com'
MODEL_ID = 'euro-coin-detector/4'
WEBHOOK_URL = os.environ.get("WEBHOOK_URL")

# roboflow client
CLIENT = InferenceHTTPClient(
    api_url=ROBOFLOW_API_URL,
    api_key=ROBOFLOW_API_KEY
)

# telegram bot
application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

# telegram handlers
async def start(update: Update, context: CallbackContext) -> None:
    await update.message.reply_text(
        "Hi! Send me a photo containing euro coins and I'll tell you how much money there is. "
        "Since I can make mistakes in detecting coin values, if the photo contains only coins of the same value, "
        "you can write that value in the description of the photo (e.g., for a photo of 10 cent coins, "
        "write 0.1 in the description)."
    )


async def handle_photo(update: Update, context: CallbackContext) -> None:
    
    await update.message.reply_text("Image received! Processing it...")

    photo_file = await update.message.photo[-1].get_file()
    photo_path = f"{photo_file.file_path}"


    caption = update.message.caption
    custom_value = None
    if caption:
        try:
            custom_value = float(caption.strip())
        except ValueError:
            await update.message.reply_text(
                'Invalid value in description. Please make sure to enter a numeric value.'
            )

    async with httpx.AsyncClient() as client:
        photo_response = await client.get(photo_path)
        with open("photo.jpg", "wb") as f:
            f.write(photo_response.content)


    try:
        result = CLIENT.infer("photo.jpg", model_id=MODEL_ID)
        detections = result["predictions"]
        total_coins, coin_counts = draw_detections("photo.jpg", detections)


        total_value = sum([float(detection['class']) / 100 for detection in detections])

        with open("photo_with_detections.jpg", "rb") as f:
            await context.bot.send_photo(chat_id=update.effective_chat.id, photo=f)


        coin_counts_str = "\n".join(
            [f"Coins of {value:.2f} euro detected: {count}" for value, count in sorted(coin_counts.items(), reverse=True)]
        )

        response_message = (
            f"Total detected: {total_value:.2f} euro\n"
            f"Total number of coins detected: {total_coins}\n\n"
            f"{coin_counts_str}"
        )


        if custom_value:
            estimated_total = total_coins * custom_value
            response_message += (
                f"\n\nEstimated total with value {custom_value:.2f} euro: {estimated_total:.2f} euro"
            )

        await update.message.reply_text(response_message)

    except Exception as e:
        logger.error(f"Error in coin detection model request: {e}")
        await update.message.reply_text(
            'An error occurred while detecting the coins.'
        )


def calculate_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2

    xi1 = max(x1, x3)
    yi1 = max(y1, y3)
    xi2 = min(x2, x4)
    yi2 = min(y2, y4)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x4 - x3) * (y4 - y3)
    union_area = box1_area + box2_area - inter_area

    iou = inter_area / union_area
    return iou


def draw_detections(image_path, detections):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    label_font = ImageFont.load_default()
    number_font = ImageFont.load_default()

    total_coins = 0
    detected_boxes = []
    coin_counts = {}

    for detection in detections:
        x = detection["x"] - detection["width"] / 2
        y = detection["y"] - detection["height"] / 2
        width, height = detection["width"], detection["height"]
        box = [x, y, x + width, y + height]

        if any(calculate_iou(box, b) > 0.5 for b in detected_boxes):
            continue
        detected_boxes.append(box)

        draw.rectangle(box, outline="red", width=3)
        value = float(detection["class"]) / 100
        label = f"{value:.2f}€ ({detection['confidence']:.2f})"
        draw.text((x, y - 15), label, fill="red", font=label_font)

        total_coins += 1
        coin_counts[value] = coin_counts.get(value, 0) + 1

    image.save("photo_with_detections.jpg")
    return total_coins, coin_counts


# FastAPI route
@app.post("/webhook")
async def telegram_webhook(request: Request):
    data = await request.json()
    await application.update_queue.put(Update.de_json(data, application.bot))
    return {"ok": True}


@app.on_event("startup")
async def on_startup():
    
    webhook_url = f"{WEBHOOK_URL}/webhook"
    if not webhook_url.startswith("https://"):
        logger.warning("Webhook must use HTTPS. Render automatically provides this.")
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


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)