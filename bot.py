import logging
import httpx
from PIL import Image, ImageDraw, ImageFont
from inference_sdk import InferenceHTTPClient
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext
from dotenv import load_dotenv
from pathlib import Path
import os

env_path = Path('.') / '.env'
if env_path.exists():
    load_dotenv(dotenv_path=env_path)

# Logging configuration
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

# Telegram bot token and inference client configuration
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY")
ROBOFLOW_API_URL = 'https://detect.roboflow.com'
MODEL_ID = 'euro-coin-detector/4'

# Configure the inference client
CLIENT = InferenceHTTPClient(
    api_url=ROBOFLOW_API_URL,
    api_key=ROBOFLOW_API_KEY
)

# Function to handle /start messages
async def start(update: Update, context: CallbackContext) -> None:
    await update.message.reply_text(
        "Hi! Send me a photo containing euro coins and I'll tell you how much money there is. "
        "Since I can make mistakes in detecting coin values, if the photo contains only coins of the same value, "
        "you can write that value in the description of the photo (e.g., for a photo of 10 cent coins, "
        "write 0.1 in the description)."
    )

# Function to handle sent photos
async def handle_photo(update: Update, context: CallbackContext) -> None:
    # Send confirmation message right away
    await update.message.reply_text("Image received! Processing it...")

    photo_file = await update.message.photo[-1].get_file()
    photo_path = f"{photo_file.file_path}"

    # Extract the value from the photo description
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

    # Send inference request to the coin detection model
    try:
        result = CLIENT.infer("photo.jpg", model_id=MODEL_ID)
        detections = result["predictions"]
        total_coins, coin_counts = draw_detections("photo.jpg", detections)

        # Calculate the total value of coins (dividing by 100)
        total_value = sum([float(detection['class']) / 100 for detection in detections])

        with open("photo_with_detections.jpg", "rb") as f:
            await context.bot.send_photo(chat_id=update.effective_chat.id, photo=f)

        # Format coin count per denomination
        coin_counts_str = "\n".join(
            [f"Coins of {value:.2f} euro detected: {count}" for value, count in sorted(coin_counts.items(), reverse=True)]
        )

        response_message = (
            f"Total detected: {total_value:.2f} euro\n"
            f"Total number of coins detected: {total_coins}\n\n"
            f"{coin_counts_str}"
        )

        # Calculate and add the estimated total if a custom value was provided
        if custom_value:
            estimated_total = total_coins * custom_value
            response_message += (
                f"\n\nEstimated total with value {custom_value:.2f} euro: {estimated_total:.2f} euro"
            )

        # Send message with the total coins, total number of coins, and estimated total (if applicable)
        await update.message.reply_text(response_message)

    except Exception as e:
        logger.error(f"Error in coin detection model request: {e}")
        await update.message.reply_text(
            'An error occurred while detecting the coins.'
        )

# Function to calculate Intersection over Union (IoU)
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

# Function to draw detections on the image and count boxes
def draw_detections(image_path, detections):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    # Use a system font for labels
    label_font_size = 24
    try:
        label_font = ImageFont.truetype("arial.ttf", label_font_size)
    except IOError:
        label_font = ImageFont.load_default()

    # Use a larger font for numbers
    number_font_size = 48
    try:
        number_font = ImageFont.truetype("arial.ttf", number_font_size)
    except IOError:
        number_font = ImageFont.load_default()

    total_coins = 0
    detected_boxes = []
    coin_counts = {}

    for idx, detection in enumerate(detections):
        x = detection["x"] - detection["width"] / 2
        y = detection["y"] - detection["height"] / 2
        width = detection["width"]
        height = detection["height"]
        box = [x, y, x + width, y + height]

        # Check for overlaps
        overlap = False
        for detected_box in detected_boxes:
            iou = calculate_iou(box, detected_box)
            if iou > 0.5:
                overlap = True
                break

        if not overlap:
            detected_boxes.append(box)

            # Draw the rectangle
            draw.rectangle(box, outline="red", width=3)

            # Draw the label above the rectangle
            value = float(detection['class']) / 100
            label = f"{value:.2f} euro ({detection['confidence']:.2f})"
            text_width, text_height = draw.textsize(label, font=label_font)
            draw.text((x, y - text_height), label, fill="red", font=label_font)

            # Draw the number in the center
            number_label = f"{total_coins + 1}"
            number_text_width, number_text_height = draw.textsize(number_label, font=number_font)
            number_text_x = x + (width - number_text_width) / 2
            number_text_y = y + (height - number_text_height) / 2
            draw.text((number_text_x, number_text_y), number_label, fill="blue", font=number_font)

            total_coins += 1

            # Update coin count
            if value in coin_counts:
                coin_counts[value] += 1
            else:
                coin_counts[value] = 1

    image.save("photo_with_detections.jpg")
    return total_coins, coin_counts

# Main function to start the bot
def main() -> None:
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))

    application.run_polling()

if __name__ == '__main__':
    main()