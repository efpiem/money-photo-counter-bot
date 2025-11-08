# Money Photo Counter Telegram Bot

Send a photo with euro coins and the bot will detect, annotate, and count them. If your photo contains only coins of the same value, add that value in the photo caption (for example: `0.1` for 10‑cent coins) and the bot will also estimate the total using your hint.

The python script bot.py serves a FastAPI app that receives Telegram webhooks and runs Roboflow model inference asynchronously for better responsiveness. Images are resized and JPEG‑compressed before inference to reduce latency.

## Run the server

Start the FastAPI app with Uvicorn (ASGI server):

```cmd
uvicorn bot:app --host 0.0.0.0 --port 8000
```

When the server starts, it will register the Telegram webhook at `WEBHOOK_URL/webhook`.

## How it works

- FastAPI exposes `POST /webhook` to receive Telegram updates.
- On startup, the bot registers handlers (`/start`, photo messages) and sets the webhook.
- On photo receipt, the bot:
	- downloads the image via Telegram API to memory,
	- resizes and compresses it (max side ~1024px) to speed up inference,
	- calls Roboflow inference using the SDK's async API, guarded by concurrency limits and a timeout,
	- deduplicates overlapping detections (IoU > 0.5),
	- draws boxes and labels and sends the annotated image back,
	- reports total coins, per‑value counts, and total value.