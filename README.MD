# Money Photo Counter Telegram Bot

Send a photo with euro coins and the bot will detect, annotate, and count them. If your photo contains only coins of the same value, add that value in the photo caption (for example: `0.1` for 10‑cent coins) and the bot will also estimate the total using your hint.

This version serves a FastAPI app that receives Telegram webhooks and runs model inference asynchronously for better responsiveness. Images are resized and JPEG‑compressed before inference to reduce latency.

## Requirements

Installed via `requirements.txt`:
- python-telegram-bot==22.3
- Pillow==11.3.0
- inference-sdk==0.51.10
- fastapi==0.115.2
- uvicorn==0.30.6

## Configuration (environment variables)

Set these variables in your shell or hosting environment:
- `TELEGRAM_BOT_TOKEN` – your bot token from BotFather
- `ROBOFLOW_API_KEY` – your Roboflow API key
- `ROBOFLOW_API_URL` – e.g. `https://detect.roboflow.com`
- `MODEL_ID` – your model id, e.g. `euro-coin-detector/4`
- `WEBHOOK_URL` – your public base URL, e.g. `https://your-domain.example`

On Windows (cmd.exe), you can set them for the current session like this:

```cmd
set TELEGRAM_BOT_TOKEN=123456:ABC...
set ROBOFLOW_API_KEY=rf_xxx...
set ROBOFLOW_API_URL=https://detect.roboflow.com
set MODEL_ID=euro-coin-detector/4
set WEBHOOK_URL=https://your-domain.example
```

Note: The app does not automatically read a `.env` file; define variables in your environment or run Uvicorn via a process that loads them.

## Run the server

Start the FastAPI app with Uvicorn (the ASGI server):

```cmd
uvicorn bot:app --host 0.0.0.0 --port 8000
```

When the server starts, it will register the Telegram webhook at `WEBHOOK_URL/webhook`. Ensure `WEBHOOK_URL` is reachable by Telegram (443/80/88/8443). For local development, use a tunneling tool (e.g., `ngrok http 8000`) and set `WEBHOOK_URL` to the public tunnel URL.

## How it works

- FastAPI exposes `POST /webhook` to receive Telegram updates.
- On startup, the bot registers handlers (`/start`, photo messages) and sets the webhook.
- On photo receipt, the bot:
	- downloads the image via Telegram API to memory,
	- resizes and compresses it (max side ~1024px) to speed up inference,
	- calls Roboflow inference in a background thread (non‑blocking),
	- deduplicates overlapping detections (IoU > 0.5),
	- draws boxes and labels and sends the annotated image back,
	- reports total coins, per‑value counts, and total value (from deduplicated results).

## Troubleshooting

- Unresolved imports when installing: make sure you ran `pip install -r requirements.txt` in your active Python environment.
- Webhook not receiving updates: verify `WEBHOOK_URL` points to a publicly reachable HTTPS URL and that Uvicorn is running.
- Slow responses: this version already resizes images and runs inference off the main loop. Extremely large images or slow networks can still add latency.

