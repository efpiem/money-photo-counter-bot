# Money Photo Counter Telegram Bot

Send a photo with euro coins, and the bot will detect and count them.

Since the bot can make mistakes in detecting coin values, if the photo contains **only coins of the same value**, you can write that value in the **description/caption** of the photo.  
For example, for a photo containing only 10 cent coins, in the caption write: 0.1  

## Running locally
1. **Install dependencies**
```bash
pip install -r requirements.txt
```
2. **Create your local environment file**

Copy the example `.env` file to create your own:
```bash
cp .env.example .env
```
Open .env in a text editor and fill in your API keys.
Save the file. Your bot will now be able to read the environment variables when running locally.

3. **Run the bot**
```bash
python bot.py
```
