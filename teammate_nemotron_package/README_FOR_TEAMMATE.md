# Scoring task: 32 replies with Nemotron (about 15 minutes)

You will run one AI model over 32 chat replies and send back a results file. No GPU needed, no big
downloads, and nothing from the main project is required. **Use your own free OpenRouter account** -
never share or reuse someone else's API key.

You were given **one** task file: `tasks_teammate1.csv` **or** `tasks_teammate2.csv`. Use only that one.

## 1. Install Python (skip if you already have it)

Download Python 3.10 or newer from <https://www.python.org/downloads/> and **tick "Add Python to PATH"**
during installation.

## 2. Install the three packages

Open PowerShell (or Command Prompt) in the folder containing these files and run:

```
pip install openai pandas python-dotenv
```

## 3. Get a free OpenRouter API key

1. Go to <https://openrouter.ai> and sign up (free).
2. Open **Keys** in your account menu and create a new key. It starts with `sk-or-`.
3. In this folder, create a file named exactly `.env` containing one line:

```
OPENROUTER_API_KEY=sk-or-your-own-key-here
```

Keep this key private. Do not send it to anyone, including us.

## 4. Run the scoring

```
python score_nemotron.py --tasks tasks_teammate1.csv
```

(use your own file name). It prints progress every 5 replies and takes about 15 minutes for 32 replies,
because the free tier requires a short pause between requests. Leave the window open until it finishes.

## 5. Send the result back

The script creates **`tasks_teammate1_scores.csv`** (or `tasks_teammate2_scores.csv`) in the same folder.
Send that file back. Nothing else is needed.

## If something goes wrong

| Message | What to do |
|---|---|
| `No OPENROUTER_API_KEY found` | The `.env` file is missing, misspelled, or in the wrong folder. It must sit next to `score_nemotron.py`. Windows may hide the name; make sure it is `.env`, not `.env.txt`. |
| `pip is not recognised` | Python was installed without "Add to PATH". Reinstall and tick that box. |
| `429` or `rate limit` in the errors | The free tier allows 50 requests per day. Wait until tomorrow and run the same command again; replies already scored are skipped. |
| Some replies failed | Run the exact same command again later. Only the failed ones are retried. |
| The window closed early | Just run the command again; it continues where it stopped. |

## What this is for

These scores are one of the independent measures in our thesis evaluation. The model reads only the
user's message and the reply, never which system produced it, so the scoring stays unbiased. The
conversations come from the public EmpatheticDialogues research dataset, so no personal data is involved.
