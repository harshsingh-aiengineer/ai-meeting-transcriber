# Demo script

About 60–90 seconds. Use a short spoken clip, not a long meeting. The sample used in development was about 9 seconds.

Before you start:

- Streamlit is open at http://127.0.0.1:8501
- Ollama is running and `qwen2.5:3b` is installed
- The sidebar model is `base` or `tiny`, device `cpu`, compute `int8`

## 1. Upload meeting audio

Click **Upload** and choose a short WAV or MP3. Say what the clip contains: a decision, an owner, and a risk.

## 2. Start transcription

Click **Transcribe**. Point at the progress line. The first run downloads Whisper weights. Later runs reuse the cache.

## 3. Show the timestamped transcript

Open the **Timestamped Transcript** tab. Read one line with its start time. Mention the **SRT** and **JSON** tabs as the same segments in other formats.

## 4. Show the AI summary

Scroll to **Meeting Summary**. Read the title and the one-paragraph summary. If the model was already warm, this appears after transcription without a second click.

## 5. Show action items

Open **Action Items**. Point at task, owner, deadline, and priority. Then show **Decisions** as the separate list.

## 6. Download the report

In **Downloads**, click **Markdown**. Say the file contains the title, date, summary, key points, decisions, action items, risks, follow-up questions, and the transcript.

## If Ollama is off

Leave the transcript on screen. The summary area shows: AI analysis unavailable — start Ollama to enable meeting intelligence. Transcription does not roll back.

## What not to claim

Do not quote speed or accuracy numbers. This demo does not include a benchmark. Speaker labels, when enabled, are pause estimates, not voice identification.
