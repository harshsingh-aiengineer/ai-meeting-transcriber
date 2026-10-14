# Interview notes

Talking points for this repository. Numbers below are limits in the code, not measured benchmarks.

## Architecture

Streamlit is the UI. `TranscriptionService` calls faster-whisper and keeps one model loaded for the process. ffmpeg turns the upload into a 16 kHz mono WAV in a temporary directory. Segments are formatted as plain text, a timestamped transcript, SRT, and JSON. `MeetingAnalysisService` checks Ollama, then asks for JSON. `MeetingStore` writes meeting metadata to SQLite. Transcript text stays in files under `data/outputs/`.

The LLM is not on the transcription path. A failed analysis returns a result object. It does not raise into the upload flow.

## Why faster-whisper

The project needs local speech-to-text without a paid API. faster-whisper runs OpenAI Whisper weights through CTranslate2. The app uses CPU and `int8` by default, and it offers `tiny`, `base`, and `small`. The model is loaded once and reused. PyAV is pinned below version 19 because faster-whisper 1.2.1 still passes `metadata_errors` to `av.open`, and PyAV 19 removed that argument.

## Why a local LLM

Meeting audio is private business context. Ollama keeps the brief on a machine the team controls. The model name is `OLLAMA_MODEL`. The default is `qwen2.5:3b`. The app asks Ollama which models are installed before it sends a transcript. A missing model gets a pull instruction. It is not treated as a crash.

## How model fallback works

`OllamaClient.health()` calls the tags endpoint with a short timeout. If the server cannot be reached, analysis is marked unavailable and the UI shows: AI analysis unavailable — start Ollama to enable meeting intelligence. The transcript, SRT, and downloads are already written. If Ollama is up but the named model is missing, the message names that model and the installed models. Malformed model JSON is parsed into the same result object, with `available` false, instead of taking down the page.

## Bottlenecks

- Transcription is synchronous inside the Streamlit process. A long recording blocks that session on CPU.
- The first run of a Whisper size downloads weights before any text appears.
- Analysis sends at most four chunks of about 10,000 characters. A longer transcript is only partly analyzed, and the result says so.
- One Ollama model call is another wait after transcription. There is no job queue.
- Speaker labels are a pause heuristic. They are not a diarization model.

## Scaling considerations

This is a single-user local app. SQLite and the `data/` directory are enough for one machine. A multi-user deployment would need authentication, a queue for transcription jobs, object storage for audio and reports, and a way to run Whisper off the web process. GPU would be a separate compute path. The current Docker image is CPU-only and does not bundle model weights.

## Security considerations

Uploads are checked for extension, emptiness, and size. Filenames are reduced to a safe ASCII name and stored under a generated meeting id. Saved paths must stay inside the output directory. SQLite stores metadata, not audio bytes. Logs are JSON lines. They drop credential-like assignments and cut very long messages so a transcript is not written to the log. There is no login. The service should stay on a private network or behind an authenticated proxy.

## Future production improvements

These are not built:

- A worker queue so the web process does not run Whisper inline
- Real diarization
- Auth and per-user storage
- Search over saved meetings
- Editable action items before export
- GPU transcription only when the container actually has CUDA

A 60–90 second click path is in [demo-script.md](demo-script.md).
