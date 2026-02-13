# Setup

Use `SETUP_SIMPLE.md` for full quick-start steps.

## Desktop Run

```bash
python main.py
```

## ngrok Auto-Start (Desktop Mode)

- The app now auto-starts ngrok when launching `main.py` (best effort).
- One-time requirement:

```bash
ngrok config add-authtoken <YOUR_TOKEN>
```

## Ports

- App server: `127.0.0.1:8000`
- ngrok local API/status: `127.0.0.1:4040`
- Public ngrok HTTPS URL forwards to app port `8000`

## Disable ngrok Auto-Start

```bash
AUTO_START_NGROK=false python main.py
```
