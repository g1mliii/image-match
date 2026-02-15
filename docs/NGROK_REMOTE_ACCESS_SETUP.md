# ngrok Remote Access Setup (No Domain Required)

This is the easiest off-site setup when you do not own a Cloudflare/domain setup.

## What This Gives You

- Keep local use:
  - `http://127.0.0.1:8000/mobile`
  - `http://<LAN-IP>:8000/mobile`
- Add remote HTTPS URL from ngrok for use on any network.
- Keep processing on desktop/backend; mobile only uploads/views results.

## One-Time Setup (Host Desktop)

1. Install ngrok
   - https://ngrok.com/download
2. In app, open `CONNECT PHONE`
3. Paste token in `NGROK TOKEN`
4. Click `SETUP TOKEN`

## Daily Use (Very Simple)

1. Start your app normally (`RUN.bat`).
2. ngrok auto-starts in background (if installed and configured).
3. In app, open `CONNECT PHONE`.
4. Click `AUTO NGROK`.
5. Share the shown remote URL with your client.

The app auto-saves the ngrok URL and uses it in QR code / remote display.

If you prefer manual tunnel start, use:

- `tools\Start Ngrok Tunnel.bat`

## Security Minimum (Recommended)

- Keep using the mobile PIN (already required by app).
- Regenerate PIN regularly from `CONNECT PHONE`.
- Only share URL + PIN with authorized users.
- Stop ngrok when not needed (close ngrok window).

## Optional Stronger Security

- Use ngrok traffic policy / identity controls (OAuth allowlist).
- Restrict by client emails or organization domain.

## Notes

- ngrok local API/status is on `http://127.0.0.1:4040` (not same as app port).
- App listens on `http://127.0.0.1:8000` in desktop mode.
- ngrok free URLs can change when tunnel restarts.
- If URL changes, click `AUTO NGROK` again.
- If your app uses a different port, run:
  - `tools\Start Ngrok Tunnel.bat 5000`
- To disable auto-start:
  - `set AUTO_START_NGROK=false`

## Troubleshooting

- "Could not reach ngrok API":
  - ngrok is not running.
- "No active ngrok tunnels found":
  - start tunnel with `ngrok http http://127.0.0.1:8000`.
- "No HTTPS ngrok tunnel found":
  - ensure standard `ngrok http` mode is used.
