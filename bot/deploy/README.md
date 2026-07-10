# VPS deploy checklist (public repo — zero trading secrets in CI)

One-time VPS setup (any 1 vCPU / 1 GB box, Bybit-permitted jurisdiction;
needs python3.11, NTP, systemd, outbound HTTPS/WSS only):

1. Create a dedicated `deploy` user; `loginctl enable-linger deploy`.
2. `git clone https://github.com/<owner>/<repo>.git /opt/traderbot` (anonymous
   https — public repo; the deploy SSH key is for the CI->VPS session only).
3. `python3.11 -m venv /opt/traderbot/venv && venv/bin/pip install -r bot/requirements.txt`
4. Create `/opt/traderbot/.env` by hand from `bot/.env.example`, `chmod 600 .env`.
   LIVE Bybit key: trade-only permission, IP-whitelisted to this VPS.
   The Windows dev machine only ever holds DEMO keys.
5. Install the user unit (see header of `bot.service`), start, check
   `systemctl --user status bot` and the Telegram "bot started" message.
6. Install the update command: `ln -s /opt/traderbot/bot/deploy/update.sh
   ~/bin/traderbot-update` (ensure `~/bin` on PATH for ssh sessions), or call
   the script by full path from CI.
7. Add the CI deploy key: generate a dedicated ed25519 keypair, public half to
   `~deploy/.ssh/authorized_keys`, private half to the GitHub repo secret
   `VPS_SSH_KEY`; `VPS_HOST` secret = `deploy@<ip>`.

GitHub side: repo Settings -> Environments -> `vps` holding the TWO secrets
(`VPS_SSH_KEY`, `VPS_HOST`). Nothing else. NO Bybit keys, NO Telegram token in
GitHub, ever. Enable secret scanning + push protection.

Rollback: Actions -> deploy -> Run workflow with `ref` = a previous commit sha.
