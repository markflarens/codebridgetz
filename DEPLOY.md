# Deploying (Railway)

Railway isn't reachable from the environment that prepared this repo, so
these steps are for you to click through once the code is pushed.

1. **New Project → Deploy from GitHub repo.** Pick this repo. Railway
   will detect `Dockerfile` and `railway.json` in the repo root and build
   from those automatically — no manual build settings needed.
2. Wait for the first build. It installs ImageMagick and the Python deps
   from `backend/requirements.txt` inside the container (see `Dockerfile`
   for why ImageMagick is there: HEIC photos from iPhones).
3. Once deployed, open **Settings → Networking → Generate Domain** if a
   public URL wasn't created automatically. Railway's generated domains
   are HTTPS by default — this is what satisfies the
   `capture="environment"` camera-input requirement on iOS Safari (it
   needs a secure context).
4. Auto-deploy on push is on by default for a GitHub-connected service —
   no extra step needed; every push to the connected branch redeploys.
5. Sanity check once it's live:
   - Open the root URL in a browser — the frontend (same FastAPI app)
     should load.
   - `GET /api/marker` should return the marker PDF.
   - `POST /api/measure` with a real photo should return a JSON result —
     see `README.md`'s API section for the response shape.

## Verifying deployed == local

`tools/analyze_photo.py` only talks to the pipeline in-process; it
doesn't hit the API. To compare the live deployment against a local run
on the same photo, `POST` the photo to the live URL and diff the
`diameter_mm`/`family_estimates` fields against a local
`python3 tools/analyze_photo.py <photo>` run — for example:

```bash
curl -s -F "file=@real_ring_A_IMG_9783.jpg" https://<your-app>.up.railway.app/api/measure | python3 -m json.tool
```

Only do this once `requirements.txt` is pinned to the versions that
produced your verified local numbers (see repo README note on this) —
otherwise "local" and "deployed" can each be right and still disagree,
for the same reason this repo's bundled 16.80mm/26.47mm result didn't
reproduce on a fresh, unpinned install.
