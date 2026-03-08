# DeepFace Microservice

AI face detection & recognition service for VetriPhotography.

## Deploy on Render (Free Tier)

1. **Push this `deepface-service/` folder to a separate GitHub repo** (or use monorepo)
2. Go to [render.com](https://render.com) → New → Web Service
3. Connect the repo, set:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn --bind 0.0.0.0:$PORT --timeout 300 --workers 1 app:app`
   - **Python Version**: 3.11
4. Environment Variables:
   - `DEEPFACE_SECRET` = your secret key (same as `DEEPFACE_SECRET` in your Next.js `.env.local`)
5. Deploy!

## Endpoints

| Method | Path       | Description |
|--------|-----------|-------------|
| GET    | /health   | Health check |
| POST   | /detect   | Detect faces, return bounding boxes |
| POST   | /represent| Generate 512-dim face embeddings |
| POST   | /group    | Cluster photos by face identity |
| POST   | /verify   | Compare two face embeddings |

## Example

```bash
curl -X POST https://your-service.onrender.com/detect \
  -H "Authorization: Bearer YOUR_SECRET" \
  -H "Content-Type: application/json" \
  -d '{"img_url": "https://res.cloudinary.com/..."}'
```
