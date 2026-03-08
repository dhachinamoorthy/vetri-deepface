"""
DeepFace Microservice for VetriPhotography
==========================================
Deployed on Render (free tier). Provides:
  - POST /detect     → Detect faces in an image, return count + bounding boxes
  - POST /represent  → Generate 512-dim face embeddings (Facenet512)
  - POST /group      → Cluster all images in a gallery by face identity
  - GET  /health     → Health check

Accepts images as:
  - base64 string in JSON body
  - URL to fetch from (e.g., Cloudinary / R2 URL)
"""

import os
import io
import json
import base64
import hashlib
import traceback
from typing import Any

import numpy as np
import requests as http_requests
from PIL import Image as PILImage
from flask import Flask, request, jsonify
from flask_cors import CORS

# Pre-import deepface — models will be downloaded on first call
from deepface import DeepFace

app = Flask(__name__)
CORS(app)

# Service secret for auth (set in Render env vars)
SERVICE_SECRET = os.environ.get("DEEPFACE_SECRET", "")


def verify_auth() -> bool:
    """Verify the request has the correct auth header."""
    if not SERVICE_SECRET:
        return True  # No secret configured = allow all (dev mode)
    auth = request.headers.get("Authorization", "")
    return auth == f"Bearer {SERVICE_SECRET}"


def load_image_from_request(data: dict) -> str | None:
    """
    Extract image source from request data.
    Returns a path/URL/base64 string that DeepFace can consume.
    """
    if "img_url" in data and data["img_url"]:
        return data["img_url"]
    if "img_base64" in data and data["img_base64"]:
        return data["img_base64"]
    return None


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "deepface-vetri"})


@app.route("/detect", methods=["POST"])
def detect_faces():
    """
    Detect faces in an image. Returns face count and bounding boxes.
    Body: { "img_url": "https://..." } or { "img_base64": "..." }
    """
    if not verify_auth():
        return jsonify({"error": "Unauthorized"}), 401

    try:
        data = request.get_json(force=True)
        img_source = load_image_from_request(data)
        if not img_source:
            return jsonify({"error": "Provide img_url or img_base64"}), 400

        faces = DeepFace.extract_faces(
            img_path=img_source,
            detector_backend="retinaface",
            enforce_detection=False,
            align=True,
        )

        result = []
        for face in faces:
            area = face.get("facial_area", {})
            result.append({
                "x": int(area.get("x", 0)),
                "y": int(area.get("y", 0)),
                "w": int(area.get("w", 0)),
                "h": int(area.get("h", 0)),
                "confidence": float(face.get("confidence", 0)),
            })

        return jsonify({
            "face_count": len(result),
            "faces": result,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/represent", methods=["POST"])
def represent():
    """
    Generate face embedding(s) for an image using Facenet512.
    Returns 512-dim vector per detected face.
    Body: { "img_url": "https://..." }
    """
    if not verify_auth():
        return jsonify({"error": "Unauthorized"}), 401

    try:
        data = request.get_json(force=True)
        img_source = load_image_from_request(data)
        if not img_source:
            return jsonify({"error": "Provide img_url or img_base64"}), 400

        embeddings = DeepFace.represent(
            img_path=img_source,
            model_name="Facenet512",
            detector_backend="retinaface",
            enforce_detection=False,
            align=True,
        )

        result = []
        for emb in embeddings:
            area = emb.get("facial_area", {})
            result.append({
                "embedding": emb["embedding"],
                "face": {
                    "x": int(area.get("x", 0)),
                    "y": int(area.get("y", 0)),
                    "w": int(area.get("w", 0)),
                    "h": int(area.get("h", 0)),
                },
            })

        return jsonify({
            "face_count": len(result),
            "embeddings": result,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/group", methods=["POST"])
def group_faces():
    """
    Given a list of photos with their embeddings, cluster them by identity.
    Body: {
      "photos": [
        { "id": 1, "embeddings": [[...512 floats...], ...] },
        ...
      ],
      "threshold": 0.40  // optional, default cosine distance threshold
    }
    Returns: { "clusters": { "person_1": [1,3,5], "person_2": [2,4], "no_face": [6,7] } }
    """
    if not verify_auth():
        return jsonify({"error": "Unauthorized"}), 401

    try:
        data = request.get_json(force=True)
        photos = data.get("photos", [])
        threshold = float(data.get("threshold", 0.40))

        if not photos:
            return jsonify({"error": "No photos provided"}), 400

        # Collect all face embeddings with photo references
        all_faces: list[dict[str, Any]] = []
        no_face_photo_ids: list[int] = []

        for photo in photos:
            photo_id = photo["id"]
            embs = photo.get("embeddings", [])
            if not embs:
                no_face_photo_ids.append(photo_id)
                continue
            for emb in embs:
                vec = np.array(emb, dtype=np.float32)
                all_faces.append({"photo_id": photo_id, "embedding": vec})

        # Simple greedy clustering by cosine distance
        clusters: list[list[dict]] = []

        for face in all_faces:
            placed = False
            for cluster in clusters:
                # Compare with cluster representative (first face)
                rep = cluster[0]["embedding"]
                # Cosine distance
                cos_sim = float(np.dot(face["embedding"], rep) / (
                    np.linalg.norm(face["embedding"]) * np.linalg.norm(rep) + 1e-10
                ))
                distance = 1.0 - cos_sim
                if distance < threshold:
                    cluster.append(face)
                    placed = True
                    break
            if not placed:
                clusters.append([face])

        # Build result: assign person labels
        result: dict[str, list[int]] = {}

        for i, cluster in enumerate(clusters):
            label = f"person_{i + 1}"
            photo_ids = list(set(f["photo_id"] for f in cluster))
            result[label] = sorted(photo_ids)

        if no_face_photo_ids:
            result["no_face"] = sorted(no_face_photo_ids)

        # Also build a photo_id → groups mapping
        photo_groups: dict[int, list[str]] = {}
        for label, pids in result.items():
            for pid in pids:
                if pid not in photo_groups:
                    photo_groups[pid] = []
                photo_groups[pid].append(label)

        return jsonify({
            "clusters": result,
            "photo_groups": photo_groups,
            "total_persons": len(clusters),
            "total_photos_with_faces": len(set(f["photo_id"] for f in all_faces)),
            "total_photos_no_face": len(no_face_photo_ids),
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/verify", methods=["POST"])
def verify_faces():
    """
    Compare two face embeddings directly (cosine distance).
    Body: { "embedding1": [...], "embedding2": [...], "threshold": 0.40 }
    """
    if not verify_auth():
        return jsonify({"error": "Unauthorized"}), 401

    try:
        data = request.get_json(force=True)
        emb1 = np.array(data["embedding1"], dtype=np.float32)
        emb2 = np.array(data["embedding2"], dtype=np.float32)
        threshold = float(data.get("threshold", 0.40))

        cos_sim = float(np.dot(emb1, emb2) / (
            np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-10
        ))
        distance = 1.0 - cos_sim

        return jsonify({
            "verified": distance < threshold,
            "distance": round(distance, 6),
            "similarity": round(cos_sim, 6),
            "threshold": threshold,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5005))
    app.run(host="0.0.0.0", port=port, debug=False)
