import numpy as np
import time
import cv2
import torch # Needed so the CUDAExecutionProvider can be created (access the CUDA that was installed with PyTorch). 
             # Can also use onnxruntime.preload_dlls() before starting the session.
import onnxruntime as ort
from sklearn.metrics.pairwise import cosine_similarity

class MobileFaceEmbedder:
    def __init__(self):
        model_path = str("C:/Users/bkarr/.insightface/models/antelopev2/antelopev2/1k3d68.onnx")
        
        self.session = ort.InferenceSession(
            model_path,
            providers=["CUDAExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

    def preprocess(self, face_crop_bgr):
        img = cv2.resize(face_crop_bgr, (192, 192))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img = (img / 255.0 - 0.5) / 0.5  # Normalize to [-1, 1]
        img = np.transpose(img, (2, 0, 1))  # HWC to CHW
        img = np.expand_dims(img, axis=0).astype(np.float32)  # (1, 3, 192, 192)
        return img

    def get_embedding(self, face_crop_bgr):
        img = self.preprocess(face_crop_bgr)
        emb = self.session.run(None, {self.input_name: img})[0]
        return emb[0]  # 512-dim vector


class FaceMemory:
    def __init__(self, threshold=0.6):
        self.memory = []
        self.next_id = 1
        self.threshold = threshold
        self.embedder = MobileFaceEmbedder()

    def match_or_add(self, face_crop):
        try:
            emb = self.embedder.get_embedding(face_crop)
        except:
            return None

        known_embs = [entry["embedding"] for entry in self.memory]
        if known_embs:
            sims = cosine_similarity([emb], known_embs)[0]
            best_idx = np.argmax(sims)
            if sims[best_idx] > self.threshold:
                self.memory[best_idx]["last_seen"] = time.time()
                return self.memory[best_idx]["id"]

        new_id = self.next_id
        self.memory.append({
            "id": new_id,
            "embedding": emb,
            "label": f"unknown_{new_id}",
            "last_seen": time.time()
        })
        self.next_id += 1
        return new_id