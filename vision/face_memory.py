import face_recognition
import numpy as np
import time
import cv2

class FaceMemoryManager:
    def __init__(self, threshold=0.6):
        self.memory = []
        self.next_id = 1
        self.threshold = threshold

    def match_or_add(self, face_crop):
        # Convert from BGR (OpenCV) to RGB for face_recognition
        try:
            rgb_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        except Exception:
            return None
        # Detect face locations in the crop
        face_locs = face_recognition.face_locations(rgb_crop)
        if not face_locs:
            return None
        
        encs = face_recognition.face_encodings(rgb_crop, known_face_locations=face_locs)
        if not encs:
            return None
        
        emb = encs[0]
        for entry in self.memory:
            sim = np.dot(entry["embedding"], emb) / (np.linalg.norm(entry["embedding"]) * np.linalg.norm(emb))
            if sim > self.threshold:
                entry["last_seen"] = time.time()
                return entry["id"]
        
        new_id = self.next_id
        self.memory.append({
            "id": new_id,
            "embedding": emb,
            "label": f"unknown_{new_id}",
            "last_seen": time.time()
        })
        self.next_id += 1
        return new_id