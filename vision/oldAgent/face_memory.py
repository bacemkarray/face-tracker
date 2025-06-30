import face_recognition
import numpy as np
import time
import cv2

class FaceMemory:
    def __init__(self, threshold=0.6):
        self.memory = []
        self.next_id = 1
        self.threshold = threshold

    def match_or_add(self, face_crop):
        # Convert from BGR (OpenCV) to RGB for face_recognition
        try:
            rgb_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        except:
            return None
        # Detect face locations in the crop
        locs = face_recognition.face_locations(rgb_crop)
        if not locs:
            return None
        
        encs = face_recognition.face_encodings(rgb_crop, known_face_locations=locs)
        if not encs:
            return None
        emb = encs[0]

        known_embs = [e["embedding"] for e in self.memory]
        if known_embs:
            dists = face_recognition.face_distance(known_embs, emb)
            best = np.argmin(dists)
            if dists[best] < self.threshold:
                self.memory[best]["last_seen"] = time.time()
                return self.memory[best]["id"]

        new_id = self.next_id
        self.memory.append({
            "id": new_id,
            "embedding": emb,
            "label": f"unknown_{new_id}",
            "last_seen": time.time()
        })
        self.next_id += 1
        return new_id