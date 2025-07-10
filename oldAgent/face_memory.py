import time
import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from insightface.app import FaceAnalysis

# Necessary to ensure the YOLO box being fed in matches with the box detected by FaceAnalysis.
# Needed until I decide to isolate the face recognition from the rest of the InsightFace pipeline. 
def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0]);  yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]);  yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])
    return inter / float(areaA + areaB - inter + 1e-6)

class FaceEmbedder:
    def __init__(self, model_name: str = "buffalo_l"):
        # loads SCRFD-500MF detector + MobileFaceNet recognizer
        self.app = FaceAnalysis(name=model_name, provider=['CUDAExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))

    def get_embedding_on_frame(self, full_frame_bgr: np.ndarray, target_bbox: tuple[int,int,int,int]) -> np.ndarray | None:
        rgb = cv2.cvtColor(full_frame_bgr, cv2.COLOR_BGR2RGB)
        faces = self.app.get(rgb)
        if not faces:
            return None

        # pick the detected face whose bbox best overlaps the YOLO bbox
        best_face, best_iou = None, 0.0
        for f in faces:
            x1,y1,x2,y2 = map(int, f.bbox)
            iou = compute_iou(target_bbox, (x1,y1,x2,y2))
            if iou > best_iou:
                best_iou, best_face = iou, f

        if best_face is None or best_iou < 0.5:  # tweak threshold as needed
            return None

        return best_face.embedding


class FaceMemory:
    def __init__(self, threshold: float = 0.4, model_name: str = "buffalo_l"):
        self.memory: list[dict] = []
        self.next_id = 1
        self.threshold = threshold
        self.embedder = FaceEmbedder(model_name)

    def match_or_add(self, full_frame: np.ndarray, target_bbox: tuple[int,int,int,int]) -> int | None:
        emb = self.embedder.get_embedding_on_frame(full_frame, target_bbox)
        if emb is None:
            return None

        known = [e["embedding"] for e in self.memory]
        if known:
            sims = cosine_similarity([emb], known)[0]
            best = int(np.argmax(sims))
            if sims[best] > self.threshold:
                self.memory[best]["last_seen"] = time.time()
                return self.memory[best]["id"]

        new_id = self.next_id
        self.memory.append({
            "id": new_id,
            "embedding": emb,
            "last_seen": time.time()
        })
        self.next_id += 1
        return new_id