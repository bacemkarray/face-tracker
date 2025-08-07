import time
import cv2
import asyncio
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from insightface.app import FaceAnalysis
from graph_invoke_test import send_embedding
import asyncio
from concurrent.futures import ThreadPoolExecutor
from langgraph_sdk import get_sync_client


# Necessary to ensure the YOLO box being fed in matches with the box detected by FaceAnalysis.
# Needed until I decide to remove YOLO or isolate face embedding from the rest of the
# InsightFace pipeline. Until then, will cause noticeable drops in performance.
def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0]);  yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]);  yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])
    return inter / float(areaA + areaB - inter + 1e-6)



class FaceMemory:
    def __init__(self, threshold: float = 0.4, model_name: str = "buffalo_l"):
        self.memory: list[dict] = []
        self.next_id = 1
        self.threshold = threshold
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.client = get_sync_client(url="http://localhost:8123")

    def push_to_graph(self, face_id: int, emb: np.ndarray):
        def task():
            print(f"📤 [sync] Storing face {face_id}")
            try:
                self.client.store.put_item(
                    ["face_embeddings"],
                    key=str(face_id),
                    value={"embedding": emb.tolist()}
                )
                print(f"✅ [sync] Stored    face {face_id}")
            except Exception as e:
                print(f"🔥 [sync] Failed to store face {face_id}: {e}")
        self.executor.submit(task)

        

    def match_or_add(self, emb: np.ndarray) -> int | None:
        if emb is None:
            return None

        known = [e["embedding"] for e in self.memory]
        if known:
            sims = cosine_similarity([emb], known)[0]
            best = int(np.argmax(sims))
            if sims[best] > self.threshold:
                return self.memory[best]["id"]

            # update langgraph for this existing face
            # self.push_to_graph(face_id, emb)
            return face_id


        face_id = self.next_id
        self.memory.append({
            "id": face_id,
            "embedding": emb,
        })
        self.next_id += 1

        # push new face into Redis
        self.push_to_graph(face_id, emb)
        return face_id