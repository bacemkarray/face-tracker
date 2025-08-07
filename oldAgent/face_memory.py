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