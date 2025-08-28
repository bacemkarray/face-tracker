import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ThreadPoolExecutor
from langgraph_sdk import get_sync_client
import threading


class FaceMemory:
    def __init__(self, threshold: float = 0.4):
        self.memory: list[dict] = []
        self.next_id = 1
        self.threshold = threshold
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.client = get_sync_client(url="http://localhost:8123")
        self.lock = threading.Lock()

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
        
        face_id = "???:" + str(self.next_id)
        
        known = [e["embedding"] for e in self.memory]
        if known:
            sims = cosine_similarity([emb], known)[0]
            best = int(np.argmax(sims))
            if sims[best] > self.threshold:
                return self.memory[best]["id"]

            # update langgraph for this existing face
            self.push_to_graph(face_id, emb)
            return face_id


        face_id = "???:" + str(self.next_id)
        self.memory.append({
            "id": face_id,
            "embedding": emb,
        })
        self.next_id += 1

        # push new face into Redis
        self.push_to_graph(face_id, emb)
        return face_id
    

    def rename_face(self, old_id, new_id):
        with self.lock:
            for entry in self.memory:
                if entry["id"] == old_id:
                    new_entry = dict(entry)
                    new_entry["id"] = new_id
                    self.memory.append(new_entry)
                    self.memory.remove(entry)
                    print(f"✅ Renamed {old_id} → {new_id}")
                    return True
        return False