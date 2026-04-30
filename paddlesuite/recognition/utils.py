import os
import pickle
import numpy as np
import faiss


class VectorSearcher:
    """Wrapper for Faiss-based vector search with HNSW32 or other indices"""

    def __init__(self, index_config):
        self.index_config = index_config
        if not self.index_config:
            self.enabled = False
            return

        self.enabled = True
        self.index_dir = self.index_config.get("index_dir")
        self.return_k = self.index_config.get("return_k", 5)
        self.score_thres = self.index_config.get("score_thres", 0.5)
        self.dist_type = self.index_config.get("dist_type", "IP")

        # Load Faiss index
        index_path = os.path.join(self.index_dir, "vector.index")
        id_map_path = os.path.join(self.index_dir, "id_map.pkl")

        if not os.path.exists(index_path) or not os.path.exists(id_map_path):
            self.enabled = False
            return

        try:
            if self.dist_type == "hamming":
                self.index = faiss.read_index_binary(index_path)
            else:
                self.index = faiss.read_index(index_path)

            with open(id_map_path, "rb") as f:
                self.id_map = pickle.load(f)
        except Exception:
            self.enabled = False

    def search(self, query_features, return_k=None):
        if not self.enabled:
            return None

        if isinstance(query_features, list):
            query_features = np.array(query_features, dtype=np.float32)

        k = return_k or self.return_k

        if self.dist_type == "hamming":
            distances, indices = self.index.search(query_features, k)
        else:
            distances, indices = self.index.search(query_features, k)

        results = []
        for dist_row, idx_row in zip(distances, indices):
            top_results = []
            for dist, idx in zip(dist_row, idx_row):
                if idx == -1:
                    break

                if self.dist_type == "hamming":
                    score_valid = dist <= self.score_thres
                else:
                    score_valid = dist >= self.score_thres

                if score_valid:
                    doc_info = self.id_map.get(idx, f"unknown_{idx}")
                    top_results.append(
                        {
                            "id": int(idx),
                            "score": float(dist),
                            "doc": doc_info,
                        }
                    )

            results.append(top_results)

        return results
