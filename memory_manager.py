import random
import hashlib
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# =====================================================================
# 1. LOCAL SEMANTIC GRAPH STORE (Neo4j Fallback & Local Simulator)
# =====================================================================

class LocalGraphStore:
    """
    A lightweight, in-memory semantic graph database simulator.
    Mirrors Neo4j's nodes-and-relationships model for self-contained testing.
    """
    def __init__(self):
        self.nodes: Dict[str, Dict[str, Any]] = {}  # entity_id -> {label, properties}
        self.edges: List[Dict[str, Any]] = []      # list of {"source": ..., "target": ..., "type": ..., "properties": ...}

    def add_node(self, entity_id: str, label: str, properties: Dict[str, Any]) -> None:
        key = entity_id.lower().strip()
        self.nodes[key] = {
            "label": label,
            "properties": properties
        }

    def add_relationship(self, source: str, target: str, rel_type: str, properties: Dict[str, Any]) -> None:
        src_key = source.lower().strip()
        tgt_key = target.lower().strip()
        
        # Auto-create nodes if missing
        if src_key not in self.nodes:
            self.add_node(source, "Entity", {"name": source})
        if tgt_key not in self.nodes:
            self.add_node(target, "Entity", {"name": target})
            
        self.edges.append({
            "source": src_key,
            "target": tgt_key,
            "type": rel_type.upper().strip(),
            "properties": properties
        })

    def query_relationships(self, keyword: str) -> List[str]:
        """
        Retrieves matching semantic paths based on entity name.
        Allows the Supervisor to dynamically query user preferences and relations.
        """
        key_clean = keyword.lower().strip()
        results = []
        for edge in self.edges:
            src = edge["source"]
            tgt = edge["target"]
            etype = edge["type"]
            if key_clean in src or key_clean in tgt:
                # Format into a clean, human-readable semantic fact
                src_display = self.nodes[src]["properties"].get("name", src.capitalize())
                tgt_display = self.nodes[tgt]["properties"].get("name", tgt.capitalize())
                results.append(f"Semantic Relation: {src_display} - [{etype}] -> {tgt_display}")
        return results

# Singleton instance of the local graph store representing global semantic memory
graph_store = LocalGraphStore()


# =====================================================================
# 2. EPISODIC VECTOR MEMORY (Qdrant Client Wrapper)
# =====================================================================

COLLECTION_NAME = "episodic_memories"

# Initialize Qdrant Client in-memory for zero-latency local isolation
qdrant_client = QdrantClient(":memory:")

# Bootstrap vector collection schema
try:
    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )
except Exception:
    pass

def _get_stable_embedding(text: str) -> List[float]:
    """
    Generates a deterministic vector of size 384 based on the string content.
    Provides local vector similarity searching without requiring external downloads or keys.
    """
    seed = int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16) % 10**8
    rng = random.Random(seed)
    return [rng.uniform(-1, 1) for _ in range(384)]


# =====================================================================
# 3. BACKGROUND MEMORY MANAGER AGENT
# =====================================================================

class MemoryManagerAgent:
    """
    Asynchronous Memory Manager Agent that extracts semantic relationships
    for the Graph DB (Neo4j) and logs raw episodic summaries into the Vector DB (Qdrant).
    """
    
    @staticmethod
    def extract_and_store_memories(user_id: str, completed_tasks: List[Dict[str, Any]], facts: List[str]) -> None:
        """
        Called when a session/run concludes. It extracts long-term insights and logs them.
        """
        if not facts:
            return
            
        print(f"\n[Memory Manager] Initiating background memory extraction for user: {user_id}...")
        
        # 1. Epissodic Memory (Vector DB): Summarize the session and save to Qdrant
        raw_summary = f"Session completed. Facts logged: {'; '.join(facts)}"
        vector = _get_stable_embedding(raw_summary)
        
        point_id = int(hashlib.md5(raw_summary.encode("utf-8")).hexdigest(), 16) % 10**12
        qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "user_id": user_id,
                        "summary": raw_summary,
                        "facts": facts
                    }
                )
            ]
        )
        print(" - Episodic memory saved to Qdrant.")
        
        # 2. Semantic Memory (Graph DB): Extract relationships and store in LocalGraphStore
        # Here we extract core entities/preferences dynamically
        for fact in facts:
            fact_lower = fact.lower()
            if "blueprint" in fact_lower or "house" in fact_lower:
                graph_store.add_relationship("User", "Modern House", "Prefers", {"style": "Modern Architecture"})
                graph_store.add_relationship("User", "blueprints.txt", "Generated", {"type": "layout"})
            if "recipe" in fact_lower or "lasagna" in fact_lower:
                graph_store.add_relationship("User", "Lasagna", "Prefers", {"cuisine": "Italian"})
                graph_store.add_relationship("User", "recipe.txt", "Generated", {"type": "culinary"})
                
        print(" - Semantic relations extracted and stored in Graph Store.")

    @staticmethod
    def retrieve_memories(user_id: str, query: str) -> List[str]:
        """
        Queries both Vector (Qdrant) and Graph stores to construct rich, cross-session working memory.
        """
        retrieved_facts = []
        
        print(f"\n[Memory Manager] Querying global memory stores for search query: '{query}'...")
        
        # 1. Search Vector DB
        query_vector = _get_stable_embedding(query)
        try:
            response = qdrant_client.query_points(
                collection_name=COLLECTION_NAME,
                query=query_vector,
                limit=2
            )
            search_results = response.points
        except Exception as e:
            print(f"Error querying Qdrant: {e}")
            search_results = []
        
        for res in search_results:
            payload = res.payload
            if payload and payload.get("user_id") == user_id:
                for f in payload.get("facts", []):
                    retrieved_facts.append(f"Historical Epissodic Fact: {f}")
                    
        # 2. Search Graph DB
        # Look for keywords in the query to trigger semantic graph retrievals
        keywords = ["house", "lasagna", "blueprint", "recipe", "cuisine", "architecture"]
        query_lower = query.lower()
        for kw in keywords:
            if kw in query_lower:
                graph_relations = graph_store.query_relationships(kw)
                for rel in graph_relations:
                    retrieved_facts.append(rel)
                    
        # Remove duplicates
        retrieved_facts = list(set(retrieved_facts))
        
        print(f" - Successfully retrieved {len(retrieved_facts)} historical facts from global memory.")
        return retrieved_facts
