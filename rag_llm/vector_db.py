import chromadb
from chromadb.utils import embedding_functions
from safety_policies import SAFETY_RULES

def initialize_vector_store():
    print("Initializing ChromaDB Vector Store...")

    chroma_client = chromadb.Client()

    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    try:
        chroma_client.delete_collection("game_safety_policies")
    except Exception:
        pass

    collection = chroma_client.create_collection(
        name="game_safety_policies",
        embedding_function=embedding_func
    )

    documents = [rule["content"] for rule in SAFETY_RULES]
    metadatas = [
        {"rule_id": rule["rule_id"], "category": rule["category"]} 
        for rule in SAFETY_RULES
    ]
    ids = [rule["rule_id"] for rule in SAFETY_RULES]

    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )

    print(f"Successfully indexed {len(SAFETY_RULES)} safety rules into ChromaDB!")
    return collection


def query_top_rules(vector_store, text: str, n_results: int = 2, distance_threshold: float = 1.05):
    results = vector_store.query(
        query_texts=[text],
        n_results=n_results
    )
    
    rules = []
    if results and "documents" in results and results["documents"]:
        distances = results.get("distances", [[]])[0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]

        for doc, meta, dist in zip(documents, metadatas, distances):
            if dist <= distance_threshold:
                rules.append({
                    "content": doc,
                    "rule_id": meta.get("rule_id", "UNKNOWN"),
                    "category": meta.get("category", "General"),
                    "distance": dist
                })
    
    if not rules:
        rules.append({
            "content": "POLICY_NONE: General chat policy. Standard benign conversation requiring no safety action.",
            "rule_id": "NONE",
            "category": "Standard Clean Conversation",
            "distance": 999.0
        })

    return rules


if __name__ == "__main__":
    collection = initialize_vector_store()

    test_query_1 = "a player asking an 11 year old kid for discord and school location"
    print(f"\n--- SANITY TEST 1: '{test_query_1}' ---")
    rules_1 = query_top_rules(collection, test_query_1, n_results=1)
    print(f"Top Match: [{rules_1[0]['rule_id']} - {rules_1[0]['category']}] (Dist: {rules_1[0]['distance']:.2f})")

    test_query_2 = "shut THE FUCK UP everyone!"
    print(f"\n--- SANITY TEST 2: '{test_query_2}' ---")
    rules_2 = query_top_rules(collection, test_query_2, n_results=1)
    print(f"Top Match: [{rules_2[0]['rule_id']} - {rules_2[0]['category']}] (Dist: {rules_2[0]['distance']:.2f})")

    test_query_3 = "Hi, how's everybody doing?"
    print(f"\n--- SANITY TEST 3: '{test_query_3}' ---")
    rules_3 = query_top_rules(collection, test_query_3, n_results=1)
    print(f"Top Match: [{rules_3[0]['rule_id']} - {rules_3[0]['category']}]")
    print("-------------------------\n")