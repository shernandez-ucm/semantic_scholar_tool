import json
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from uuid import uuid4
import tqdm

class ResearchPaperSearch:
    def __init__(self, model_name="nomic-embed-text:latest", persist_directory="./chroma_db"):
        # Initialize Ollama embeddings
        self.embeddings = OllamaEmbeddings(model=model_name)
        self.persist_directory = persist_directory
        self.vector_store = None

    def ingest_jsonl(self, file_path):
        """Loads JSONL, creates LangChain documents, and builds the vector store."""
        documents = []
        ids=set()
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                
                # Combine title and abstract for the embedding content
                # This ensures the vector search 'understands' the full context
                page_content = f"Title: {data['title']}\nAbstract: {data['abstract']}"
                
                # Keep original data in metadata for easy retrieval
                metadata = {
                    "id": data.get("paperId"),
                    "title": data.get("title"),
                    "year": data.get("year"),
                    "authors": ", ".join([a['name'] for a in data.get("authors", [])])
                }
                document=Document(page_content=page_content, metadata=metadata)
                if data.get("paperId") not in ids:
                    documents.append(document)
                    ids.add(data.get("paperId"))

        print(f"Ingesting {len(documents)} papers into the vector store...")
        
        # Create and persist the vector store
        self.vector_store = Chroma(
            collection_name="collection",
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory,
        )

        batch_size = 1000
        uuids=[str(uuid4()) for _ in range(len(documents))]
        for i in tqdm.tqdm(range(0, len(documents), batch_size)):
            batch = documents[i : i + batch_size]
            batch_uuids = uuids[i : i + batch_size]
            self.vector_store.add_documents(batch,ids=batch_uuids)
        print("Ingestion complete.")

    def search(self, query, k=3):
        """Searches for related papers and returns titles and abstracts."""
        if not self.vector_store:
            # Load existing store if not already in memory
            self.vector_store = Chroma(
                persist_directory=self.persist_directory, 
                embedding_function=self.embeddings
            )

        results = self.vector_store.similarity_search(query, k=k)
        
        search_output = []
        for doc in results:
            # Extracting title and abstract from the page_content or metadata
            # Since we stored the full text in page_content, we can parse it
            # or simply use the metadata for the title.
            title = doc.metadata.get("title")
            # We didn't store abstract in metadata to save space, 
            # so we pull it from the document text.
            abstract = doc.page_content.split("Abstract: ")[-1]
            
            search_output.append({
                "title": title,
                "abstract": abstract,
                "year": doc.metadata.get("year"),
                "id": doc.metadata.get("paperId")
            })
            
        return search_output

# --- Usage Example ---
if __name__ == "__main__":
    search_engine = ResearchPaperSearch()

    # 1. Ingest your file (only need to do this once)
    # search_engine.ingest_jsonl("papers.jsonl")

    # 2. Perform a search
    query = "Explainable AI in education"
    papers = search_engine.search(query, k=2)

    print(f"\nResults for: '{query}'\n" + "="*30)
    for i, paper in enumerate(papers, 1):
        print(f"{i}. TITLE: {paper['title']}")
        print(f"   YEAR: {paper['year']}")
        print(f"   ABSTRACT: {paper['abstract'][:300]}...") # Truncated for display
        print("-" * 30)