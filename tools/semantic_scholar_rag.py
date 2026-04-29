import json
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from uuid import uuid4
import tqdm
from typing import List, Dict, Any
from datetime import datetime

class ResearchPaperSearch:
    def __init__(self, embedding_model="nomic-embed-text:latest", persist_directory="./chroma_db",batch_size=1000, llm_model="mistral:latest" ):
        # Initialize Ollama embeddings
        self.embeddings = OllamaEmbeddings(model=embedding_model,
                                           num_ctx=8182)
        self.llm = OllamaLLM(model=llm_model, temperature=0.0)
        self.persist_directory = persist_directory
        self.vector_store = None
        self.batch_size = batch_size

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

        batch_size = self.batch_size
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

    def chat(self, query: str, search_results: List[Dict[str, Any]], chat_history: List[Dict[str, str]] = None) -> str:
        """
        Chat with the Ollama LLM using search results as context.
        
        Args:
            query: The user's question
            search_results: Results from the search() method
            chat_history: Optional list of previous messages [{"role": "user"/"assistant", "content": "..."}]
        
        Returns:
            The LLM's response
        """
        if chat_history is None:
            chat_history = []
        
        # Format search results as context
        context = "RESEARCH CONTEXT:\n"
        for i, paper in enumerate(search_results, 1):
            context += f"\n[Paper {i}] {paper['title']} ({paper['year']})\n"
            context += f"Abstract: {paper['abstract'][:500]}...\n"
        
        # Build prompt with chat history
        prompt_text = ""
        for msg in chat_history:
            if msg["role"] == "user":
                prompt_text += f"User: {msg['content']}\n"
            elif msg["role"] == "assistant":
                prompt_text += f"Assistant: {msg['content']}\n"
        
        full_prompt = f"""{context}

CONVERSATION HISTORY:
{prompt_text}

Current Question: {query}

Based on the research papers provided above, answer the question comprehensively. Reference specific papers when relevant."""
        
        response = self.llm.invoke(full_prompt)
        return response

    def extract_wiki_content(self, papers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract structured wiki content from papers using Ollama.
        
        Args:
            papers: List of paper data from search() or similar
        
        Returns:
            Dictionary with extracted wiki content organized by category
        """
        wiki_data = {
            "claims": [],
            "concepts": [],
            "materials": [],
            "methods": [],
            "results": [],
            "datasets": [],
            "tools": [],
            "open_questions": []
        }
        
        for paper in papers:
            paper_text = f"Title: {paper['title']}\nAbstract: {paper['abstract']}"
            
            extraction_prompt = f"""Analyze this research paper and extract information in JSON format:

Paper: {paper_text}

Extract and return ONLY a valid JSON object (no markdown, no code blocks) with these categories:
{{
    "claims": [list of main claims or findings],
    "concepts": [key concepts and definitions mentioned],
    "materials": [materials, resources, or datasets used],
    "methods": [methodologies or techniques employed],
    "results": [key results and observations],
    "datasets": [specific datasets or data sources],
    "tools": [software tools or frameworks used],
    "open_questions": [questions for future research]
}}

Provide brief, concise entries. Return ONLY the JSON object."""
            
            try:
                response = self.llm.invoke(extraction_prompt)
                # Parse JSON response
                extracted = json.loads(response.strip())
                
                # Add paper reference to each item
                for category in wiki_data:
                    if category in extracted and extracted[category]:
                        for item in extracted[category]:
                            wiki_data[category].append({
                                "content": item,
                                "source_paper": paper['title'],
                                "year": paper['year'],
                                "paper_id": paper['id']
                            })
            except json.JSONDecodeError:
                print(f"Warning: Could not parse JSON for paper '{paper['title']}'")
                continue
        
        return wiki_data

    def generate_wiki(self, query: str, k: int = 5, output_file: str = None) -> str:
        """
        Generate a structured wiki from search results.
        
        Args:
            query: Search query
            k: Number of papers to retrieve
            output_file: Optional file path to save the wiki
        
        Returns:
            Formatted wiki string
        """
        # Search for papers
        papers = self.search(query, k=k)
        
        # Extract structured content
        wiki_data = self.extract_wiki_content(papers)
        
        # Format as wiki
        wiki_text = f"""# SCIENTIFIC DISCOVERY WIKI
## Topic: {query}
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Summary
{len(papers)} relevant papers analyzed.

---

## CLAIMS & FINDINGS
Main assertions and discoveries from the literature:

"""
        
        for claim in wiki_data["claims"][:10]:
            wiki_text += f"- **{claim['content']}** (Source: {claim['source_paper']}, {claim['year']})\n"
        
        wiki_text += f"""
---

## CORE CONCEPTS
Key definitions and conceptual framework:

"""
        for concept in wiki_data["concepts"][:10]:
            wiki_text += f"- **{concept['content']}** (Source: {concept['source_paper']}, {concept['year']})\n"
        
        wiki_text += f"""
---

## METHODOLOGIES
Research methods and approaches:

"""
        for method in wiki_data["methods"][:10]:
            wiki_text += f"- {method['content']} (Source: {method['source_paper']}, {method['year']})\n"
        
        wiki_text += f"""
---

## MATERIALS & RESOURCES
Datasets, tools, and resources used:

### Materials
"""
        for material in wiki_data["materials"][:10]:
            wiki_text += f"- {material['content']} (Source: {material['source_paper']}, {material['year']})\n"
        
        wiki_text += f"""
### Datasets
"""
        for dataset in wiki_data["datasets"][:10]:
            wiki_text += f"- {dataset['content']} (Source: {dataset['source_paper']}, {dataset['year']})\n"
        
        wiki_text += f"""
### Tools & Frameworks
"""
        for tool in wiki_data["tools"][:10]:
            wiki_text += f"- {tool['content']} (Source: {tool['source_paper']}, {tool['year']})\n"
        
        wiki_text += f"""
---

## KEY RESULTS
Important findings and observations:

"""
        for result in wiki_data["results"][:15]:
            wiki_text += f"- {result['content']} (Source: {result['source_paper']}, {result['year']})\n"
        
        wiki_text += f"""
---

## OPEN QUESTIONS & FUTURE WORK
Areas for further research:

"""
        for question in wiki_data["open_questions"][:10]:
            wiki_text += f"- {question['content']} (Source: {question['source_paper']}, {question['year']})\n"
        
        wiki_text += f"""
---

## SOURCE PAPERS

"""
        for i, paper in enumerate(papers, 1):
            wiki_text += f"{i}. **{paper['title']}** ({paper['year']})\n"
            wiki_text += f"   Abstract: {paper['abstract'][:300]}...\n\n"
        
        # Save to file if specified
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(wiki_text)
            print(f"Wiki saved to {output_file}")
        
        return wiki_text

# --- Usage Example ---
if __name__ == "__main__":
    search_engine = ResearchPaperSearch()

    # 1. Ingest your file (only need to do this once)
    search_engine.ingest_jsonl("papers.jsonl")

    # 2. Perform a search
    query = "Explainable AI in education"
    papers = search_engine.search(query, k=2)

    print(f"\nResults for: '{query}'\n" + "="*30)
    for i, paper in enumerate(papers, 1):
        print(f"{i}. TITLE: {paper['title']}")
        print(f"   YEAR: {paper['year']}")
        print(f"   ABSTRACT: {paper['abstract'][:300]}...") # Truncated for display
        print("-" * 30)

    # 3. Chat with the search results
    print(f"\n\nCHAT EXAMPLE\n" + "="*30)
    chat_query = "What are the main challenges in explainable AI for education?"
    response = search_engine.chat(chat_query, papers)
    print(f"Question: {chat_query}")
    print(f"Response: {response}\n")

    # 4. Multi-turn conversation
    print(f"MULTI-TURN CONVERSATION\n" + "="*30)
    chat_history = [
        {"role": "user", "content": chat_query},
        {"role": "assistant", "content": response}
    ]
    followup = "What are the proposed solutions?"
    followup_response = search_engine.chat(followup, papers, chat_history)
    print(f"Follow-up: {followup}")
    print(f"Response: {followup_response}\n")

    # 5. Generate structured wiki
    print(f"\nGENERATING STRUCTURED WIKI\n" + "="*30)
    wiki = search_engine.generate_wiki(query, k=5, output_file="wiki_output.md")
    print("Wiki preview (first 500 chars):")
    print(wiki[:500] + "...")
    print(f"\nFull wiki saved to 'wiki_output.md'")