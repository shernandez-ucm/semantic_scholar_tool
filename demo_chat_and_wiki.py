"""
Demo script showing how to use the new chat and wiki generation features
from the ResearchPaperSearch class.

Features demonstrated:
- Single-turn chat with research context
- Multi-turn conversations
- Structured wiki generation with categories
"""

"""
Text query that will be matched against the paper's title and abstract. All terms are stemmed in English. By default all terms in the query must be present in the paper.

The match query supports the following syntax:

+ for AND operation
| for OR operation
- negates a term
" collects terms into a phrase
* can be used to match a prefix
( and ) for precedence
~N after a word matches within the edit distance of N (Defaults to 2 if N is omitted)
~N after a phrase matches with the phrase terms separated up to N terms apart (Defaults to 2 if N is omitted)
Examples:

fish ladder matches papers that contain "fish" and "ladder"
fish -ladder matches papers that contain "fish" but not "ladder"
fish | ladder matches papers that contain "fish" or "ladder"
"fish ladder" matches papers that contain the phrase "fish ladder"
(fish ladder) | outflow matches papers that contain "fish" and "ladder" OR "outflow"
fish~ matches papers that contain "fish", "fist", "fihs", etc.
"fish ladder"~3 mathces papers that contain the phrase "fish ladder" or "fish is on a ladder"
"""
from tools.semantic_scholar_tool import Tools
from tools.semantic_scholar_rag import ResearchPaperSearch

def main():
    #search_tool = Tools("papers.jsonl")
    #search_tool.search_bulk_papers(topic="(causal inference) + education",fieldsOfStudy="")
    # Initialize the search engine
    search_engine = ResearchPaperSearch(
        embedding_model="qwen3-embedding:0.6b",
        persist_directory="./chroma_db",
        llm_model="gemma4:latest"  # You can use other models like "neural-chat", "llama2", etc.
    )
    search_engine.ingest_jsonl("papers.jsonl")
    # ============================================
    # EXAMPLE 1: Basic Search and Chat
    # ============================================
    print("=" * 60)
    print("EXAMPLE 1: Basic Search and Chat")
    print("=" * 60)
    
    query = "What are the latest advancements in causal inference in education?"
    print(f"\nSearching for: {query}")
    
    # Search for relevant papers
    papers = search_engine.search(query, k=3)
    print(f"Found {len(papers)} papers\n")
    
    # Ask a question about the results
    question = "What are the main challenges in causal inference?"
    print(f"Question: {question}\n")
    response = search_engine.chat(question, papers)
    print(f"Answer:\n{response}\n")

    # ============================================
    # EXAMPLE 2: Multi-turn Conversation
    # ============================================
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Multi-turn Conversation")
    print("=" * 60)
    
    # Initialize chat history with first exchange
    chat_history = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": response}
    ]
    
    # Ask a follow-up question
    followup1 = "What datasets are commonly used for training these models?"
    print(f"\nFollow-up Question 1: {followup1}\n")
    followup_response1 = search_engine.chat(followup1, papers, chat_history)
    print(f"Answer:\n{followup_response1}\n")
    
    # Add to history and ask another follow-up
    chat_history.append({"role": "user", "content": followup1})
    chat_history.append({"role": "assistant", "content": followup_response1})
    
    followup2 = "How do results compare and what metrics are used ?"
    print(f"Follow-up Question 2: {followup2}\n")
    followup_response2 = search_engine.chat(followup2, papers, chat_history)
    print(f"Answer:\n{followup_response2}\n")

    # ============================================
    # EXAMPLE 3: Structured Wiki Generation
    # ============================================
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Structured Wiki Generation")
    print("=" * 60)
    
    wiki_topic = "Causal Inference in Education"
    print(f"\nGenerating wiki for: {wiki_topic}")
    print("This will extract and organize key information from papers...\n")
    
    # Generate wiki and save to file
    wiki = search_engine.generate_wiki(
        query=wiki_topic,
        k=5,  # Analyze top 5 papers
        output_file="research_wiki.md"
    )
    
    print("Wiki preview (first 1000 chars):\n")
    print(wiki[:1000])
    print("\n... (full wiki saved to 'research_wiki.md')\n")

    
    print("\nStarting interactive chat session...")
    print("Type 'quit' to exit, 'new' to search new topic, 'wiki' to generate wiki\n")
    
    current_papers = None
    history = []
    current_query = "causal inference in education"
    
    while True:
        # Search if no papers loaded
        if current_papers is None:
            print(f"\nSearching for: {current_query}")
            current_papers = search_engine.search(current_query, k=3)
            print(f"Loaded {len(current_papers)} papers")
            history = []
        
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() == 'quit':
            print("Goodbye!")
            break
        elif user_input.lower() == 'new':
            current_query = input("Enter new search topic: ").strip()
            current_papers = None
            continue
        elif user_input.lower() == 'wiki':
            output_file = input("Enter output filename (default: wiki.md): ").strip() or "wiki.md"
            search_engine.generate_wiki(current_query, k=len(current_papers), output_file=output_file)
            continue
        elif not user_input:
            continue
        
        # Chat with papers
        response = search_engine.chat(user_input, current_papers, history)
        print(f"\nAssistant: {response}")
        
        # Add to history
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
