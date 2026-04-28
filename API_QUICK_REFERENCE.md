# Quick Reference: Chat & Wiki API

## Initialization

```python
from semantic_scholar_rag import ResearchPaperSearch

# Basic initialization
search_engine = ResearchPaperSearch()

# With custom models
search_engine = ResearchPaperSearch(
    model_name="nomic-embed-text:latest",      # Embedding model
    llm_model="mistral:latest",                # Chat/reasoning model
    persist_directory="./chroma_db",
    batch_size=1000
)
```

## Search

```python
# Search for papers
papers = search_engine.search(
    query="your research topic",
    k=5  # Number of results
)

# Returns: List[Dict] with keys: 'title', 'abstract', 'year', 'id'
```

## Chat API

### Single-turn Chat
```python
response = search_engine.chat(
    query="Your question here",
    search_results=papers
)
```

### Multi-turn Conversation
```python
history = []

# Turn 1
response1 = search_engine.chat("First question", papers)
history.append({"role": "user", "content": "First question"})
history.append({"role": "assistant", "content": response1})

# Turn 2 (maintains context from turn 1)
response2 = search_engine.chat("Follow-up question", papers, history)
history.append({"role": "user", "content": "Follow-up question"})
history.append({"role": "assistant", "content": response2})
```

## Wiki Generation

### Generate Complete Wiki
```python
wiki_text = search_engine.generate_wiki(
    query="your topic",
    k=5,
    output_file="output.md"  # Optional
)
```

### Extract Structured Data
```python
wiki_data = search_engine.extract_wiki_content(papers)

# Access by category
wiki_data["claims"]          # Main findings
wiki_data["concepts"]        # Key concepts
wiki_data["methods"]         # Methodologies
wiki_data["materials"]       # Materials/resources
wiki_data["datasets"]        # Datasets
wiki_data["tools"]           # Software tools
wiki_data["results"]         # Results
wiki_data["open_questions"]  # Future work
```

## Data Structures

### Paper Object
```python
{
    "title": str,           # Paper title
    "abstract": str,        # Paper abstract
    "year": int,           # Publication year
    "id": str              # Paper ID
}
```

### Chat History
```python
[
    {"role": "user", "content": "question"},
    {"role": "assistant", "content": "answer"},
    {"role": "user", "content": "follow-up"},
    {"role": "assistant", "content": "response"}
]
```

### Wiki Data Structure
```python
{
    "claims": [
        {
            "content": str,           # The claim/finding
            "source_paper": str,      # Paper title
            "year": int,             # Publication year
            "paper_id": str          # Paper ID
        },
        # ... more items
    ],
    "concepts": [...],      # Similar structure
    "materials": [...],     # Similar structure
    "methods": [...],       # Similar structure
    "results": [...],       # Similar structure
    "datasets": [...],      # Similar structure
    "tools": [...],         # Similar structure
    "open_questions": [...]  # Similar structure
}
```

## Common Patterns

### Research Questions & Answers
```python
papers = search_engine.search("your topic", k=5)

questions = [
    "What are the main findings?",
    "What methods are used?",
    "What are the limitations?",
    "What future work is needed?"
]

for q in questions:
    answer = search_engine.chat(q, papers)
    print(f"Q: {q}\nA: {answer}\n")
```

### Compare Topics
```python
papers1 = search_engine.search("topic A", k=5)
papers2 = search_engine.search("topic B", k=5)

comparison = search_engine.chat(
    "Compare the approaches in these papers",
    papers1 + papers2
)
```

### Extract Key Information
```python
papers = search_engine.search("your topic", k=5)
wiki_data = search_engine.extract_wiki_content(papers)

# Get all tools
tools = [item['content'] for item in wiki_data['tools']]

# Get datasets
datasets = [item['content'] for item in wiki_data['datasets']]

# Get all results
results = [item['content'] for item in wiki_data['results']]
```

### Save Multi-topic Wiki
```python
topics = ["topic1", "topic2", "topic3"]

for topic in topics:
    wiki = search_engine.generate_wiki(
        topic,
        k=5,
        output_file=f"{topic}_wiki.md"
    )
```

## Configuration Examples

### Speed-optimized
```python
search_engine = ResearchPaperSearch(
    model_name="nomic-embed-text:latest",
    llm_model="mistral:7b",      # Fast model
    batch_size=500
)
```

### Quality-optimized
```python
search_engine = ResearchPaperSearch(
    model_name="nomic-embed-text:latest",
    llm_model="llama2:70b",      # Higher quality
    batch_size=100
)
```

### Creative Output
```python
search_engine = ResearchPaperSearch(llm_model="mistral:latest")
search_engine.llm.temperature = 0.7  # More creative
```

## Available LLM Models
- `mistral:latest` - Fast, balanced
- `llama2:latest` - Larger, more capable
- `neural-chat:latest` - Optimized for chat
- `orca-mini:latest` - Good reasoning
- `dolphin-mixtral:latest` - Multi-domain

## Output Examples

### Wiki Section Format
```markdown
## CLAIMS & FINDINGS
- **Claim text here** (Source: Paper Title, 2023)
- **Another finding** (Source: Another Paper, 2022)

## METHODOLOGIES
- Methodology description (Source: Paper Title, 2023)
```

### Chat Response Format
Conversational, contextual responses based on provided papers.

## Limits

- Chat context window: Limited by Ollama (typically 4k-32k tokens)
- Wiki extraction: Handles papers with good abstracts best
- Processing time: Depends on model size and number of papers
