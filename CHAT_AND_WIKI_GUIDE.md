# Chat and Wiki Generation Features

## Overview

The enhanced `ResearchPaperSearch` class now includes two major new features:

1. **Chat with Research Results**: Interactive conversations with an LLM about search results
2. **Structured Wiki Generation**: Automatic organization of research findings into a structured wiki

## Features

### 1. Chat Function

#### Basic Usage
```python
from semantic_scholar_rag import ResearchPaperSearch

# Initialize
search_engine = ResearchPaperSearch(llm_model="mistral:latest")

# Search for papers
papers = search_engine.search("machine learning in healthcare", k=5)

# Ask a question
question = "What are the main applications of ML in healthcare?"
response = search_engine.chat(question, papers)
print(response)
```

#### Multi-turn Conversations
```python
# Initialize chat history
chat_history = []

# First turn
question1 = "What are recent advances in medical imaging?"
response1 = search_engine.chat(question1, papers)
chat_history.append({"role": "user", "content": question1})
chat_history.append({"role": "assistant", "content": response1})

# Follow-up questions with context
question2 = "What about deep learning specifically?"
response2 = search_engine.chat(question2, papers, chat_history)
chat_history.append({"role": "user", "content": question2})
chat_history.append({"role": "assistant", "content": response2})

# Continue conversation...
question3 = "Which datasets are most commonly used?"
response3 = search_engine.chat(question3, papers, chat_history)
```

#### Method Signature
```python
def chat(self, 
         query: str, 
         search_results: List[Dict[str, Any]], 
         chat_history: List[Dict[str, str]] = None) -> str:
    """
    Chat with the Ollama LLM using search results as context.
    
    Args:
        query: The user's question
        search_results: Results from the search() method
        chat_history: Optional list of previous messages 
                     [{"role": "user"/"assistant", "content": "..."}]
    
    Returns:
        The LLM's response
    """
```

### 2. Structured Wiki Generation

#### Basic Usage
```python
# Generate wiki from search results
wiki = search_engine.generate_wiki(
    query="quantum computing algorithms",
    k=5,
    output_file="quantum_wiki.md"
)

# Wiki is returned as string and saved to file
print(wiki)
```

#### Wiki Categories

The generated wiki automatically organizes information into these categories:

- **Claims & Findings**: Main assertions and discoveries
- **Core Concepts**: Key definitions and conceptual frameworks
- **Methodologies**: Research methods and approaches
- **Materials & Resources**:
  - Materials: Physical or logical resources used
  - Datasets: Specific datasets or data sources
  - Tools & Frameworks: Software tools and libraries
- **Key Results**: Important findings and observations
- **Open Questions & Future Work**: Areas for further research
- **Source Papers**: Full list of analyzed papers

#### Method Signature
```python
def generate_wiki(self, 
                  query: str, 
                  k: int = 5, 
                  output_file: str = None) -> str:
    """
    Generate a structured wiki from search results.
    
    Args:
        query: Search query
        k: Number of papers to retrieve
        output_file: Optional file path to save the wiki
    
    Returns:
        Formatted wiki string
    """
```

#### Extracting Structured Content
```python
# Get raw structured data for custom processing
papers = search_engine.search("blockchain", k=5)
wiki_data = search_engine.extract_wiki_content(papers)

# Access specific categories
for claim in wiki_data["claims"]:
    print(f"{claim['content']} (from {claim['source_paper']}, {claim['year']})")

# Available keys: claims, concepts, materials, methods, results, 
#                 datasets, tools, open_questions
```

#### Method Signature
```python
def extract_wiki_content(self, 
                        papers: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Extract structured wiki content from papers using Ollama.
    
    Args:
        papers: List of paper data from search() or similar
    
    Returns:
        Dictionary with extracted wiki content organized by category
    """
```

## Configuration

### Changing the LLM Model

The `ResearchPaperSearch` class accepts different Ollama models:

```python
# Using Mistral (recommended for speed)
search_engine = ResearchPaperSearch(llm_model="mistral:latest")

# Using Llama2 (good quality)
search_engine = ResearchPaperSearch(llm_model="llama2:latest")

# Using Neural Chat (optimized for chat)
search_engine = ResearchPaperSearch(llm_model="neural-chat:latest")

# Using Orca (good reasoning)
search_engine = ResearchPaperSearch(llm_model="orca-mini:latest")
```

### Adjusting Temperature

The LLM temperature is set to 0.3 by default (more focused). To modify:

```python
# Create custom LLM instance with different temperature
from langchain_ollama import OllamaLLM

search_engine = ResearchPaperSearch()
search_engine.llm = OllamaLLM(model="mistral:latest", temperature=0.7)  # More creative
```

## Examples

### Example 1: Research Literature Review
```python
search_engine = ResearchPaperSearch()
search_engine.ingest_jsonl("papers.jsonl")

# Generate wiki for a research topic
wiki = search_engine.generate_wiki(
    "explainable AI in healthcare",
    k=10,
    output_file="xai_healthcare_wiki.md"
)

# Open the file to see structured research summary
```

### Example 2: Interactive Research Assistant
```python
papers = search_engine.search("neural networks", k=5)
history = []

while True:
    question = input("Ask about the research: ")
    response = search_engine.chat(question, papers, history)
    print(f"Answer: {response}\n")
    
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": response})
```

### Example 3: Comparative Analysis
```python
# Compare two research areas
papers1 = search_engine.search("machine learning", k=5)
papers2 = search_engine.search("deep learning", k=5)

wiki1 = search_engine.generate_wiki("machine learning", k=5, 
                                    output_file="ml_wiki.md")
wiki2 = search_engine.generate_wiki("deep learning", k=5, 
                                    output_file="dl_wiki.md")

# Ask comparative questions
comparison = search_engine.chat(
    "What are the key differences between ML and DL approaches?",
    papers1 + papers2
)
print(comparison)
```

### Example 4: Extract Datasets and Tools
```python
papers = search_engine.search("computer vision", k=5)
wiki_data = search_engine.extract_wiki_content(papers)

print("Datasets used in computer vision research:")
for dataset in wiki_data["datasets"]:
    print(f"  - {dataset['content']} ({dataset['source_paper']})")

print("\nTools and frameworks:")
for tool in wiki_data["tools"]:
    print(f"  - {tool['content']} ({dataset['source_paper']})")
```

## Advanced Usage

### Custom Prompt Engineering

You can extend the class to add custom prompts:

```python
class CustomResearchPaperSearch(ResearchPaperSearch):
    def summarize_papers(self, papers: List[Dict[str, Any]]) -> str:
        """Custom method for summarizing papers"""
        context = "\n".join([p["abstract"] for p in papers])
        prompt = f"""Summarize the following research abstracts in 3 bullet points:

{context}

Summary:"""
        return self.llm.invoke(prompt)

# Usage
search_engine = CustomResearchPaperSearch()
summary = search_engine.summarize_papers(papers)
```

### Batch Wiki Generation

```python
topics = [
    "deep learning",
    "reinforcement learning",
    "natural language processing",
    "computer vision"
]

for topic in topics:
    filename = f"{topic.replace(' ', '_')}_wiki.md"
    print(f"Generating wiki for {topic}...")
    search_engine.generate_wiki(topic, k=5, output_file=filename)
    print(f"Saved to {filename}")
```

## Performance Tips

1. **Reduce number of papers**: Fewer papers = faster processing but less comprehensive
2. **Use faster models**: `mistral:latest` is faster than `llama2:latest`
3. **Batch requests**: Generate multiple wikis/chats in sequence
4. **Cache results**: Store chat responses to avoid redundant queries
5. **Adjust k parameter**: For quick analysis use k=2-3, for comprehensive use k=10+

## Troubleshooting

### Issue: JSON parsing errors in wiki generation
**Solution**: The LLM may return malformed JSON. The code handles this gracefully, but you can improve it:
- Use a more structured model like `orca-mini`
- Increase temperature slightly (0.5 instead of 0.3)
- Provide more example papers

### Issue: Slow performance
**Solution**: 
- Use a smaller model: `mistral:7b` instead of `llama2:70b`
- Reduce k value (number of papers)
- Use fewer categories in wiki extraction

### Issue: Generic or unhelpful responses
**Solution**:
- Increase k value (search more papers)
- Use a more capable model like `llama2`
- Provide more specific search queries

## Integration with OpenWebUI

You can use these functions in OpenWebUI by adding them as tools:

```python
# In OpenWebUI tools configuration
search_engine = ResearchPaperSearch()

def research_chat(topic: str, question: str) -> str:
    """Search research and chat about results"""
    papers = search_engine.search(topic, k=5)
    return search_engine.chat(question, papers)

def research_wiki(topic: str) -> str:
    """Generate structured wiki for research topic"""
    return search_engine.generate_wiki(topic, k=5)
```

## Next Steps

- Run the demo: `python demo_chat_and_wiki.py`
- Experiment with different topics and questions
- Customize wiki categories for your domain
- Integrate with your research workflow
