"""
Semantic Scholar Advanced Search & Recommendation Tool for OpenWebUI

This tool enables LLM-driven literature searches, bulk dataset retrieval, and
paper recommendations using the Semantic Scholar Academic Graph API.
It implements a manual retry mechanism to handle rate limits and server errors
without external dependencies like urllib3.

Author: Gemini
Version: 1.2.0
"""

import os
import requests
from datetime import datetime
import json
from dotenv import load_dotenv

class Tools:
    def __init__(self,persist_file: str = f"papers.jsonl"):
        load_dotenv()  # Load environment variables from .env file
        self.api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
        self.base_url = "https://api.semanticscholar.org/graph/v1/paper/search"
        self.headers = {'X-API-KEY': self.api_key}
        self.persist_file = persist_file  # File to store retrieved papers

    # Add your custom tools using pure Python code here, make sure to add type hints and descriptions

    def search_bulk_papers(self,
                    topic: str = "",
                    sort: str = "citationCount:desc",
                    year: str = "2023-",
                    fields: str = "title,year,abstract,citationCount",
                    fieldsOfStudy:str = "Computer Science,Environmental Science",
                    publicationTypes:str = "JournalArticle,Conference,Review",
                    minCitationCount:int = 10) -> str:
        """
        Find bulk papers related to a given topic.
        """
        params = {
            "query": topic,
            "sort": sort,  # Optional: Sort by relevance, citationCount, influentialCitationCount, etc.
            "fields": fields,  # Optional: Specify the fields you want in the response
            "year": year,
            "sort": sort,
            "fieldsOfStudy": fieldsOfStudy,
            "publicationTypes": publicationTypes,
            "minCitationCount": minCitationCount
        }
        response = requests.get(self.base_url+"/bulk",headers=self.headers,params=params) 
        results = response.json()
        #print(f"Will retrieve an estimated {results['total']} documents")
        retrieved = 0
        with open(self.persist_file, "a") as file:
            while True:
                if "data" in results:
                    retrieved += len(results["data"])
                    print(f"Retrieved {retrieved} papers...")
                    for paper in results["data"]:
                        print(json.dumps(paper), file=file)
                if "token" not in results:
                    break
                params["token"] = results["token"]
                response=requests.get(self.base_url+"/bulk",headers=self.headers,params=params)
                results = response.json() 
        print(f"Done! Retrieved {retrieved} papers total")

    def find_papers(self, 
                    topic: str = "",
                    result_limit: int = 10,
                    sort: str = "citationCount:desc",
                    fieldsOfStudy:str = "Computer Science,Environmental Science") -> str:
        """
        Find a limited number of papers related to a given topic.
        """
        params = {
            "query": topic,
            "limit": result_limit,  # Optional: Limit the number of results returned
            "fields": "title,url,abstract",  # Optional: Specify the fields you want in the response
            "fieldsOfStudy": fieldsOfStudy,  # Optional: Specify the field of study
            "sort": sort  # Optional: Sort by relevance, citationCount, influentialCitationCount, etc.
        }
        try:
            response = requests.get(self.base_url,headers=self.headers,params=params)
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx and 5xx)
            results = response.json()
            papers = results.get('data', [])
            if not papers:
                return "No papers found for the given topic."
            paper_texts = []
            for idx, paper in enumerate(papers, start=1):
                title = paper.get('title', 'No title')
                url = paper.get('url', 'No URL')
                abstract = paper.get('abstract', 'No abstract')
                paper_texts.append(
                    f"Paper {idx}: {title}\nURL: {url}\nAbstract: {abstract}\n-----------"
                )

            return "\n\n".join(paper_texts)
        except requests.RequestException as e:
            return f"Error fetching paper data: {str(e)}"

