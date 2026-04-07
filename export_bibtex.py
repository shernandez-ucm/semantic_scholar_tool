#!/usr/bin/env python3
"""
Script to export papers from JSONL format to BibTeX format.
"""

import json
import argparse
import sys
import re

def sanitize_bibtex_key(title, paper_id):
    """Generate a unique BibTeX key from title and paper_id."""
    # Use paper ID as key for uniqueness, with title as fallback
    if paper_id:
        return paper_id[:12]
    # Fallback: use first few words of title
    words = re.findall(r'\w+', title.lower())
    return ''.join(words[:3]) if words else 'paper'

def format_authors(authors_list):
    """Format authors list for BibTeX."""
    if not authors_list:
        return ""
    author_names = [author.get('name', '') for author in authors_list]
    return " and ".join(author_names)

def escape_bibtex(text):
    """Escape special characters for BibTeX."""
    if not text:
        return ""
    # Replace special characters that need escaping in BibTeX
    text = text.replace('&', r'\&')
    text = text.replace('$', r'\$')
    text = text.replace('#', r'\#')
    text = text.replace('%', r'\%')
    text = text.replace('_', r'\_')
    # Handle quotes
    text = text.replace('"', '``')
    return text

def export_to_bibtex(input_file, output_file):
    """
    Convert JSONL papers to BibTeX format.
    
    Args:
        input_file (str): Path to input JSONL file
        output_file (str): Path to output BibTeX file
    """
    bibtex_entries = []
    
    with open(input_file, 'r', encoding='utf-8') as infile:
        for line_num, line in enumerate(infile, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                paper = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON on line {line_num}: {e}", file=sys.stderr)
                continue
            
            # Extract fields
            key = sanitize_bibtex_key(paper.get('title', ''), paper.get('paperId', ''))
            title = escape_bibtex(paper.get('title', ''))
            authors = format_authors(paper.get('authors', []))
            year = paper.get('year', '')
            abstract = escape_bibtex(paper.get('abstract', ''))
            citation_count = paper.get('citationCount', 0)
            
            # Build BibTeX entry
            entry = f"@article{{{key},\n"
            
            if title:
                entry += f"  title = {{{title}}},\n"
            
            if authors:
                entry += f"  author = {{{authors}}},\n"
            
            if year:
                entry += f"  year = {{{year}}},\n"
            
            if abstract:
                entry += f"  abstract = {{{abstract}}},\n"
            
            if citation_count:
                entry += f"  citationcount = {{{citation_count}}},\n"
            
            # Add paper ID as custom field
            if paper.get('paperId'):
                entry += f"  paperid = {{{paper.get('paperId')}}},\n"
            
            # Add open access info if available
            pdf_info = paper.get('openAccessPdf', {})
            if isinstance(pdf_info, dict):
                if pdf_info.get('url'):
                    entry += f"  url = {{{pdf_info.get('url')}}},\n"
                if pdf_info.get('status'):
                    entry += f"  accessstatus = {{{pdf_info.get('status')}}},\n"
                if pdf_info.get('license'):
                    entry += f"  license = {{{pdf_info.get('license')}}},\n"
            
            # Remove trailing comma and newline, then close
            entry = entry.rstrip(',\n') + "\n"
            entry += "}\n\n"
            
            bibtex_entries.append(entry)
    
    # Write all entries to output file
    if output_file == '-':
        sys.stdout.write(''.join(bibtex_entries))
    else:
        with open(output_file, 'w', encoding='utf-8') as outfile:
            outfile.write(''.join(bibtex_entries))
        print(f"Successfully exported {len(bibtex_entries)} papers to {output_file}", file=sys.stderr)

def main():
    parser = argparse.ArgumentParser(description='Export papers from JSONL to BibTeX format.')
    parser.add_argument('--input', '-i', required=True, help='Input JSONL file path')
    parser.add_argument('--output', '-o', default='-', help='Output BibTeX file path (use "-" for stdout)')
    
    args = parser.parse_args()
    
    export_to_bibtex(args.input, args.output)

if __name__ == '__main__':
    main()
