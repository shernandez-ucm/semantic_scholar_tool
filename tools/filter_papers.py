#!/usr/bin/env python3
"""
Script to filter papers from a JSONL file based on minimum citation count and presence of abstract.
"""

import json
import argparse
import sys

def filter_papers(input_file, output_file, min_citations):
    """
    Filter papers from input JSONL file and write to output JSONL file.
    
    Args:
        input_file (str): Path to input JSONL file
        output_file (str): Path to output JSONL file (use '-' for stdout)
        min_citations (int): Minimum citation count threshold
    """
    with open(input_file, 'r', encoding='utf-8') as infile:
        if output_file == '-':
            outfile = sys.stdout
        else:
            outfile = open(output_file, 'w', encoding='utf-8')
        
        try:
            for line_num, line in enumerate(infile, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    paper = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping invalid JSON on line {line_num}: {e}", file=sys.stderr)
                    continue
                
                # Check citation count
                citation_count = paper.get('citationCount', 0)
                if citation_count < min_citations:
                    continue
                
                # Check abstract
                abstract = paper.get('abstract')
                if abstract is None or abstract.strip() == '':
                    continue
                
                # Remove disclaimer from openAccessPdf if present
                if 'openAccessPdf' in paper and isinstance(paper['openAccessPdf'], dict) and 'disclaimer' in paper['openAccessPdf']:
                    del paper['openAccessPdf']['disclaimer']
                
                # Write the filtered paper
                json.dump(paper, outfile, ensure_ascii=False)
                outfile.write('\n')
                
        finally:
            if output_file != '-':
                outfile.close()

def main():
    parser = argparse.ArgumentParser(description='Filter papers from JSONL file based on citation count and abstract presence.')
    parser.add_argument('--input', '-i', required=True, help='Input JSONL file path')
    parser.add_argument('--output', '-o', default='-', help='Output JSONL file path (use "-" for stdout)')
    parser.add_argument('--min-citations', '-c', type=int, default=0, help='Minimum citation count threshold (default: 0)')
    
    args = parser.parse_args()
    
    filter_papers(args.input, args.output, args.min_citations)

if __name__ == '__main__':
    main()