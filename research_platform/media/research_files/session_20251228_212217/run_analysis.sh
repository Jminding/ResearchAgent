#!/bin/bash
cd /Users/jminding/Desktop/Code/Research Agent/research_platform/media/research_files/session_20251228_212217
python3 analyze_revision.py > analysis_output.txt 2>&1
cat analysis_output.txt
