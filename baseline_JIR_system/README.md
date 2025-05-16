- docker build .
- docker run -d -p 9200:9200 <HASH>
- python3 index_opensearch.py (takes ~ 1 hour)
- python3 run_baseline --gemini_key <GEMINI_KEY> --ref_only true


# TODO update paths
# TODO update paths in run_baseline_stream