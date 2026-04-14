import sys
sys.path.insert(0, ".")  # run from project root

from search import PubMedSearchTool
from extraction import EvidenceExtractor
import torch

search = PubMedSearchTool()
extractor = EvidenceExtractor()

CLAIM = (
    "Abemaciclib is effective for treating pediatric high-grade glioma."
)

QUERY = 'abemaciclib monotherapy'

# Search query
print(f"\nSearching PubMed: {QUERY!r}")
pmids = search.search(QUERY, max_results=10)
print(f"PMIDs: {pmids}\n")
if not pmids:
    print("No PMIDs returned — check network or query.")

# Semantic scholar get details
papers = search.fetch_details(pmids)
print(f"Retrieved {len(papers)} papers from Semantic Scholar\n")

# Extract evidence from papers
all_evidence = []
for paper in papers:
    text = paper.get("abstract", "").strip()
    if not text:
        print(f"  [SKIP] No abstract: {paper.get('title', '?')[:60]}")
        continue

    items = extractor.extract(claim=CLAIM, text=text, paper=paper)
    all_evidence.extend(items)

    print(f"\n{paper['title'][:70]}")
    print(f"  {paper['authors'][:50]} ({paper['year']})")
    if items:
        for item in items:
            print(f"  [{item.stance.upper():13s} {item.confidence:.2f}] {item.statement[:100]}")
    else:
        print("  (no relevant evidence found)")

# Print summary
sup  = [e for e in all_evidence if e.stance == "supporting"]
con  = [e for e in all_evidence if e.stance == "contradicting"]
neu  = [e for e in all_evidence if e.stance == "neutral"]
print(f"\n{'='*60}")
print(f"Total: {len(all_evidence)}  |  Supporting: {len(sup)}  |  Contradicting: {len(con)}  |  Neutral: {len(neu)}")

