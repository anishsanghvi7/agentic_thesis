"""
PubMed search tool.

Uses Biopython's Entrez API — free, no key required for low-volume use.
Register an NCBI API key for higher rate limits (10 req/s vs 3 req/s).

"""

from __future__ import annotations
import os
import time
import logging
from typing import Optional

from Bio import Entrez

logger = logging.getLogger(__name__)

# NCBI requires an email address for API use
Entrez.email = os.getenv("NCBI_EMAIL", "z5421465@ad.unsw.edu.au")
Entrez.api_key = os.getenv("NCBI_API_KEY", None)  # increases rate limit

class PubMedSearchTool:
    """
    Wraps the NCBI Entrez API to search PubMed and fetch paper metadata.

    Key methods:
        search(query, max_results)  → list of PMIDs
        fetch_details(pmids)        → list of paper metadata dicts
        fetch_abstract(pmid)        → abstract text string
    """

    def __init__(self, rate_limit_delay: float = 0.4):
        """
        Args:
            rate_limit_delay: Seconds to wait between requests.
                                NCBI allows 3 req/s without a key, 10/s with one.
        """
        self.delay = rate_limit_delay if not Entrez.api_key else 0.12

    def search(self, query: str, max_results: int = 20) -> list[str]:
        """
        Search PubMed and return a list of PMIDs.

        Args:
            query: PubMed query string. Supports standard PubMed syntax
                    e.g. 'abemaciclib[tiab] AND "breast cancer"[MeSH] AND ("2018"[PDat]:"2025"[PDat])'
            max_results: Maximum number of PMIDs to return.

        Returns:
            List of PMID strings, ordered by relevance.
        """
        logger.info(f"Searching PubMed: {query!r} (max={max_results})")
        try:
            handle = Entrez.esearch(
                db="pubmed",
                term=query,
                retmax=max_results,
                sort="relevance",
                usehistory="y",
            )
            record = Entrez.read(handle)
            handle.close()
            time.sleep(self.delay)

            pmids = record.get("IdList", [])
            logger.info(f"Found {len(pmids)} results (total matching: {record.get('Count', '?')})")
            return pmids

        except Exception as e:
            logger.error(f"PubMed search failed: {e}")
            return []

    def fetch_details(self, pmids: list[str]) -> list[dict]:
        """
        Fetch paper metadata (title, authors, year, abstract, DOI) for a list of PMIDs.

        Returns:
            List of dicts with keys: pmid, title, authors, year, abstract, doi, journal
        """
        if not pmids:
            return []

        logger.info(f"Fetching details for {len(pmids)} papers")
        try:
            handle = Entrez.efetch(
                db="pubmed",
                id=",".join(pmids),
                rettype="xml",
                retmode="xml",
            )
            records = Entrez.read(handle)
            handle.close()
            time.sleep(self.delay)

            papers = []
            for article in records.get("PubmedArticle", []):
                papers.append(self._parse_article(article))
            return papers

        except Exception as e:
            logger.error(f"Entrez fetch failed: {e}")
            return []

    def fetch_abstract(self, pmid: str) -> Optional[str]:
        """
        Fetch just the abstract text for a single PMID. Quick and cheap.
        """
        details = self.fetch_details([pmid])
        if details:
            return details[0].get("abstract")
        return None

    def _parse_article(self, article: dict) -> dict:
        """Parse an Entrez PubmedArticle record into a flat dict."""
        try:
            medline = article["MedlineCitation"]
            art = medline["Article"]

            # Title
            title = str(art.get("ArticleTitle", ""))

            # Authors — "LastName FM" format
            authors_list = art.get("AuthorList", [])
            authors = []
            for a in authors_list[:6]:  # cap at 6, add et al.
                last = a.get("LastName", "")
                initials = a.get("Initials", "")
                if last:
                    authors.append(f"{last} {initials}".strip())
            author_str = ", ".join(authors)
            if len(authors_list) > 6:
                author_str += " et al."

            # Year
            pub_date = art.get("Journal", {}).get("JournalIssue", {}).get("PubDate", {})
            year = pub_date.get("Year") or pub_date.get("MedlineDate", "")[:4]
            try:
                year = int(year)
            except (ValueError, TypeError):
                year = None

            # Abstract
            abstract_obj = art.get("Abstract", {})
            abstract_texts = abstract_obj.get("AbstractText", [])
            if isinstance(abstract_texts, list):
                abstract = " ".join(str(t) for t in abstract_texts)
            else:
                abstract = str(abstract_texts)

            # DOI
            doi = None
            for id_obj in article.get("PubmedData", {}).get("ArticleIdList", []):
                if str(id_obj.attributes.get("IdType", "")) == "doi":
                    doi = str(id_obj)
                    break

            # Journal
            journal = art.get("Journal", {}).get("Title", "")

            # PMID
            pmid = str(medline.get("PMID", ""))

            return {
                "pmid": pmid,
                "title": title,
                "authors": author_str,
                "year": year,
                "abstract": abstract,
                "doi": doi,
                "journal": journal,
            }

        except Exception as e:
            logger.warning(f"Failed to parse article: {e}")
            return {}