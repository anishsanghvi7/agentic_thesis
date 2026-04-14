"""
Evidence extractor — fully local, no external API calls.

Pipeline:
    1. Split text into sentences (spaCy or regex fallback)
    2. Filter to sentences relevant to the claim (keyword Jaccard)
    3. Classify each relevant sentence via a local NLI model (HuggingFace)

Model: cross-encoder/nli-MiniLM2-L6-H768  (~90 MB, CPU-friendly)

Install:
    pip install transformers torch spacy
    python -m spacy download en_core_web_sm
"""

from __future__ import annotations
import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# DEFAULT_NLI_MODEL = "cross-encoder/nli-MiniLM2-L6-H768"
DEFAULT_NLI_MODEL = "gsarti/biobert-nli" # this works with zero-shot-classification

# Map NLI output labels → our stance labels
NLI_TO_STANCE = {
    "entailment":    "supporting",
    "contradiction": "contradicting",
    "neutral":       "neutral",
}

MIN_RELEVANCE  = 0.05
MIN_CONFIDENCE = 0.30

STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "in", "to", "for", "with",
    "that", "this", "was", "were", "is", "are", "be", "by", "at",
    "from", "as", "on", "we", "our", "their", "also", "these", "those",
}

class EvidenceItem:
    def __init__(self, statement, stance, confidence, source_title,
                    source_pmid, source_year, source_authors, relevance_score=0.0):
        self.statement       = statement
        self.stance          = stance
        self.confidence      = confidence
        self.source_title    = source_title
        self.source_pmid     = source_pmid
        self.source_year     = source_year
        self.source_authors  = source_authors
        self.relevance_score = relevance_score

    def to_dict(self):
        return {
            "statement":       self.statement,
            "stance":          self.stance,
            "confidence":      round(self.confidence, 3),
            "relevance_score": round(self.relevance_score, 3),
            "source_title":    self.source_title,
            "source_pmid":     self.source_pmid,
            "source_year":     self.source_year,
            "source_authors":  self.source_authors,
        }

    def __repr__(self):
        return f"[{self.stance.upper()} {self.confidence:.2f}] {self.statement[:80]}..."

class EvidenceExtractor:
    """
    Extract stance-labelled evidence from paper text, fully locally.

    The NLI model is loaded once at construction time to avoid the
    lazy-load issues with transformers 5.x.

    Usage:
        extractor = EvidenceExtractor()   # loads model here (~5s on first run)
        items = extractor.extract(claim="...", text=paper["abstract"], paper=paper)
    """

    def __init__(
        self,
        nli_model: str = DEFAULT_NLI_MODEL,
        min_relevance: float = MIN_RELEVANCE,
        min_confidence: float = MIN_CONFIDENCE,
    ):
        self.min_relevance  = min_relevance
        self.min_confidence = min_confidence

        # Load NLI model immediately — do not use lazy loading
        logger.info(f"Loading NLI model: {nli_model}")
        try:
            from transformers import pipeline as hf_pipeline
        except ImportError:
            raise ImportError("Run: pip install transformers torch")

        self._clf = hf_pipeline(
            "zero-shot-classification",
            model=nli_model,
            device=-1,          # CPU; change to 0 for GPU
        )
        logger.info("NLI model ready.")

        # spaCy sentence splitter — optional, falls back to regex
        self._nlp = None
        try:
            import spacy
            self._nlp = spacy.load("en_core_web_sm", disable=["ner", "tagger", "parser"])
            self._nlp.enable_pipe("senter")
            logger.info("spaCy loaded.")
        except Exception:
            logger.info("spaCy not available — using regex sentence splitter.")

    # ── Public ────────────────────────────────────────────────────────────────

    def extract(self, claim: str, text: str, paper: dict) -> list[EvidenceItem]:
        if not text or not text.strip():
            return []

        sentences    = self._split_sentences(text)
        claim_tokens = self._tokenise(claim)

        results = []
        for sentence in sentences:
            # Step 1: fast keyword relevance check
            relevance = self._jaccard(self._tokenise(sentence), claim_tokens)
            if relevance < self.min_relevance:
                continue

            # Step 2: NLI classification
            stance, confidence = self._classify(claim, sentence)
            if confidence < self.min_confidence:
                continue

            results.append(EvidenceItem(
                statement=sentence,
                stance=stance,
                confidence=confidence,
                source_title=paper.get("title", ""),
                source_pmid=paper.get("pmid", ""),
                source_year=paper.get("year"),
                source_authors=paper.get("authors", ""),
                relevance_score=relevance,
            ))

        results = self._deduplicate(results)
        results.sort(key=lambda x: x.confidence, reverse=True)
        logger.info(f"  {len(results)} evidence items from '{paper.get('title','?')[:50]}'")
        return results

    # ── Private ───────────────────────────────────────────────────────────────

    def _classify(self, claim: str, sentence: str) -> tuple[str, float]:
        """
        NLI: sentence = premise (what the paper says)
                claim    = hypothesis (what we are testing)
        """
        result     = self._clf(sentence, claim, multi_label=False)
        top_label  = result["labels"][0].lower()
        top_score  = float(result["scores"][0])
        stance     = NLI_TO_STANCE.get(top_label, "neutral")
        return stance, top_score

    def _split_sentences(self, text: str) -> list[str]:
        if self._nlp is not None:
            doc = self._nlp(text)
            sentences = [s.text.strip() for s in doc.sents]
        else:
            sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text)]
        return [s for s in sentences if len(s) >= 30]

    @staticmethod
    def _tokenise(text: str) -> set[str]:
        tokens = re.findall(r"\b[a-z]{3,}\b", text.lower())
        return {t for t in tokens if t not in STOPWORDS}

    @staticmethod
    def _jaccard(a: set, b: set) -> float:
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    @staticmethod
    def _deduplicate(items: list[EvidenceItem]) -> list[EvidenceItem]:
        seen, unique = [], []
        for item in sorted(items, key=lambda x: x.confidence, reverse=True):
            tokens = set(item.statement.lower().split())
            if not any(len(tokens & s) / max(len(tokens | s), 1) > 0.75 for s in seen):
                seen.append(tokens)
                unique.append(item)
        return unique