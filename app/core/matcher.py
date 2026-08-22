"""
Hybrid Lexical + Fine-Tuned Semantic Matcher and Greedy 1:1 Assignment Engine.
Strictly implements the frozen Phase 8 / Experiment C matching architecture.
"""

import re
import numpy as np
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer
from app.core.config import (
    DEFAULT_MODEL_PATH,
    TAU_MARGIN,
    TAU_TOP1_SCORE,
    SEMANTIC_BLEND_ALPHA,
    MIN_SIMILARITY_FLOOR
)
from app.core.models import InvoiceLine, POLine, CandidateMatch


class HybridMatcher:
    """
    Singleton wrapper for the validated hybrid matching and greedy assignment engine.
    """
    _instance: Optional["HybridMatcher"] = None
    _model: Optional[SentenceTransformer] = None

    def __init__(self, model_path: Optional[str] = None):
        target_path = model_path or str(DEFAULT_MODEL_PATH)
        if HybridMatcher._model is None:
            print(f"Loading Fine-Tuned Semantic Model: {target_path} on CPU...")
            HybridMatcher._model = SentenceTransformer(target_path, device="cpu")
        self.model = HybridMatcher._model

    @staticmethod
    def normalize(text: str) -> str:
        """Normalized lowercase alphanumeric string."""
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @classmethod
    def jaccard(cls, a: str, b: str) -> float:
        """Token-level Jaccard similarity."""
        ta = set(cls.normalize(a).split())
        tb = set(cls.normalize(b).split())
        if not ta or not tb:
            return 0.0
        return len(ta & tb) / len(ta | tb)

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """Batch encodes texts to normalized CPU embeddings."""
        return self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )

    def compute_similarity_matrix(
        self,
        invoice_lines: List[InvoiceLine],
        po_lines: List[POLine]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[bool]]:
        """
        Computes pairwise similarity matrix across all (invoice_line, po_line) pairs.
        Returns: (hybrid_matrix, lexical_matrix, semantic_matrix, is_ambiguous_flags)
        """
        n_inv = len(invoice_lines)
        n_po = len(po_lines)

        if n_inv == 0 or n_po == 0:
            # is_ambiguous_flags must stay length n_inv: match_and_assign indexes it
            # per invoice line regardless of how many PO lines are available to score.
            return np.zeros((n_inv, n_po)), np.zeros((n_inv, n_po)), np.zeros((n_inv, n_po)), [False] * n_inv

        # Stage 1: Lexical scoring & gating check
        lex_matrix = np.zeros((n_inv, n_po))
        is_ambiguous_flags = []

        for i, il in enumerate(invoice_lines):
            l_scores = [self.jaccard(il.description, pl.description) for pl in po_lines]
            sorted_lex = sorted([(l_scores[j], j) for j in range(n_po)], key=lambda x: -x[0])
            l_top1 = sorted_lex[0][0]
            l_top2 = sorted_lex[1][0] if len(sorted_lex) > 1 else 0.0
            l_margin = l_top1 - l_top2

            is_ambiguous = (l_margin <= TAU_MARGIN) or (l_top1 <= TAU_TOP1_SCORE)
            is_ambiguous_flags.append(is_ambiguous)

            for j in range(n_po):
                lex_matrix[i, j] = l_scores[j]

        # Stage 2: Fine-Tuned Semantic Model CPU encoding
        inv_texts = [il.description for il in invoice_lines]
        po_texts = [pl.description for pl in po_lines]

        inv_embs = self.encode_texts(inv_texts)
        po_embs = self.encode_texts(po_texts)

        sem_matrix = inv_embs @ po_embs.T

        # Stage 2b: Blended scoring when triggered
        hybrid_matrix = np.zeros((n_inv, n_po))
        for i in range(n_inv):
            is_amb = is_ambiguous_flags[i]
            for j in range(n_po):
                if is_amb:
                    hybrid_matrix[i, j] = (
                        SEMANTIC_BLEND_ALPHA * sem_matrix[i, j]
                        + (1.0 - SEMANTIC_BLEND_ALPHA) * lex_matrix[i, j]
                    )
                else:
                    hybrid_matrix[i, j] = lex_matrix[i, j]

        return hybrid_matrix, lex_matrix, sem_matrix, is_ambiguous_flags

    def match_and_assign(
        self,
        invoice_lines: List[InvoiceLine],
        po_lines: List[POLine]
    ) -> List[Dict]:
        """
        Executes hybrid scoring, greedy 1:1 assignment, and candidate ranking.
        """
        n_inv = len(invoice_lines)
        n_po = len(po_lines)

        hybrid_matrix, lex_matrix, sem_matrix, is_ambiguous_flags = (
            self.compute_similarity_matrix(invoice_lines, po_lines)
        )

        # Stage 3: Greedy 1:1 Mutual-Exclusive Assignment
        pairs = []
        for i in range(n_inv):
            for j in range(n_po):
                pairs.append((hybrid_matrix[i, j], i, j))

        # Sort descending by score, tiebreak by po_line_no for deterministic stability
        pairs.sort(key=lambda x: (-x[0], po_lines[x[2]].po_line_no))

        assigned_inv = set()
        used_po = set()
        inv_to_po = {}
        inv_to_score = {}
        inv_to_margin = {}

        for score, i, j in pairs:
            if i not in assigned_inv and j not in used_po:
                if score >= MIN_SIMILARITY_FLOOR:
                    inv_to_po[i] = j
                    inv_to_score[i] = score
                    assigned_inv.add(i)
                    used_po.add(j)

        # Compute confidence margin
        for i in range(n_inv):
            if i in inv_to_po:
                assigned_j = inv_to_po[i]
                other_scores = [hybrid_matrix[i, j] for j in range(n_po) if j != assigned_j]
                runner_up = max(other_scores) if other_scores else 0.0
                inv_to_margin[i] = inv_to_score[i] - runner_up
            else:
                inv_to_margin[i] = 0.0

        # Construct structured candidate rankings per invoice line
        results = []
        for i, il in enumerate(invoice_lines):
            ranked_po_indices = np.argsort(-hybrid_matrix[i, :])
            top_candidates = []
            for po_idx in ranked_po_indices:
                pl = po_lines[po_idx]
                top_candidates.append(CandidateMatch(
                    po_line_no=pl.po_line_no,
                    description=pl.description,
                    ordered_qty=pl.ordered_qty,
                    uom=pl.uom,
                    unit_price=pl.unit_price,
                    lexical_score=round(float(lex_matrix[i, po_idx]), 4),
                    semantic_score=round(float(sem_matrix[i, po_idx]), 4),
                    hybrid_score=round(float(hybrid_matrix[i, po_idx]), 4),
                    is_semantic_routed=is_ambiguous_flags[i]
                ))

            assigned_j = inv_to_po.get(i)
            assigned_po = po_lines[assigned_j] if assigned_j is not None else None

            results.append({
                "line_no": il.line_no,
                "invoice_line": il,
                "assigned_po": assigned_po,
                "score": round(float(inv_to_score.get(i, 0.0)), 4),
                "confidence_margin": round(float(inv_to_margin.get(i, 0.0)), 4),
                "is_semantic_routed": is_ambiguous_flags[i],
                "top_candidates": top_candidates
            })

        return results
