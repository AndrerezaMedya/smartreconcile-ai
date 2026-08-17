"""
Detailed examination of query-level routing errors and precision gating.
"""

import json
from pathlib import Path

# Load routing analysis
with open("results/experiment_C_routing_analysis.json", encoding="utf-8") as f:
    routing_data = json.load(f)

for dname, rdata in routing_data["routing_per_dataset"].items():
    print(f"\n{'='*70}")
    print(f"Dataset: {dname}")
    print(f"  Matched queries: {rdata['matched_queries']}")
    print(f"  Lexical direct: {rdata['lexical_direct_pct']}%")
    print(f"  Semantic routed: {rdata['semantic_routed_pct']}%")
    print(f"  Beneficial: {rdata['beneficial_interventions']}, Harmful: {rdata['harmful_interventions']}, Neutral: {rdata['neutral_changes']}")
    
    print("\nHARMFUL INTERVENTIONS (Lexical was RIGHT, Semantic made WRONG):")
    for q in rdata["query_details"]:
        if q["action_type"] == "HARMFUL_INTERVENTION":
            print(f"  [{q['query_id']}] inv='{q['invoice_line']}' | lex_margin={q['lex_margin']} lex_top1={q['lex_top1']}")
            print(f"     Target PO: {q['correct_desc']}")
            print(f"     Semantic Picked: {q['chosen_desc']}")
    
    print("\nBENEFICIAL INTERVENTIONS (Lexical was WRONG, Semantic made RIGHT):")
    for q in rdata["query_details"]:
        if q["action_type"] == "BENEFICIAL_INTERVENTION":
            print(f"  [{q['query_id']}] inv='{q['invoice_line']}' | lex_margin={q['lex_margin']} lex_top1={q['lex_top1']}")
            print(f"     Target PO: {q['correct_desc']}")
            print(f"     Semantic Picked: {q['chosen_desc']}")
