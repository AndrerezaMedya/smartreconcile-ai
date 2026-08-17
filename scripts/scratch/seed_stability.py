import statistics

rows = [
    ("seed42", 0.8519, 0.8378, 0.7143),
    ("seed43", 0.8519, 0.8378, 0.7143),
    ("seed44", 0.8519, 0.8378, 0.7857),
]
print("Multi-seed stability (FT-v2, train_v2, 148 examples):")
print(f"  {'seed':<10} {'syn_v2':>8} {'hard_neg':>9} {'adv_v2':>8}")
print("  " + "-"*40)
for label, s, h, a in rows:
    print(f"  {label:<10} {s:>8.4f} {h:>9.4f} {a:>8.4f}")
vals_s = [r[1] for r in rows]
vals_h = [r[2] for r in rows]
vals_a = [r[3] for r in rows]
print(f"  {'mean':<10} {statistics.mean(vals_s):>8.4f} {statistics.mean(vals_h):>9.4f} {statistics.mean(vals_a):>8.4f}")
print(f"  {'std':<10} {statistics.stdev(vals_s):>8.4f} {statistics.stdev(vals_h):>9.4f} {statistics.stdev(vals_a):>8.4f}")
print()
print("Key observations:")
print("  syn_v2: 0.000 std — perfectly stable across seeds")
print("  hard_neg: 0.000 std — perfectly stable")
print("  adv_v2: seed44=0.7857 (best), seeds 42/43=0.7143")
print()
print("Best for production: seed44 (adv_v2 +7pp) if adversarial performance is key")
print("Best by val MRR: seed43 (0.446) — marginal difference")
print()
print("Recall: All semantic models FAIL NFR-09 (syn_v2=0.852 < 0.90)")
print("NFR-09 is MET by: Lexical (1.000) and Hybrid (1.000, alpha=0.0 = Lexical)")
