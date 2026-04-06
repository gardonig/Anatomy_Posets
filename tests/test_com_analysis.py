"""
Extended analysis of MatrixBuilder query patterns with real CoM structures.
Provides detailed statistics on bilateral handling, transitive closure, and query gaps.
"""

import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

# Setup imports
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.anatomy_poset.core.io import load_structures_from_json
from src.anatomy_poset.core.matrix_builder import MatrixBuilder
from src.anatomy_poset.core.axis_models import AXIS_VERTICAL


def analyze_bilateral_structures():
    """Identify and analyze bilateral structures in the CoM data."""
    structures_path = ROOT / "data" / "structures" / "CoM_cleaned_global_avg_xyz.json"
    structures = load_structures_from_json(str(structures_path))

    print(f"\n{'='*70}")
    print(f"BILATERAL STRUCTURES ANALYSIS")
    print(f"{'='*70}")
    print()

    # Find bilateral pairs (Left/Right)
    bilateral_cores: Dict[str, List[Tuple[int, str]]] = {}

    for idx, s in enumerate(structures):
        name = s.name.lower()

        # Extract core and side
        if "left" in name:
            core = name.replace("_left", "").replace("left", "").strip()
            if core:
                bilateral_cores.setdefault(core, []).append((idx, "Left"))
        elif "right" in name:
            core = name.replace("_right", "").replace("right", "").strip()
            if core:
                bilateral_cores.setdefault(core, []).append((idx, "Right"))

    print(f"Total bilateral structure pairs found: {len(bilateral_cores)}")
    print()

    bilateral_pairs = []
    for core, sides in bilateral_cores.items():
        if len(sides) == 2:
            print(f"Core '{core}':")
            for idx, side in sorted(sides):
                print(f"  [{idx}] {structures[idx].name}")
            bilateral_pairs.append(tuple(sorted([s[0] for s in sides])))

    print()
    print(f"Total complete bilateral pairs: {len(bilateral_pairs)}")
    print()

    # Estimate questions saved by bilateral mirroring
    # Each bilateral pair answer saves 3 questions:
    # - Diagonal (left vs right) never asked
    # - Answer to (left, X) automatically fills (right, X)
    # This depends on how many X structures they relate to

    return len(bilateral_pairs)


def analyze_query_statistics():
    """Detailed statistics on query iteration."""
    structures_path = ROOT / "data" / "structures" / "CoM_cleaned_global_avg_xyz.json"
    structures = load_structures_from_json(str(structures_path))

    print(f"\n{'='*70}")
    print(f"QUERY ITERATION STATISTICS")
    print(f"{'='*70}")
    print()

    mb = MatrixBuilder(structures, axis=AXIS_VERTICAL)

    # Track by gap size
    gap_statistics: Dict[int, Dict[str, int]] = {}
    question_count = 0

    pairs_asked: List[Tuple[int, int]] = []

    while True:
        pair = mb.next_pair()
        if pair is None:
            break

        i, j = pair
        question_count += 1
        pairs_asked.append(pair)

        gap = j - i
        if gap not in gap_statistics:
            gap_statistics[gap] = {"asked": 0, "skipped": 0}
        gap_statistics[gap]["asked"] += 1

        mb.record_response_matrix(i, j, 1)

    # Count skipped pairs by gap
    for gap in range(1, mb.n):
        for i in range(mb.n - gap):
            j = i + gap
            # Check if this pair was asked
            if gap not in gap_statistics:
                gap_statistics[gap] = {"asked": 0, "skipped": 0}
            if (i, j) not in pairs_asked:
                gap_statistics[gap]["skipped"] = gap_statistics[gap].get("skipped", 0) + 1

    print("Questions by gap size (j - i):")
    print(f"{'Gap':<6} {'Asked':<8} {'Skipped':<8} {'Total':<8} {'Ask %':<8}")
    print(f"{'-'*50}")

    total_asked = 0
    total_possible = 0

    for gap in sorted(gap_statistics.keys()):
        stats = gap_statistics[gap]
        asked = stats["asked"]
        skipped = stats["skipped"]
        total = asked + skipped
        ask_percent = (asked / total * 100) if total > 0 else 0

        print(f"{gap:<6} {asked:<8} {skipped:<8} {total:<8} {ask_percent:>6.1f}%")
        total_asked += asked
        total_possible += total

    print(f"{'-'*50}")
    print(f"{'TOTAL':<6} {total_asked:<8} {total_possible - total_asked:<8} {total_possible:<8} {total_asked/total_possible*100:>6.1f}%")
    print()

    # Analyze which pairs were skipped
    print(f"\nTop 10 Gap Sizes by Skip Rate:")
    print(f"{'Gap':<6} {'Ask %':<8} {'Questions':<12}")
    print(f"{'-'*40}")

    gap_efficiency = []
    for gap in sorted(gap_statistics.keys()):
        stats = gap_statistics[gap]
        asked = stats["asked"]
        total = asked + stats["skipped"]
        ask_percent = (asked / total * 100) if total > 0 else 0
        gap_efficiency.append((gap, ask_percent, asked))

    gap_efficiency.sort(key=lambda x: x[1], reverse=True)

    for gap, ask_percent, asked in gap_efficiency[:10]:
        print(f"{gap:<6} {ask_percent:>6.1f}%     {asked}")

    print()


def analyze_transitive_closure_impact():
    """Analyze impact of transitive closure on question reduction."""
    structures_path = ROOT / "data" / "structures" / "CoM_cleaned_global_avg_xyz.json"
    structures = load_structures_from_json(str(structures_path))

    print(f"\n{'='*70}")
    print(f"TRANSITIVE CLOSURE IMPACT ANALYSIS")
    print(f"{'='*70}")
    print()

    mb = MatrixBuilder(structures, axis=AXIS_VERTICAL)
    n = mb.n

    # Count cells that will be filled by transitive closure
    questions_asked = 0
    cells_filled_by_question = 0
    cells_filled_by_inference = 0

    while True:
        pair = mb.next_pair()
        if pair is None:
            break

        i, j = pair
        questions_asked += 1
        cells_filled_by_question += 1  # The direct answer

        mb.record_response_matrix(i, j, 1)

    # After all questions, check what's filled
    total_upper = n * (n - 1) // 2
    cells_filled_by_inference = sum(
        1 for i in range(n) for j in range(i + 1, n)
        if mb.M[i][j] is not None
    ) - cells_filled_by_question

    print(f"Total structures: {n}")
    print(f"Total upper triangle cells: {total_upper}")
    print()
    print(f"Questions directly asked: {questions_asked}")
    print(f"Cells filled by direct questions: {cells_filled_by_question}")
    print(f"Cells filled by transitive inference: {cells_filled_by_inference}")
    print()
    print(f"Question efficiency: {cells_filled_by_question / questions_asked:.1f} cells per question (direct)")
    print(f"Total filled: {cells_filled_by_question + cells_filled_by_inference} / {total_upper}")
    print()
    print(f"Reduction factor: {total_upper / questions_asked:.1f}x fewer questions than maximum")
    print()


def analyze_com_values_distribution():
    """Analyze distribution of CoM values and their impact on query patterns."""
    structures_path = ROOT / "data" / "structures" / "CoM_cleaned_global_avg_xyz.json"
    structures = load_structures_from_json(str(structures_path))

    print(f"\n{'='*70}")
    print(f"CENTER OF MASS (CoM) DISTRIBUTION ANALYSIS")
    print(f"{'='*70}")
    print()

    mb = MatrixBuilder(structures, axis=AXIS_VERTICAL)

    # Get sorted structures by CoM
    vertical_coms = [mb.structures[i].com_vertical for i in range(mb.n)]

    print(f"Vertical CoM (Z-axis, superior-inferior):")
    print(f"  Minimum: {min(vertical_coms):.2f} (lowest/most inferior)")
    print(f"  Maximum: {max(vertical_coms):.2f} (highest/most superior)")
    print(f"  Range: {max(vertical_coms) - min(vertical_coms):.2f}")
    print()

    # Analyze gap distribution
    print(f"Top 10 Structures by Vertical CoM (after sort):")
    print(f"{'Idx':<5} {'Structure':<30} {'Com_Vertical':<15}")
    print(f"{'-'*50}")

    for i in range(min(10, mb.n)):
        s = mb.structures[i]
        print(f"{i:<5} {s.name:<30} {s.com_vertical:>14.2f}")

    print()
    print(f"Bottom 10 Structures by Vertical CoM:")
    print(f"{'Idx':<5} {'Structure':<30} {'Com_Vertical':<15}")
    print(f"{'-'*50}")

    for i in range(max(0, mb.n - 10), mb.n):
        s = mb.structures[i]
        print(f"{i:<5} {s.name:<30} {s.com_vertical:>14.2f}")

    print()

    # Count ties (equal CoM values)
    from collections import Counter
    com_counts = Counter(vertical_coms)
    ties = {com: count for com, count in com_counts.items() if count > 1}

    print(f"CoM value ties (multiple structures with same vertical CoM):")
    if ties:
        print(f"Number of tied groups: {len(ties)}")
        for com, count in sorted(ties.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {com:.2f}: {count} structures")
    else:
        print(f"No ties found - all structures have unique vertical CoM values")

    print()


if __name__ == "__main__":
    analyze_bilateral_structures()
    analyze_query_statistics()
    analyze_transitive_closure_impact()
    analyze_com_values_distribution()

    print(f"\n{'='*70}")
    print(f"ANALYSIS COMPLETE")
    print(f"{'='*70}")
    print()
