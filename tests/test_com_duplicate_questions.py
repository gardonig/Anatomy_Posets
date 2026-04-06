"""
Test for duplicate questions in MatrixBuilder using real CoM structures.

Loads CoM_cleaned_global_avg_xyz.json and runs through full query iteration,
tracking all (i,j) pairs asked to detect:
1. Duplicate questions (same pair asked multiple times)
2. Both (i,j) and (j,i) being asked (they shouldn't be)
3. Any pairs violating the i < j invariant
"""

import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

# Ensure imports work
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.anatomy_poset.core.io import load_structures_from_json
from src.anatomy_poset.core.matrix_builder import MatrixBuilder
from src.anatomy_poset.core.axis_models import AXIS_VERTICAL


def test_com_structures_no_duplicate_questions_exhaustive() -> None:
    """
    Load real CoM structures, run full MatrixBuilder iteration,
    and verify no duplicate questions are asked.
    """
    # Load structures
    structures_path = ROOT / "data" / "structures" / "CoM_cleaned_global_avg_xyz.json"
    assert structures_path.exists(), f"CoM structures file not found at {structures_path}"

    structures = load_structures_from_json(str(structures_path))
    assert len(structures) > 0, "No structures loaded from JSON"

    print(f"\n{'='*70}")
    print(f"DUPLICATE QUESTIONS TEST - CoM Structures")
    print(f"{'='*70}")
    print(f"Loaded {len(structures)} structures from {structures_path.name}")
    print(f"Testing MatrixBuilder for vertical axis")
    print()

    # Initialize MatrixBuilder
    mb = MatrixBuilder(structures, axis=AXIS_VERTICAL)
    print(f"MatrixBuilder initialized: {mb.n} structures (after sorting by CoM)")
    print()

    # Track all pairs asked
    pairs_asked: List[Tuple[int, int]] = []
    pairs_set: Set[Tuple[int, int]] = set()
    question_count = 0

    # Iterate through all questions
    print("Iterating through next_pair()...")
    while True:
        pair = mb.next_pair()
        if pair is None:
            break

        i, j = pair
        question_count += 1
        pairs_asked.append(pair)

        # Check invariant: i < j (strict upper triangle)
        if i >= j:
            print(f"ERROR: Invalid pair ({i}, {j}) - violates i < j invariant!")
            assert False, f"Pair ({i}, {j}) violates i < j invariant"

        # Check for duplicates
        if pair in pairs_set:
            print(f"ERROR: Duplicate pair {pair} asked on question #{question_count}!")
            assert False, f"Duplicate pair {pair} asked"

        pairs_set.add(pair)

        # Check if inverse was already asked
        inverse = (j, i)
        if inverse in pairs_set:
            print(f"ERROR: Both ({i}, {j}) and ({j}, {i}) were asked!")
            assert False, f"Both directions of pair asked: ({i}, {j}) and ({j}, {i})"

        # Answer with YES for faster iteration
        mb.record_response_matrix(i, j, 1)

    print(f"Total questions asked: {question_count}")
    print()

    # Analysis report
    print(f"{'='*70}")
    print("ANALYSIS REPORT")
    print(f"{'='*70}")
    print()

    # Check for duplicate pairs
    duplicates: Dict[Tuple[int, int], int] = {}
    for pair in pairs_asked:
        duplicates[pair] = duplicates.get(pair, 0) + 1

    duplicate_pairs = {p: count for p, count in duplicates.items() if count > 1}

    if duplicate_pairs:
        print("DUPLICATES FOUND:")
        for pair, count in sorted(duplicate_pairs.items()):
            print(f"  Pair {pair} was asked {count} times")
        assert False, f"Found {len(duplicate_pairs)} duplicate pairs"
    else:
        print("✓ No duplicate pairs found")

    # Check for both directions
    both_directions: List[Tuple[int, int]] = []
    for i, j in pairs_asked:
        if (j, i) in pairs_set:
            both_directions.append((i, j))

    if both_directions:
        print(f"\nBoth directions asked:")
        for pair in both_directions:
            print(f"  ({pair[0]}, {pair[1]}) and ({pair[1]}, {pair[0]})")
        assert False, f"Found {len(both_directions)} pairs asked in both directions"
    else:
        print("✓ No pair asked in both directions (i,j) and (j,i)")

    print()

    # Matrix fill report
    print(f"{'='*70}")
    print("MATRIX FILL REPORT")
    print(f"{'='*70}")
    print()

    # Count None cells in upper triangle
    none_count = 0
    total_upper = 0
    for i in range(mb.n):
        for j in range(i + 1, mb.n):
            total_upper += 1
            if mb.M[i][j] is None:
                none_count += 1

    print(f"Upper triangle cells: {total_upper}")
    print(f"Filled cells: {total_upper - none_count}")
    print(f"Unfilled (None) cells: {none_count}")

    if none_count == 0:
        print("✓ Upper triangle fully filled after exhaustion")
    else:
        print(f"⚠ {none_count} cells remain unfilled (may be due to transitive closure or cycles)")

    print()

    # Statistics
    print(f"{'='*70}")
    print("STATISTICS")
    print(f"{'='*70}")
    print(f"Total structures: {len(structures)}")
    print(f"Structure count (after sort): {mb.n}")
    print(f"Total possible pairs (upper triangle): {mb.n * (mb.n - 1) // 2}")
    print(f"Questions actually asked: {question_count}")
    print(f"Questions skipped (by transitive closure, etc): {mb.n * (mb.n - 1) // 2 - question_count}")

    if question_count > 0:
        skipped = mb.n * (mb.n - 1) // 2 - question_count
        skip_percentage = (skipped / (mb.n * (mb.n - 1) // 2)) * 100
        print(f"Skip percentage: {skip_percentage:.1f}%")

    print()
    print(f"{'='*70}")
    print("✓ ALL TESTS PASSED - No duplicates found!")
    print(f"{'='*70}")
    print()


def test_com_structures_no_duplicate_questions_mixed_answers() -> None:
    """
    Test with mixed answers (YES/NO/NOT_SURE) to ensure no duplicates
    regardless of answer patterns.
    """
    # Load structures
    structures_path = ROOT / "data" / "structures" / "CoM_cleaned_global_avg_xyz.json"
    structures = load_structures_from_json(str(structures_path))

    print(f"\n{'='*70}")
    print(f"DUPLICATE QUESTIONS TEST - Mixed Answers")
    print(f"{'='*70}")
    print(f"Testing with cycling: YES -> NO -> NOT_SURE")
    print()

    # Initialize MatrixBuilder
    mb = MatrixBuilder(structures, axis=AXIS_VERTICAL)

    # Track all pairs asked
    pairs_set: Set[Tuple[int, int]] = set()
    question_count = 0
    answers = [1, -1, 0]  # YES, NO, NOT_SURE

    # Iterate with mixed answers
    while True:
        pair = mb.next_pair()
        if pair is None:
            break

        i, j = pair
        question_count += 1

        # Check for duplicates
        assert pair not in pairs_set, f"Duplicate pair {pair} asked"
        pairs_set.add(pair)

        # Cycle through answers
        answer = answers[question_count % 3]
        mb.record_response_matrix(i, j, answer)

    print(f"Total questions asked: {question_count}")
    print("✓ No duplicates with mixed answers")
    print()


def run_detailed_analysis() -> None:
    """
    Run a detailed analysis showing which structures are being compared,
    useful for understanding the iteration pattern.
    """
    structures_path = ROOT / "data" / "structures" / "CoM_cleaned_global_avg_xyz.json"
    structures = load_structures_from_json(str(structures_path))

    print(f"\n{'='*70}")
    print(f"DETAILED ANALYSIS - First 20 Questions")
    print(f"{'='*70}")
    print()

    mb = MatrixBuilder(structures, axis=AXIS_VERTICAL)

    # Build name map
    name_map = {i: mb.structures[i].name for i in range(mb.n)}

    print(f"{'Q#':<4} {'i':<3} {'j':<3} Structure_i (CoM)         Structure_j (CoM)")
    print(f"{'-'*70}")

    question_count = 0
    while question_count < 20:
        pair = mb.next_pair()
        if pair is None:
            break

        i, j = pair
        question_count += 1

        s_i = mb.structures[i]
        s_j = mb.structures[j]

        print(f"{question_count:<4} {i:<3} {j:<3} {s_i.name:<25} {s_j.name:<25}")
        print(f"{'':4}     {s_i.com_vertical:>8.2f}              {s_j.com_vertical:>8.2f}")

        mb.record_response_matrix(i, j, 1)

    print()


if __name__ == "__main__":
    # Run the detailed analysis first
    run_detailed_analysis()

    # Run the main tests
    print("\nRunning comprehensive tests...")
    test_com_structures_no_duplicate_questions_exhaustive()
    test_com_structures_no_duplicate_questions_mixed_answers()

    print("\n✓ ALL TESTS COMPLETED SUCCESSFULLY")
