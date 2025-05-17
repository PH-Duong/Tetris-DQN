import numpy as np
from a_star import evaluate

num_test_runs = 1 # Số lần chạy kiểm thử

all_scores = []
all_tetrominoes = []
all_cleared_lines = []
for i in range(num_test_runs):
    result = evaluate()
    score, tetrominoes, cleared_lines  = result[0],result[1],result[2]
    all_scores.append(score)
    all_tetrominoes.append(tetrominoes)
    all_cleared_lines.append(cleared_lines)
    print("TEST %d | Score: %d, Tetrominoes: %d, Cleared lines: %d" %(i,score,tetrominoes,cleared_lines))
print(all_scores,all_tetrominoes,all_cleared_lines)
# Tính toán và in thống kê
print("\n--- Test Results Summary ---")
print(f"Number of runs: {num_test_runs}")

print("\nScores:")
print(f"  Mean: {np.mean(all_scores):.2f}")
print(f"  Std Dev: {np.std(all_scores, ddof=1):.2f}")
print(f"  Max: {np.max(all_scores)}")
print(f"  Min: {np.min(all_scores)}")

print("\nTetrominoes:")
print(f"  Mean: {np.mean(all_tetrominoes):.2f}")
print(f"  Std Dev: {np.std(all_tetrominoes, ddof=1):.2f}")
print(f"  Max: {np.max(all_tetrominoes)}")
print(f"  Min: {np.min(all_tetrominoes)}")

print("\nCleared Lines:")
print(f"  Mean: {np.mean(all_cleared_lines):.2f}")
print(f"  Std Dev: {np.std(all_cleared_lines, ddof=1):.2f}")
print(f"  Max: {np.max(all_cleared_lines)}")
print(f"  Min: {np.min(all_cleared_lines)}")
