import argparse
import torch
import cv2
from src.tetris import Tetris
import numpy as np

def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of Deep Q Network to play Tetris""")

    parser.add_argument("--width", type=int, default=10, help="The common width for all images")
    parser.add_argument("--height", type=int, default=20, help="The common height for all images")
    parser.add_argument("--block_size", type=int, default=30, help="Size of a block")
    parser.add_argument("--fps", type=int, default=300, help="frames per second")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument("--output", type=str, default="output.mp4")
    parser.add_argument("--num_test_runs", type=int, default=1)
 

    args = parser.parse_args()
    return args


def test(opt):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)
    if torch.cuda.is_available():
        model = torch.load("{}/tetris".format(opt.saved_path))
    else:
        model = torch.load("{}/tetris".format(opt.saved_path), map_location=lambda storage, loc: storage, weights_only=False)
    model.eval()
    env = Tetris(width=opt.width, height=opt.height, block_size=opt.block_size)
    env.reset()
    if torch.cuda.is_available():
        model.cuda()
    # out = cv2.VideoWriter(opt.output, cv2.VideoWriter_fourcc(*"MJPG"), opt.fps,
    #                       (int(1.5*opt.width*opt.block_size), opt.height*opt.block_size))
    while True:
        next_steps = env.get_next_states()
        next_actions, next_states = zip(*next_steps.items())
        next_states = torch.stack(next_states)
        if torch.cuda.is_available():
            next_states = next_states.cuda()
        predictions = model(next_states)[:, 0]
        index = torch.argmax(predictions).item()
        action = next_actions[index]
        _, done = env.step(action, render=True, video=None)

        if done:
            return [env.score,env.tetrominoes,env.cleared_lines]
            #out.release()
            # break

if __name__ == "__main__":
    opt = get_args()
    num_test_runs = opt.num_test_runs # Số lần chạy kiểm thử

    all_scores = []
    all_tetrominoes = []
    all_cleared_lines = []
    for i in range(num_test_runs):
        result = test(opt)
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
