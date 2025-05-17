import cv2 # Để hiển thị và chờ phím
from src.tetris import Tetris # Import môi trường game Tetris
# Không cần torch hay deep_q_network.py nữa cho A*

class Args:
    """Lớp giả lập các tham số dòng lệnh cho dễ sử dụng."""
    def __init__(self):
        self.width = 10
        self.height = 20
        self.block_size = 30
        self.max_steps_per_episode = 5000 # Giới hạn số khối trong một game
        self.render = False
        self.printStep = True

def calculate_heuristic_cost(board_properties_tensor):
    """
    Tính toán chi phí heuristic cho một trạng thái bảng game.
    Trạng thái tệ hơn sẽ có chi phí cao hơn.

    Args:
        board_properties_tensor: Một tensor (hoặc list/tuple) chứa các đặc trưng:
                                 [lines_cleared_in_this_step, holes, bumpiness, aggregate_height]

    Returns:
        float: Giá trị heuristic cost.
    """
    if hasattr(board_properties_tensor, 'tolist'): # Nếu là tensor PyTorch
        properties = board_properties_tensor.tolist()
    else:
        properties = list(board_properties_tensor)

    lines_cleared_in_this_step = properties[0]
    num_holes = properties[1]
    total_bumpiness = properties[2]
    aggregate_height = properties[3]

    # --- TRỌNG SỐ CHO HEURISTIC (CẦN TINH CHỈNH KỸ LƯỠNG!) ---
    # Đây là phần quan trọng nhất để A* hoạt động hiệu quả.
    # Giá trị dương có nghĩa là "tệ", giá trị âm có nghĩa là "tốt".
    # Chúng ta muốn TỐI THIỂU HÓA heuristic_cost.

    weight_aggregate_height = 0.510066  # Càng cao càng tệ
    weight_lines_cleared = -0.760666    # Xóa hàng là rất tốt (giảm chi phí)
    weight_num_holes = 0.35663          # Càng nhiều lỗ càng tệ
    weight_total_bumpiness = 0.184483   # Càng gồ ghề càng tệ

    # Tính toán heuristic
    cost = (weight_aggregate_height * aggregate_height +
            weight_lines_cleared * lines_cleared_in_this_step + # lines_cleared_in_this_step đã là số dương
            weight_num_holes * num_holes +
            weight_total_bumpiness * total_bumpiness)

    return cost

def choose_action_with_a_star(tetris_environment):
    """
    Chọn hành động (vị trí đặt khối) tốt nhất bằng cách sử dụng A* (Greedy Best-First).

    Args:
        tetris_environment: Instance của lớp Tetris.

    Returns:
        tuple: Hành động (x, num_rotations) tốt nhất, hoặc None nếu không có hành động nào.
    """
    possible_next_moves = tetris_environment.get_next_states()
    # possible_next_moves là một dict: {(x, rotation_index): board_properties_tensor}

    if not possible_next_moves:
        return None

    best_action = None
    min_f_cost = float('inf')
    g_cost = 0 # Hoặc 1, vì chỉ xét 1 bước, g_cost không ảnh hưởng nhiều đến việc chọn lựa tương đối

    for action, board_properties in possible_next_moves.items():
        h_cost = calculate_heuristic_cost(board_properties)
        f_cost = g_cost + h_cost

        if f_cost < min_f_cost:
            min_f_cost = f_cost
            best_action = action
        # Xử lý trường hợp f_cost bằng nhau (ví dụ: chọn ngẫu nhiên hoặc giữ cái đầu tiên)
        # Ở đây, chúng ta mặc định giữ hành động đầu tiên tìm thấy có f_cost tốt nhất.

    return best_action

def run_tetris_with_a_star(options):
    """
    Chạy game Tetris với AI sử dụng thuật toán A*.
    """
    env = Tetris(width=options.width, height=options.height, block_size=options.block_size)
    env.reset()

    game_over = False
    total_score = 0

    print("Bắt đầu game Tetris với A* Agent.")
    print("Nhấn phím bất kỳ trên cửa sổ game để đóng khi game over.")

    while not game_over and env.tetrominoes < options.max_steps_per_episode:
        if options.render:
            env.render(video=None) # Hiển thị trạng thái hiện tại của game

        # AI chọn hành động tốt nhất
        chosen_action = choose_action_with_a_star(env)

        if chosen_action is None:
            print("A* không tìm thấy hành động nào. Game có thể bị kẹt.")
            game_over = True # Không có nước đi hợp lệ
            break

        # Thực hiện hành động đã chọn
        reward, game_over = env.step(action=chosen_action, render=True) # render=False vì đã render ở trên
        total_score += reward

        if options.printStep:
            print(f"Khối: {env.tetrominoes}, Hành động: {chosen_action}, "
                f"Điểm bước: {reward:.2f}, Tổng điểm: {total_score}, "
                f"Hàng đã xóa: {env.cleared_lines}")

        if game_over:
            print("\nGAME OVER!")
            print(f"Điểm cuối cùng: {total_score}")
            print(f"Tổng số khối đã đặt: {env.tetrominoes}")
            print(f"Tổng số hàng đã xóa: {env.cleared_lines}")
            break
        
        if env.tetrominoes >= options.max_steps_per_episode:
            print(f"\nĐạt giới hạn {options.max_steps_per_episode} khối.")
            print(f"Điểm cuối cùng: {total_score}")
            print(f"Tổng số khối đã đặt: {env.tetrominoes}")
            print(f"Tổng số hàng đã xóa: {env.cleared_lines}")
            break
    if options.render:
        env.render(video=None) # Render trạng thái cuối cùng
    cv2.waitKey(0) # Chờ người dùng nhấn phím để đóng cửa sổ
    cv2.destroyAllWindows()

    return (total_score,env.tetrominoes,env.cleared_lines)

if __name__ == "__main__":
    game_options = Args()
    run_tetris_with_a_star(game_options)

def evaluate():
    game_options = Args()
    game_options.printStep = False
    return run_tetris_with_a_star(game_options)