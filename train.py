import argparse # Xử lý tham số dòng lệnh
import os # Tương tác hệ điều hành (tạo thư mục)
import shutil # Thao tác file/thư mục (xóa thư mục log cũ)
from random import random, randint, sample # Hàm ngẫu nhiên cho epsilon-greedy và lấy mẫu bộ nhớ

import numpy as np # Thao tác mảng
import torch # Thư viện PyTorch cho mạng nơ-ron và tính toán tensor
import torch.nn as nn # Các lớp xây dựng mạng nơ-ron
from tensorboardX import SummaryWriter # Ghi log cho TensorBoard để theo dõi huấn luyện

from src.deep_q_network import DeepQNetwork # Import lớp mạng DQN đã định nghĩa
from src.tetris import Tetris # Import lớp môi trường game Tetris
from collections import deque # Cấu trúc dữ liệu hàng đợi hiệu quả cho replay memory

# --- Hàm lấy tham số dòng lệnh ---
def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of Deep Q Network to play Tetris""")
    # Các tham số cho môi trường Tetris
    parser.add_argument("--width", type=int, default=10, help="The common width for all images")
    parser.add_argument("--height", type=int, default=20, help="The common height for all images")
    parser.add_argument("--block_size", type=int, default=30, help="Size of a block")
    # Các tham số cho huấn luyện DQN
    parser.add_argument("--batch_size", type=int, default=512, help="The number of images per batch") # Kích thước lô huấn luyện từ replay memory
    parser.add_argument("--lr", type=float, default=1e-3) # Tốc độ học (alpha trong Q-learning)
    parser.add_argument("--gamma", type=float, default=0.99) # Hệ số chiết khấu (gamma trong Q-learning)
    parser.add_argument("--initial_epsilon", type=float, default=1) # Giá trị epsilon ban đầu (bắt đầu = 1, 100% khám phá)
    parser.add_argument("--final_epsilon", type=float, default=1e-3) # Giá trị epsilon cuối cùng (tiến dần về gần 0, chủ yếu khai thác)
    parser.add_argument("--num_decay_epochs", type=float, default=2000) # Số epoch để epsilon giảm từ initial xuống final
    parser.add_argument("--num_epochs", type=int, default=3000) # Tổng số epoch huấn luyện (mỗi epoch tương ứng 1 game)
    parser.add_argument("--save_interval", type=int, default=1000) # Lưu model sau mỗi bao nhiêu epoch
    parser.add_argument("--replay_memory_size", type=int, default=30000, # Kích thước tối đa của replay memory
                        help="Number of epoches between testing phases") # Chú thích này có vẻ không chính xác, đây là kích thước bộ nhớ
    parser.add_argument("--log_path", type=str, default="tensorboard") # Thư mục lưu log TensorBoard
    parser.add_argument("--saved_path", type=str, default="trained_models") # Thư mục lưu model đã huấn luyện
    parser.add_argument("--max_steps_per_episode", type=int, default=3000) # số lượng bước tối đa trong mỗi ván

    args = parser.parse_args()
    return args

# --- Hàm huấn luyện chính ---
def train(opt):
    # --- Khởi tạo ---
    if torch.cuda.is_available(): # Kiểm tra và sử dụng GPU nếu có
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123) # Đặt seed để kết quả có thể tái lập
    if os.path.isdir(opt.log_path): # Xóa thư mục log cũ nếu tồn tại
        shutil.rmtree(opt.log_path)
    os.makedirs(opt.log_path) # Tạo thư mục log mới
    writer = SummaryWriter(opt.log_path) # Khởi tạo đối tượng ghi log TensorBoard
    env = Tetris(width=opt.width, height=opt.height, block_size=opt.block_size) # Khởi tạo môi trường Tetris (Environment trong RL)

    # Khởi tạo mạng DQN và các thành phần liên quan
    model = DeepQNetwork() # Khởi tạo mạng nơ-ron DQN (Function Approximator)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr) # Chọn thuật toán tối ưu hóa (Adam) và tốc độ học
    criterion = nn.MSELoss() # Chọn hàm mất mát (Mean Squared Error) để so sánh Q dự đoán và Q mục tiêu

    state = env.reset() # Reset môi trường và lấy trạng thái ban đầu (Initial State s)
    if torch.cuda.is_available():
        model.cuda() # Chuyển model lên GPU
        state = state.cuda() # Chuyển trạng thái ban đầu lên GPU

    # Khởi tạo bộ nhớ phát lại (Experience Replay)
    replay_memory = deque(maxlen=opt.replay_memory_size) # deque với kích thước tối đa
    epoch = 0

    # --- Vòng lặp huấn luyện chính ---
    while epoch < opt.num_epochs: # Lặp qua số epoch (game) đã định
        # 1. Lấy các hành động/trạng thái tiếp theo có thể
        next_steps = env.get_next_states() # Môi trường tính toán tất cả các vị trí hạ cánh khả thi cho khối hiện tại
                                          # Trả về dạng dict: {(vị trí x, số lần xoay): tensor_trạng_thái_kết_quả}
        # 2. Chọn hành động bằng Epsilon-Greedy
        epsilon = opt.final_epsilon + (max(opt.num_decay_epochs - epoch, 0) * (
                opt.initial_epsilon - opt.final_epsilon) / opt.num_decay_epochs) # Tính epsilon hiện tại (giảm tuyến tính)
        u = random() # Sinh số ngẫu nhiên từ 0 đến 1
        random_action = u <= epsilon # Quyết định khám phá hay khai thác

        next_actions, next_states = zip(*next_steps.items()) # Tách dict thành list các hành động và list các trạng thái tương ứng
        next_states = torch.stack(next_states) # Chuyển list các tensor trạng thái thành 1 tensor duy nhất
        if torch.cuda.is_available():
            next_states = next_states.cuda() # Chuyển lên GPU

        model.eval() # Chuyển model sang chế độ đánh giá (không tính gradient) để dự đoán Q-value
        with torch.no_grad(): # Không cần tính gradient khi dự đoán để chọn hành động
            predictions = model(next_states)[:, 0] # Đưa tất cả các trạng thái tiếp theo có thể vào model để lấy Q-value dự đoán cho từng trạng thái đó
                                                  # Output của model là [batch_size, 1], nên lấy [:, 0]
        model.train() # Chuyển model về chế độ huấn luyện

        if random_action: # Nếu khám phá (Exploration)
            index = randint(0, len(next_steps) - 1) # Chọn một hành động ngẫu nhiên từ danh sách các hành động khả thi
        else: # Nếu khai thác (Exploitation)
            index = torch.argmax(predictions).item() # Chọn hành động có Q-value dự đoán cao nhất

        # Lấy hành động và trạng thái tương ứng đã chọn
        action = next_actions[index]
        # Trạng thái next_state này không được dùng trực tiếp ngay mà sẽ lấy từ env.step()

        # 3. Thực hiện hành động trong môi trường
        reward, done = env.step(action, render=False) # Agent thực hiện hành động đã chọn, môi trường trả về phần thưởng (reward) và cờ kết thúc game (done)
                                                     # Trạng thái thực tế tiếp theo (s') sẽ được lưu trữ trong env.board sau bước này.

        # Lấy trạng thái thực tế sau khi thực hiện hành động (s') để lưu vào bộ nhớ
        # Tuy nhiên, code này lại lưu 'next_state' được dự đoán từ get_next_states vào bộ nhớ.
        # Điều này hơi khác so với DQN chuẩn, nơi s' là trạng thái thực tế sau hành động.
        # Nhưng vì get_next_states trả về các đặc trưng của bảng *sau khi* khối đã hạ cánh,
        # nên next_states[index, :] chính là biểu diễn trạng thái s' cần lưu.
        next_state = next_states[index, :]

        # 4. Lưu kinh nghiệm vào bộ nhớ phát lại (Store Experience)
        # Lưu bộ (trạng thái hiện tại s, phần thưởng r, trạng thái kế tiếp s', cờ kết thúc done)
        replay_memory.append([state, reward, next_state, done])

        # 5. Xử lý kết thúc game / chuyển trạng thái
        if done or env.tetrominoes >= opt.max_steps_per_episode: # Nếu game kết thúc hoặc số lượng bước vượt mức
            final_score = env.score
            final_tetrominoes = env.tetrominoes
            final_cleared_lines = env.cleared_lines
            state = env.reset() # Reset môi trường, bắt đầu game mới, lấy trạng thái ban đầu mới
            if torch.cuda.is_available():
                state = state.cuda()
        else: # Nếu game chưa kết thúc
            state = next_state # Trạng thái kế tiếp trở thành trạng thái hiện tại cho bước tiếp theo
            continue # Bỏ qua phần huấn luyện nếu game chưa xong (huấn luyện chỉ xảy ra khi game kết thúc trong code này)

        # --- Bắt đầu phần huấn luyện mạng DQN ---
        # Chỉ huấn luyện khi replay memory đủ lớn (tránh huấn luyện với quá ít dữ liệu ban đầu)
        if len(replay_memory) < opt.replay_memory_size / 10:
            continue

        epoch += 1 # Tăng số epoch (chỉ tăng khi một game kết thúc và bắt đầu huấn luyện)

        # 6. Lấy mẫu một lô dữ liệu từ bộ nhớ (Sample Mini-batch)
        batch = sample(replay_memory, min(len(replay_memory), opt.batch_size)) # Lấy ngẫu nhiên một batch các kinh nghiệm
        state_batch, reward_batch, next_state_batch, done_batch = zip(*batch) # Tách batch thành các thành phần riêng

        # Chuyển dữ liệu batch thành tensor PyTorch
        state_batch = torch.stack(tuple(state for state in state_batch))
        reward_batch = torch.from_numpy(np.array(reward_batch, dtype=np.float32)[:, None]) # Chuyển reward thành tensor cột
        next_state_batch = torch.stack(tuple(state for state in next_state_batch))

        if torch.cuda.is_available(): # Chuyển batch lên GPU
            state_batch = state_batch.cuda()
            reward_batch = reward_batch.cuda()
            next_state_batch = next_state_batch.cuda()

        # 7. Tính toán giá trị Q dự đoán và Q mục tiêu
        # Tính Q(s, a) dự đoán bởi mạng hiện tại cho các trạng thái trong batch
        q_values = model(state_batch) # Đây là Q(s) dự đoán cho trạng thái s (vì model chỉ output 1 giá trị)

        # Tính giá trị Q mục tiêu (Target Q-value)
        model.eval() # Chuyển sang eval mode để tính giá trị cho trạng thái tiếp theo mà không ảnh hưởng bởi batch norm/dropout (nếu có) và không tính gradient
        with torch.no_grad():
            # Dự đoán Q(s') cho tất cả các trạng thái tiếp theo trong batch
            next_prediction_batch = model(next_state_batch)
        model.train() # Chuyển lại train mode

        # Tính y_batch (target Q-value) dựa trên phương trình Bellman:
        # Nếu done = True (trạng thái s' là cuối cùng): target = r
        # Nếu done = False: target = r + gamma * Q(s')
        # Lưu ý: Đây là một phiên bản đơn giản hóa của Bellman, thường là r + gamma * max_a' Q(s', a').
        # Việc model chỉ output 1 giá trị Q(s) ngụ ý rằng giá trị này đại diện cho giá trị tối ưu có thể đạt được từ trạng thái đó.
        y_batch = torch.cat(
            tuple(reward if done else reward + opt.gamma * prediction for reward, done, prediction in
                  zip(reward_batch, done_batch, next_prediction_batch)))[:, None]

        # 8. Cập nhật mạng nơ-ron
        optimizer.zero_grad() # Xóa gradient cũ
        loss = criterion(q_values, y_batch) # Tính MSE loss giữa Q dự đoán (q_values) và Q mục tiêu (y_batch)
        loss.backward() # Lan truyền ngược lỗi để tính gradient
        optimizer.step() # Cập nhật trọng số mạng dựa trên gradient và thuật toán Adam

        # --- Logging và Lưu trữ ---
        print("Epoch: {}/{}, Action: {}, Score: {}, Tetrominoes {}, Cleared lines: {}".format(
            epoch,
            opt.num_epochs,
            action, # Hành động cuối cùng dẫn đến kết thúc game
            final_score,
            final_tetrominoes,
            final_cleared_lines))
        writer.add_scalar('Train/Score', final_score, epoch - 1) # Ghi log điểm số
        writer.add_scalar('Train/Tetrominoes', final_tetrominoes, epoch - 1) # Ghi log số khối
        writer.add_scalar('Train/Cleared lines', final_cleared_lines, epoch - 1) # Ghi log số hàng xóa

        if epoch > 0 and epoch % opt.save_interval == 0: # Lưu model định kỳ
            torch.save(model, "{}/tetris_{}".format(opt.saved_path, epoch))

    # Lưu model cuối cùng sau khi hoàn thành tất cả các epoch
    torch.save(model, "{}/tetris".format(opt.saved_path))


if __name__ == "__main__":
    opt = get_args() # Lấy tham số
    train(opt) # Bắt đầu huấn luyện
