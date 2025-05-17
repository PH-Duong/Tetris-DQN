import numpy as np # Thư viện cho tính toán số học, đặc biệt là mảng đa chiều
from PIL import Image # Thư viện Pillow (PIL fork) để xử lý ảnh
import cv2 # Thư viện OpenCV để xử lý ảnh và video
from matplotlib import style # Module style từ Matplotlib để tùy chỉnh giao diện đồ thị
import torch # Thư viện PyTorch
import random # Thư viện để sinh số ngẫu nhiên

style.use("ggplot") # Sử dụng style "ggplot" cho Matplotlib


class Tetris: # Định nghĩa lớp Tetris, đại diện cho môi trường game
    # Định nghĩa màu sắc cho các loại khối và ô trống (0)
    piece_colors = [
        (0, 0, 0),       # Màu đen (ô trống)
        (255, 255, 0),   # Màu vàng (khối O)
        (147, 88, 254),  # Màu tím (khối T)
        (54, 175, 144),  # Màu xanh lá cây (khối S)
        (255, 0, 0),     # Màu đỏ (khối Z)
        (102, 217, 238), # Màu xanh dương nhạt (khối I)
        (254, 151, 32),  # Màu cam (khối L)
        (0, 0, 255)      # Màu xanh dương đậm (khối J)
    ]

    # Định nghĩa hình dạng của các khối Tetris (tetrominoes)
    # Mỗi số nguyên dương đại diện cho một loại khối, tương ứng với index trong piece_colors
    pieces = [
        [[1, 1],  # Khối O
         [1, 1]],

        [[0, 2, 0], # Khối T
         [2, 2, 2]],

        [[0, 3, 3], # Khối S
         [3, 3, 0]],

        [[4, 4, 0], # Khối Z
         [0, 4, 4]],

        [[5, 5, 5, 5]], # Khối I

        [[0, 0, 6], # Khối L
         [6, 6, 6]],

        [[7, 0, 0], # Khối J
         [7, 7, 7]]
    ]

    def __init__(self, height=20, width=10, block_size=20): # Hàm khởi tạo của lớp Tetris
        self.height = height # Chiều cao của bảng game (số hàng)
        self.width = width # Chiều rộng của bảng game (số cột)
        self.block_size = block_size # Kích thước của một ô vuông (block) khi render

        # Tạo một bảng phụ để hiển thị thông tin (điểm, số khối,...) bên cạnh bảng game chính
        self.extra_board = np.ones((self.height * self.block_size, self.width * int(self.block_size / 2), 3),
                                   dtype=np.uint8) * np.array([204, 204, 255], dtype=np.uint8) # Màu nền hồng nhạt
        self.text_color = (200, 20, 220) # Màu chữ để hiển thị thông tin

        self.reset() # Gọi hàm reset để khởi tạo trạng thái ban đầu của game

    def reset(self): # Hàm reset game về trạng thái ban đầu
        # Khởi tạo bảng game là một ma trận 0 (tất cả các ô đều trống)
        self.board = [[0] * self.width for _ in range(self.height)]
        self.score = 0 # Điểm số ban đầu
        self.tetrominoes = 0 # Số lượng khối đã xuất hiện
        self.cleared_lines = 0 # Số hàng đã xóa
        # Tạo một "túi" chứa các loại khối, xáo trộn để đảm bảo tính ngẫu nhiên khi chọn khối tiếp theo
        self.bag = list(range(len(self.pieces)))
        random.shuffle(self.bag)
        # Lấy một khối từ túi
        self.ind = self.bag.pop() # self.ind là index của khối hiện tại trong self.pieces
        self.piece = [row[:] for row in self.pieces[self.ind]] # Lấy hình dạng của khối hiện tại (tạo bản sao)
        # Đặt vị trí ban đầu cho khối ở giữa phía trên bảng
        self.current_pos = {"x": self.width // 2 - len(self.piece[0]) // 2, "y": 0}
        self.gameover = False # Cờ báo hiệu game kết thúc
        # Trả về các đặc trưng trạng thái của bảng game ban đầu
        return self.get_state_properties(self.board)

    def rotate(self, piece): # Hàm xoay một khối (piece) theo chiều kim đồng hồ
        num_rows_orig = num_cols_new = len(piece) # Số hàng của khối ban đầu = số cột của khối mới
        num_rows_new = len(piece[0]) # Số cột của khối ban đầu = số hàng của khối mới
        rotated_array = [] # Ma trận chứa khối đã xoay

        for i in range(num_rows_new):
            new_row = [0] * num_cols_new
            for j in range(num_cols_new):
                # Công thức xoay ma trận 90 độ
                new_row[j] = piece[(num_rows_orig - 1) - j][i]
            rotated_array.append(new_row)
        return rotated_array # Trả về khối đã xoay

    def get_state_properties(self, board): # Hàm tính toán các đặc trưng của trạng thái bảng game hiện tại
        # Các đặc trưng này sẽ là input cho mạng DQN
        lines_cleared, board = self.check_cleared_rows(board) # Số hàng vừa xóa và bảng game sau khi xóa
        holes = self.get_holes(board) # Số lượng "lỗ" trên bảng
        bumpiness, height = self.get_bumpiness_and_height(board) # Độ gồ ghề và chiều cao tổng của các cột

        # Trả về một tensor PyTorch chứa các đặc trưng này
        return torch.FloatTensor([lines_cleared, holes, bumpiness, height])

    def get_holes(self, board): # Hàm tính số lượng lỗ trên bảng
        # Lỗ được định nghĩa là ô trống nằm dưới một ô đã có khối
        num_holes = 0
        for col in zip(*board): # Duyệt qua từng cột của bảng (zip(*board) để chuyển vị hàng thành cột)
            row = 0
            # Tìm ô có khối đầu tiên từ trên xuống trong cột hiện tại
            while row < self.height and col[row] == 0:
                row += 1
            # Đếm số ô trống (giá trị 0) nằm dưới ô có khối đó trong cùng cột
            num_holes += len([x for x in col[row + 1:] if x == 0])
        return num_holes

    def get_bumpiness_and_height(self, board): # Hàm tính độ gồ ghề và chiều cao tổng của các cột
        board = np.array(board) # Chuyển bảng game sang dạng mảng NumPy để dễ tính toán
        mask = board != 0 # Tạo mặt nạ boolean, True ở những vị trí có khối
        # Tính chiều cao của mỗi cột (tính từ đáy lên, ô trống là 0, có khối là 1)
        # np.argmax(mask, axis=0) tìm index của True đầu tiên trong mỗi cột (chiều cao tính từ trên xuống)
        # self.height - ... để đổi thành chiều cao tính từ dưới lên
        invert_heights = np.where(mask.any(axis=0), np.argmax(mask, axis=0), self.height)
        heights = self.height - invert_heights
        total_height = np.sum(heights) # Tổng chiều cao của tất cả các cột

        # Tính độ gồ ghề: tổng chênh lệch tuyệt đối về chiều cao giữa các cột liền kề
        currs = heights[:-1] # Chiều cao các cột từ đầu đến gần cuối
        nexts = heights[1:] # Chiều cao các cột từ thứ hai đến cuối
        diffs = np.abs(currs - nexts) # Chênh lệch tuyệt đối
        total_bumpiness = np.sum(diffs) # Tổng độ gồ ghề
        return total_bumpiness, total_height

    def get_next_states(self): # Hàm tạo ra tất cả các trạng thái tiếp theo có thể có
        # Trạng thái tiếp theo được định nghĩa là trạng thái của bảng sau khi khối hiện tại được đặt xuống
        # ở một vị trí và hướng xoay cụ thể.
        states = {} # Dictionary lưu trữ các trạng thái tiếp theo: {(vị trí x, số lần xoay): tensor_đặc_trưng_trạng_thái}
        piece_id = self.ind # ID của khối hiện tại
        curr_piece = [row[:] for row in self.piece] # Lấy bản sao của khối hiện tại

        # Xác định số lần xoay tối đa cho từng loại khối để tránh dư thừa
        if piece_id == 0:  # Khối O (hình vuông) chỉ có 1 hướng xoay
            num_rotations = 1
        elif piece_id == 2 or piece_id == 3 or piece_id == 4: # Khối S, Z, I có 2 hướng xoay hiệu quả
            num_rotations = 2
        else: # Các khối còn lại (T, L, J) có 4 hướng xoay
            num_rotations = 4

        for i in range(num_rotations): # Duyệt qua các hướng xoay có thể
            valid_xs = self.width - len(curr_piece[0]) # Số vị trí x hợp lệ mà khối có thể được đặt
            for x in range(valid_xs + 1): # Duyệt qua các vị trí x có thể
                piece_to_check = [row[:] for row in curr_piece] # Tạo bản sao của khối ở hướng xoay hiện tại
                pos = {"x": x, "y": 0} # Đặt khối ở đầu bảng (y=0) tại vị trí x
                # Di chuyển khối xuống dưới cho đến khi chạm đáy hoặc chạm khối khác
                while not self.check_collision(piece_to_check, pos):
                    pos["y"] += 1
                # Sau vòng lặp, pos["y"] là vị trí y ngay trước khi va chạm.
                # Hàm truncate xử lý trường hợp khối bị kẹt (overflow) khi mới xuất hiện.
                self.truncate(piece_to_check, pos)
                # Tạo một bản sao của bảng game và đặt khối vào đó để mô phỏng
                temp_board = self.store(piece_to_check, pos)
                # Lưu trữ đặc trưng của trạng thái bảng mô phỏng này
                states[(x, i)] = self.get_state_properties(temp_board)
            curr_piece = self.rotate(curr_piece) # Xoay khối cho lần lặp tiếp theo
        return states # Trả về dictionary các trạng thái tiếp theo

    def get_current_board_state(self): # Hàm lấy trạng thái hiện tại của bảng game bao gồm cả khối đang rơi
        board = [x[:] for x in self.board] # Tạo bản sao của bảng game hiện tại
        # Vẽ khối đang rơi vào bản sao của bảng
        for y_offset in range(len(self.piece)):
            for x_offset in range(len(self.piece[y_offset])):
                if self.piece[y_offset][x_offset]: # Nếu ô đó của khối không phải là ô trống
                    # Tính toán tọa độ tuyệt đối trên bảng
                    board_y = y_offset + self.current_pos["y"]
                    board_x = x_offset + self.current_pos["x"]
                    # Đảm bảo không vẽ ra ngoài biên (mặc dù logic di chuyển và va chạm nên xử lý điều này)
                    if 0 <= board_y < self.height and 0 <= board_x < self.width:
                         board[board_y][board_x] = self.piece[y_offset][x_offset]
        return board

    def new_piece(self): # Hàm tạo một khối mới
        # Nếu túi khối rỗng, tạo túi mới và xáo trộn
        if not len(self.bag):
            self.bag = list(range(len(self.pieces)))
            random.shuffle(self.bag)
        self.ind = self.bag.pop() # Lấy index của khối mới từ túi
        self.piece = [row[:] for row in self.pieces[self.ind]] # Lấy hình dạng khối mới
        # Đặt vị trí ban đầu cho khối mới
        self.current_pos = {"x": self.width // 2 - len(self.piece[0]) // 2,
                            "y": 0
                            }
        # Kiểm tra va chạm ngay khi khối mới xuất hiện (game over nếu không có chỗ cho khối mới)
        if self.check_collision(self.piece, self.current_pos):
            self.gameover = True

    def check_collision(self, piece, pos): # Hàm kiểm tra va chạm nếu khối di chuyển xuống một bước
        future_y = pos["y"] + 1 # Vị trí y dự kiến sau khi di chuyển xuống
        for y_offset in range(len(piece)):
            for x_offset in range(len(piece[y_offset])):
                if piece[y_offset][x_offset]: # Chỉ kiểm tra các ô có khối
                    # Tính toán tọa độ tuyệt đối trên bảng của ô khối đó ở vị trí dự kiến
                    board_y = future_y + y_offset
                    board_x = pos["x"] + x_offset
                    # Kiểm tra va chạm:
                    # 1. Va chạm với đáy bảng (board_y > self.height - 1)
                    # 2. Va chạm với một khối đã có sẵn trên bảng (self.board[board_y][board_x] != 0)
                    if board_y > self.height - 1 or \
                       (0 <= board_x < self.width and self.board[board_y][board_x]): # Đảm bảo board_x hợp lệ trước khi truy cập self.board
                        return True # Có va chạm
        return False # Không có va chạm

    def truncate(self, piece, pos): # Hàm xử lý trường hợp khối bị "overflow" (kẹt) khi mới xuất hiện hoặc đặt xuống
        # Trường hợp này xảy ra khi một phần của khối được đặt vào vị trí đã có khối khác.
        # Hàm này sẽ cố gắng "cắt bớt" phần trên của khối để nó vừa vặn.
        gameover = False # Cờ báo hiệu game over nếu không thể cắt bớt
        last_collision_row = -1 # Lưu lại hàng cuối cùng của khối (tính từ trên) mà có va chạm
        for y_offset in range(len(piece)):
            for x_offset in range(len(piece[y_offset])):
                # Kiểm tra xem ô của khối (piece[y_offset][x_offset]) có trùng với ô đã có trên bảng (self.board) không
                if piece[y_offset][x_offset] and self.board[pos["y"] + y_offset][pos["x"] + x_offset]:
                    if y_offset > last_collision_row:
                        last_collision_row = y_offset # Cập nhật hàng va chạm cuối cùng

        # Nếu có va chạm (last_collision_row > -1) và vị trí đặt khối khiến phần trên của nó
        # vượt ra ngoài bảng (pos["y"] - (len(piece) - last_collision_row) < 0),
        # thì cần cắt bớt khối từ trên xuống.
        if pos["y"] - (len(piece) - last_collision_row) < 0 and last_collision_row > -1:
            while last_collision_row >= 0 and len(piece) > 1: # Lặp cho đến khi hết va chạm hoặc khối chỉ còn 1 hàng
                gameover = True # Đặt cờ game over vì có sự "overflow"
                last_collision_row = -1 # Reset để kiểm tra lại sau khi xóa hàng
                del piece[0] # Xóa hàng trên cùng của khối
                # Kiểm tra lại va chạm với khối đã bị cắt bớt
                for y_offset in range(len(piece)):
                    for x_offset in range(len(piece[y_offset])):
                        if piece[y_offset][x_offset] and self.board[pos["y"] + y_offset][pos["x"] + x_offset] and y_offset > last_collision_row:
                            last_collision_row = y_offset
        return gameover # Trả về True nếu xảy ra game over do overflow không xử lý được

    def store(self, piece, pos): # Hàm lưu khối vào bảng game (tạo một bản sao của bảng với khối đã được đặt)
        board = [x[:] for x in self.board] # Tạo bản sao của bảng game hiện tại
        for y_offset in range(len(piece)):
            for x_offset in range(len(piece[y_offset])):
                if piece[y_offset][x_offset] and not board[y_offset + pos["y"]][x_offset + pos["x"]]: # Nếu ô của khối có giá trị và ô trên bảng trống
                    board[y_offset + pos["y"]][x_offset + pos["x"]] = piece[y_offset][x_offset] # Đặt khối vào bảng
        return board # Trả về bảng game mới với khối đã được đặt

    def check_cleared_rows(self, board): # Hàm kiểm tra và xóa các hàng đã đầy
        to_delete = [] # Danh sách chứa chỉ số các hàng cần xóa
        # Duyệt bảng từ dưới lên (board[::-1]) để việc xóa không ảnh hưởng đến chỉ số các hàng chưa kiểm tra
        for i, row in enumerate(board[::-1]):
            if 0 not in row: # Nếu hàng không chứa ô trống (giá trị 0), tức là hàng đã đầy
                to_delete.append(len(board) - 1 - i) # Lưu chỉ số của hàng đầy (chỉ số gốc từ trên xuống)
        if len(to_delete) > 0: # Nếu có hàng cần xóa
            board = self.remove_row(board, to_delete) # Gọi hàm xóa hàng
        return len(to_delete), board # Trả về số hàng đã xóa và bảng game sau khi xóa

    def remove_row(self, board, indices): # Hàm xóa các hàng theo danh sách chỉ số cho trước
        # Xóa từ chỉ số cao nhất xuống thấp nhất (indices[::-1]) để tránh lỗi thay đổi chỉ số khi xóa
        for i in indices[::-1]:
            del board[i] # Xóa hàng tại chỉ số i
            # Thêm một hàng trống mới (toàn số 0) vào đầu bảng
            board = [[0 for _ in range(self.width)]] + board
        return board # Trả về bảng game sau khi đã xóa hàng và thêm hàng mới

    def step(self, action, render=True, video=None): # Hàm thực hiện một bước trong game dựa trên hành động được chọn
        # action là một tuple (x, num_rotations) - vị trí x và số lần xoay
        x, num_rotations = action
        self.current_pos = {"x": x, "y": 0} # Đặt vị trí ban đầu cho khối ở (x, 0)
        # Xoay khối theo số lần đã cho
        for _ in range(num_rotations):
            self.piece = self.rotate(self.piece)

        # Di chuyển khối xuống dưới cho đến khi va chạm
        while not self.check_collision(self.piece, self.current_pos):
            self.current_pos["y"] += 1
            if render: # Nếu có yêu cầu render
                self.render(video) # Gọi hàm render để hiển thị game

        # Xử lý trường hợp overflow sau khi khối đã hạ cánh
        overflow = self.truncate(self.piece, self.current_pos)
        if overflow: # Nếu có overflow không xử lý được
            self.gameover = True

        # Lưu khối vào bảng game chính
        self.board = self.store(self.piece, self.current_pos)

        # Kiểm tra và xóa các hàng đã đầy, cập nhật điểm số
        lines_cleared, self.board = self.check_cleared_rows(self.board)
        # Tính điểm: 1 điểm cơ bản + (số hàng xóa)^2 * chiều rộng bảng (thưởng lớn cho nhiều hàng xóa cùng lúc)
        score = 1 + (lines_cleared ** 2) * self.width
        self.score += score
        self.tetrominoes += 1 # Tăng số lượng khối đã sử dụng
        self.cleared_lines += lines_cleared # Tăng tổng số hàng đã xóa

        # Nếu game chưa kết thúc, tạo khối mới
        if not self.gameover:
            self.new_piece()
        # Nếu game kết thúc ở bước này (ví dụ do overflow hoặc khối mới không có chỗ)
        if self.gameover:
            self.score -= 2 # Phạt điểm nhỏ khi game over

        return score, self.gameover # Trả về điểm số của bước này và cờ game over

    def render(self, video=None): # Hàm render (hiển thị) trạng thái game
        # Tạo ảnh từ trạng thái bảng game hiện tại (bao gồm cả khối đang rơi nếu chưa game over)
        if not self.gameover:
            # Lấy màu cho từng ô trong bảng (bao gồm cả khối đang rơi)
            img_data = [self.piece_colors[p] for row in self.get_current_board_state() for p in row]
        else: # Nếu game over, chỉ hiển thị bảng game cố định
            img_data = [self.piece_colors[p] for row in self.board for p in row]

        # Chuyển đổi dữ liệu màu thành mảng NumPy và reshape thành kích thước ảnh
        img = np.array(img_data).reshape((self.height, self.width, 3)).astype(np.uint8)
        img = img[..., ::-1] # Chuyển đổi từ RGB (Pillow) sang BGR (OpenCV)
        img = Image.fromarray(img, "RGB") # Tạo đối tượng ảnh Pillow

        # Resize ảnh theo kích thước block_size
        img = img.resize((self.width * self.block_size, self.height * self.block_size), 0) # 0: NEAREST filter
        img = np.array(img) # Chuyển lại thành mảng NumPy

        # Vẽ lưới cho bảng game
        img[[i * self.block_size for i in range(self.height)], :, :] = 0 # Vẽ đường ngang
        img[:, [i * self.block_size for i in range(self.width)], :] = 0 # Vẽ đường dọc

        # Ghép ảnh bảng game với bảng phụ hiển thị thông tin
        img = np.concatenate((img, self.extra_board), axis=1)

        # Hiển thị thông tin điểm số, số khối, số hàng đã xóa lên bảng phụ
        cv2.putText(img, "Score:", (self.width * self.block_size + int(self.block_size / 2), self.block_size),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=self.text_color)
        cv2.putText(img, str(self.score),
                    (self.width * self.block_size + int(self.block_size / 2), 2 * self.block_size),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=self.text_color)

        cv2.putText(img, "Pieces:", (self.width * self.block_size + int(self.block_size / 2), 4 * self.block_size),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=self.text_color)
        cv2.putText(img, str(self.tetrominoes),
                    (self.width * self.block_size + int(self.block_size / 2), 5 * self.block_size),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=self.text_color)

        cv2.putText(img, "Lines:", (self.width * self.block_size + int(self.block_size / 2), 7 * self.block_size),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=self.text_color)
        cv2.putText(img, str(self.cleared_lines),
                    (self.width * self.block_size + int(self.block_size / 2), 8 * self.block_size),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=self.text_color)

        if video: # Nếu có đối tượng VideoWriter được truyền vào
            video.write(img) # Ghi frame hiện tại vào video

        # Hiển thị cửa sổ game
        cv2.imshow("Deep Q-Learning Tetris", img)
        cv2.waitKey(1) # Chờ 1ms để cửa sổ được cập nhật (quan trọng để hiển thị)