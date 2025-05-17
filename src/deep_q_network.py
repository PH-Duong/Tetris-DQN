import torch.nn as nn # Import module nn từ thư viện PyTorch để xây dựng mạng neural

class DeepQNetwork(nn.Module): # Định nghĩa lớp DeepQNetwork kế thừa từ nn.Module của PyTorch
    def __init__(self): # Hàm khởi tạo của lớp
        super(DeepQNetwork, self).__init__() # Gọi hàm khởi tạo của lớp cha (nn.Module)

        # Định nghĩa các lớp (layers) của mạng neural.
        # Ở đây sử dụng các lớp fully connected (Linear) và hàm kích hoạt ReLU.
        # Input của mạng là một vector 4 chiều (đại diện cho trạng thái của game Tetris: số hàng đã xóa, số lỗ, độ gồ ghề, chiều cao tổng).
        # Output của mạng là một giá trị Q duy nhất cho trạng thái đầu vào đó.

        # Lớp fully connected thứ nhất: 4 input features, 64 output features, theo sau là hàm kích hoạt ReLU.
        # nn.Sequential cho phép nhóm các module lại với nhau.
        # nn.ReLU(inplace=True) thực hiện ReLU trực tiếp trên input, giúp tiết kiệm bộ nhớ.
        self.conv1 = nn.Sequential(nn.Linear(4, 64), nn.ReLU(inplace=True))
        # Lớp fully connected thứ hai: 64 input features, 64 output features, theo sau là hàm kích hoạt ReLU.
        self.conv2 = nn.Sequential(nn.Linear(64, 64), nn.ReLU(inplace=True))
        # Lớp fully connected thứ ba (lớp output): 64 input features, 1 output feature (giá trị Q).
        self.conv3 = nn.Sequential(nn.Linear(64, 1))

        self._create_weights() # Gọi hàm để khởi tạo trọng số cho các lớp Linear

    def _create_weights(self): # Hàm khởi tạo trọng số cho các lớp Linear
        for m in self.modules(): # Duyệt qua tất cả các module (lớp) trong mạng
            if isinstance(m, nn.Linear): # Nếu module là một lớp Linear
                # Sử dụng phương pháp Xavier (uniform) để khởi tạo trọng số (weights).
                # Phương pháp này giúp giữ cho phương sai của các activation ổn định qua các lớp.
                nn.init.xavier_uniform_(m.weight)
                # Khởi tạo bias bằng 0.
                nn.init.constant_(m.bias, 0)

    def forward(self, x): # Hàm thực hiện quá trình truyền thẳng (forward pass) của mạng
        # x là input tensor (trạng thái của game)
        x = self.conv1(x) # Cho input qua lớp conv1
        x = self.conv2(x) # Cho output của conv1 qua lớp conv2
        x = self.conv3(x) # Cho output của conv2 qua lớp conv3 để nhận giá trị Q

        return x # Trả về giá trị Q dự đoán