import torch
import torch.nn as nn


class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.h20 = nn.Linear(hidden_size, output_size)
        self.tanh = nn.Tanh()

    def forward(self, x, hidden):
        combined = torch.cat((x, hidden), dim=1)
        hidden = self.tanh(self.i2h(combined))
        output = self.h20(hidden)
        return output, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size)


if __name__ == '__main__':
    import torch.optim as optim

    input_size = 10  # 输入特征维度
    hidden_size = 20  # 隐藏层维度
    output_size = 5  # 输出特征维度
    batch_size = 3  # 批量大小
    sequence_length = 4  # 序列长度
    learning_rate = 0.01
    num_epochs = 100

    rnn = SimpleRNN(input_size, hidden_size, output_size)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(rnn.parameters(), lr=learning_rate)

    inputs = torch.randn(sequence_length, batch_size, input_size)
    targets = torch.randn(sequence_length, batch_size, output_size)

    for epoch in range(num_epochs):
        hidden = rnn.init_hidden(batch_size)
        optimizer.zero_grad()

        loss = 0
        for i in range(sequence_length):
            output, hidden = rnn(inputs[i], hidden)
            loss += criterion(output, targets[i])

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
