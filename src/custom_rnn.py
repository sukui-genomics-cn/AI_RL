import os
import logging
import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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


def get_dataset(root="./data/mnist/", batch_size=64):
    DOWNLOAD_MNIST = False
    if not (os.path.exists(root)) or not os.listdir(root):
        logger.info("Downloading MNIST dataset")
        DOWNLOAD_MNIST = True
    train_data = MNIST(
        root=root,
        train=True,
        transform=transforms.ToTensor(),
        download=DOWNLOAD_MNIST
    )

    logger.info(f"Number of training examples: {len(train_data)}, shape: {train_data.data.shape}, "
                f"label.shape:{train_data.targets.shape}")
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=True
    )

    test_data = MNIST(
        root=root,
        train=False,
        transform=transforms.ToTensor()
    )

    # split the test data into two parts
    test_size = int(0.7 * len(test_data))
    val_size = len(test_data) - test_size
    test_data, val_data = random_split(test_data, [test_size, val_size])
    logger.info(f"val and test data split: {len(val_data)}, {len(test_data)}")

    test_loader = DataLoader(
        dataset=test_data,
        batch_size=batch_size,
        shuffle=False  # 测试集通常不需要 shuffle
    )

    val_loader = DataLoader(
        dataset=val_data,
        batch_size=batch_size,
        shuffle=True  # 验证集可以 shuffle
    )
    return train_loader, val_loader, test_loader


class RNNnet(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim):
        super(RNNnet, self).__init__()
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )
        self.out = nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        r_out, (h_n, h_c) = self.rnn(x, None)
        # choose r_out at the last time step
        out = self.out(r_out[:, -1, :])
        return out


def train(model, train_loader, eval_loader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        for step, (input_x, labels) in enumerate(train_loader):
            input_x = input_x.view(-1, 28, 28)
            logits = model(input_x)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % 50 == 0:
                predict_all, labels_all = evaluate(model, eval_loader)
                accuracy = (predict_all == labels_all).sum().item() / labels_all.size(0)
                logger.info(f'Epoch: {epoch}| train loss: {loss.data.numpy():.4f}| val accuracy: {accuracy:.4f}')


def evaluate(model, eval_loader):
    predict_all = []
    labels_all = []
    for i, (eval_x, eval_y) in enumerate(eval_loader):
        eval_x = eval_x.view(-1, 28, 28)
        eval_logits = model(eval_x)
        pred_y = torch.max(eval_logits, 1)[1]
        predict_all.extend(pred_y)
        labels_all.extend(eval_y)
    predict_all = torch.stack(predict_all, dim=0)
    labels_all = torch.stack(labels_all, dim=0)
    return predict_all, labels_all


def main():
    input_size = 28
    hidden_size = 128
    output_size = 10
    num_epochs = 5
    LR = 0.001

    train_loader, val_loader, test_loader = get_dataset()

    model = RNNnet(
        input_size=input_size,
        output_size=output_size,
        hidden_dim=hidden_size
    )
    logger.info(f'Model:\n{model}')
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_func = nn.CrossEntropyLoss()
    train(model, train_loader, val_loader, loss_func, optimizer, num_epochs)

    # print 10 prediction for test data
    predict_all, labels_all = evaluate(model, test_loader)
    accuracy = (predict_all == labels_all).sum().item() / labels_all.size(0)
    logger.info(f'Test accuracy: {accuracy}')


if __name__ == '__main__':
    main()
    # import torch.optim as optim
    #
    # input_size = 10  # 输入特征维度
    # hidden_size = 20  # 隐藏层维度
    # output_size = 5  # 输出特征维度
    # batch_size = 3  # 批量大小
    # sequence_length = 4  # 序列长度
    # learning_rate = 0.01
    # num_epochs = 100
    #
    # rnn = SimpleRNN(input_size, hidden_size, output_size)
    #
    # criterion = nn.MSELoss()
    # optimizer = optim.Adam(rnn.parameters(), lr=learning_rate)
    #
    # inputs = torch.randn(sequence_length, batch_size, input_size)
    # targets = torch.randn(sequence_length, batch_size, output_size)
    #
    # for epoch in range(num_epochs):
    #     hidden = rnn.init_hidden(batch_size)
    #     optimizer.zero_grad()
    #
    #     loss = 0
    #     for i in range(sequence_length):
    #         output, hidden = rnn(inputs[i], hidden)
    #         loss += criterion(output, targets[i])
    #
    #     loss.backward()
    #     optimizer.step()
    #
    #     if (epoch + 1) % 10 == 0:
    #         print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
