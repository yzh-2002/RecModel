import torch, math

from dataset import load_data_time_machine
from basic.model.rnn import RNN
from basic.model.gru import GRU


def predict_fn(prefix, num_preds, model, vocab, device):
    """在prefix后面生成新字符"""
    model.eval()
    outputs = [vocab[prefix[0]]]
    # 获取上一时间步的输出作为当前时间步的输入
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape((1, 1))
    state = model.init_state(batch_size=1)
    for y in prefix[1:]:  # 预热期，更新模型的隐藏状态
        _, state = model(get_input(), state)
        outputs.append(vocab[y])
    for _ in range(num_preds):  # 预测num_preds步
        y, state = model(get_input(), state)
        outputs.append(int(y.argmax(dim=1).reshape(1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])


def train_epoch(model, train_loader, loss, optimizer, device, use_random_iter):
    model.train()
    total_loss, total_tokens = 0, 0
    state = None
    for X, Y in train_loader:
        if state is None or use_random_iter:
            # 在第一次迭代或使用随机抽样时初始化state
            # 顺序分区抽样时，由于相邻batch的序列相邻，故可保留之前的state
            state = model.init_state(batch_size=X.shape[0])
        else:
            # 顺序分区，state依赖于在此之前所有batch的计算，不利于梯度计算，故每个batch需要将其从计算图分离
            # 使得state的梯度计算总是限制在一个batch内
            state.detach_()
        y = Y.T.reshape(-1)
        X, y = X.to(device), y.to(device)
        y_hat, state = model(X, state)
        l = loss(y_hat, y.long()).mean()
        total_loss += l.item() * len(y)
        total_tokens += len(y)
        optimizer.zero_grad()
        l.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
    return math.exp(total_loss / total_tokens)


def train(model, train_loader, vocab, lr, num_epochs, device,
          use_random_iter=False):
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr)
    predict = lambda prefix: predict_fn(prefix, 50, model, vocab, device)
    for epoch in range(num_epochs):
        ppl = train_epoch(model, train_loader, loss, optimizer, device, use_random_iter)
        if (epoch + 1) % 10 == 0:
            print(predict('time traveller'))
    print(f'困惑度 {ppl:.1f} {str(device)}')
    print(predict('time traveller'))
    print(predict('traveller'))


if __name__ == "__main__":
    batch_size, num_steps = 32, 35
    train_iter, vocab = load_data_time_machine(batch_size, num_steps)
    num_hiddens = 512
    # model = RNN(len(vocab), num_hiddens).to('cuda')
    model = GRU(len(vocab), num_hiddens).to('cuda')
    num_epochs, lr = 500, 1
    train(model, train_iter, vocab, lr, num_epochs, 'cuda', use_random_iter=True)
