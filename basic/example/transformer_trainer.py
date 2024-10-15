import torch

from translate_dataset import load_data_nmt, truncate_pad
from basic.model.transformer.block import TransformerEncoder, TransformerDecoder
from basic.abstract import EncoderDecoder
from basic.loss import MaskedSoftmaxCELoss
from basic.metric import bleu


def xavier_init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
    if type(m) == torch.nn.GRU:
        for param in m._flat_weights_names:
            if "weight" in param:
                torch.nn.init.xavier_uniform_(m._parameters[param])


def train(model, data_loader, lr, num_epochs, tgt_vocab, device):
    """训练序列到序列模型"""

    model.apply(xavier_init_weights)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss = MaskedSoftmaxCELoss()
    model.train()
    total_loss, total_tokens = 0, 0
    for epoch in range(num_epochs):
        for batch in data_loader:
            optimizer.zero_grad()
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
            bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0],
                               device=device).reshape(-1, 1)
            dec_input = torch.cat([bos, Y[:, :-1]], 1)
            Y_hat, _ = model(X, dec_input, X_valid_len)
            l = loss(Y_hat, Y, Y_valid_len)
            l.sum().backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            num_tokens = Y_valid_len.sum()
            optimizer.step()
            with torch.no_grad():
                total_loss += l.sum()
                total_tokens += num_tokens
        if (epoch + 1) % 10 == 0:
            print(f'epoch {epoch + 1}, 'f'loss {total_loss / total_tokens:.3f}')


def predict_fn(model, src_sentence, src_vocab, tgt_vocab, num_steps, device):
    """序列到序列模型的预测"""
    # 在预测时将net设置为评估模式
    model.eval()
    src_tokens = src_vocab[src_sentence.lower().split(' ')] + [
        src_vocab['<eos>']]
    enc_valid_len = torch.tensor([len(src_tokens)], device=device)
    src_tokens = truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])
    # shape: (1, seq_len)
    enc_X = torch.unsqueeze(
        torch.tensor(src_tokens, dtype=torch.long, device=device), dim=0)
    enc_outputs = model.encoder(enc_X, enc_valid_len)
    dec_state = model.decoder.init_state(enc_outputs, enc_valid_len)
    # shape: (1, 1)
    dec_X = torch.unsqueeze(torch.tensor(
        [tgt_vocab['<bos>']], dtype=torch.long, device=device), dim=0)
    output_seq = []
    state = dec_state
    for _ in range(num_steps):
        # Y shape: (1, 1, vocab_size)
        Y, state = model.decoder(dec_X, state)
        # 我们使用具有预测最高可能性的词元，作为解码器在下一时间步的输入
        dec_X = Y.argmax(dim=2)
        pred = dec_X.squeeze(dim=0).type(torch.int32).item()
        # 一旦序列结束词元被预测，输出序列的生成就完成了
        if pred == tgt_vocab['<eos>']:
            break
        output_seq.append(pred)
    return ' '.join(tgt_vocab.to_tokens(output_seq))


if __name__ == "__main__":
    num_hiddens, num_layers, dropout = 32, 2, 0.1
    batch_size, num_steps = 64, 10
    lr, num_epochs, device = 0.005, 400, 'cuda'
    ffn_num_input, ffn_num_hiddens, num_heads = 32, 64, 4
    key_size, query_size, value_size = 32, 32, 32
    norm_shape = [32]
    train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size, num_steps)
    encoder = TransformerEncoder(
        len(src_vocab), key_size, query_size, value_size, num_hiddens,
        norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
        num_layers, dropout)
    decoder = TransformerDecoder(
        len(tgt_vocab), key_size, query_size, value_size, num_hiddens,
        norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
        num_layers, dropout)
    net = EncoderDecoder(encoder, decoder)
    train(net, train_iter, lr, num_epochs, tgt_vocab, device)

    engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .', 'drop it !', 'i\'m fit']
    fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .', 'laissez-le tomber !',
            'je suis en forme .']
    for eng, fra in zip(engs, fras):
        translation = predict_fn(net, eng, src_vocab, tgt_vocab, num_steps, device)
        print(f'{eng} => {translation}, bleu {bleu(translation, fra, k=2):.3f}')
    pass
