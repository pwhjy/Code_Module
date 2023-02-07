"""
train ptr for sort
"""
import argparse
import random
import warnings
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from model import PtrNetwork
from torch.optim import Adam
import torch.nn.functional as F
import numpy as np
import itertools

parser = argparse.ArgumentParser(description="PtrNet-Sorting-Integer")

parser.add_argument('--low', type=int, default=0, help='lowest value in dataset(default is 0)')
parser.add_argument('--high', type=int, default=100, help='highest value in dataset (default: 100)')
parser.add_argument('--min-length', type=int, default=5, help='minimum length of sequences (default: 5)')
parser.add_argument('--max-length', type=int, default=10, help='maximum length of sequences (default: 20)')
parser.add_argument('--train-samples', type=int, default=100000, help='number of samples in train set (default: 100000)')
parser.add_argument('--test-samples', type=int, default=1000, help='number of samples in test set (default: 1000)')

parser.add_argument('--emb-dim', type=int, default=8, help='embedding dimension (default: 8)')
parser.add_argument('--batch-size', type=int, default=256, help='input batch size for training (default: 256)')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train (default: 100)')

parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 1e-3)')
parser.add_argument('--wd', default=1e-5, type=float, help='weight decay (default: 1e-5)')

parser.add_argument('--workers', type=int, default=4, help='number of data loading workers (default: 4)')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')

class IntegerSortDataset(Dataset):
	def __init__(self, num_samples=10000, low=0, high=100, min_len=1, max_len=10, seed=1):

		self.prng = np.random.RandomState(seed=seed)
		self.input_dim = high

		# Here, we assuming that the shape of each sample is a list of list of a single integer, e.g., [[10], [3], [5], [0]]
		# It is for an easier extension later even though it is not necessary for this simple sorting example
		self.seqs = [list(map(lambda x: [x], self.prng.choice(np.arange(low, high), size=self.prng.randint(min_len, max_len+1)).tolist())) for _ in range(num_samples)]
		self.labels = [sorted(range(len(seq)), key=seq.__getitem__) for seq in self.seqs]

	def __getitem__(self, index):
		seq = self.seqs[index]
		label = self.labels[index]

		len_seq = len(seq)
		row_col_index = list(zip(*[(i, number) for i, numbers in enumerate(seq) for number in numbers]))
		num_values = len(row_col_index[0])

		i = torch.LongTensor(row_col_index)
		v = torch.FloatTensor([1]*num_values)
		data = torch.sparse.FloatTensor(i, v, torch.Size([len_seq, self.input_dim]))

		return data, len_seq, label

	def __len__(self):
		return len(self.seqs)


def sparse_seq_collate_fn(batch):
	batch_size = len(batch)

	sorted_seqs, sorted_lengths, sorted_labels = zip(*sorted(batch, key=lambda x: x[1], reverse=True))

	padded_seqs = [seq.resize_as_(sorted_seqs[0]) for seq in sorted_seqs]

	# (Sparse) batch_size X max_seq_len X input_dim
	seq_tensor = torch.stack(padded_seqs)

	# batch_size
	length_tensor = torch.LongTensor(sorted_lengths)

	padded_labels = list(zip(*(itertools.zip_longest(*sorted_labels, fillvalue=-1))))

	# batch_size X max_seq_len (-1 padding)
	label_tensor = torch.LongTensor(padded_labels).view(batch_size, -1)

	# TODO: Currently, PyTorch DataLoader with num_workers >= 1 (multiprocessing) does not support Sparse Tensor
	# TODO: Meanwhile, use a dense tensor when num_workers >= 1.
	seq_tensor = seq_tensor.to_dense()

	return seq_tensor, length_tensor, label_tensor


class AverageMeter(object):
    def __init__(self) -> None:
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def masked_accuracy(output, target, mask):
    with torch.no_grad():
        masked_output = torch.masked_select(output, mask)
        masked_target = torch.masked_select(target, mask)
        accuracy = masked_output.eq(masked_target).float().mean()
        return accuracy
    
def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed) # random的随机种子
        torch.manual_seed(args.seed) # torch的随机种子
        cudnn.deterministic = True # cuda的随机种子，设置为True，每次返回的卷积算法将是确定的，即默认算法
        warnings.warn("")

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    cudnn.benchmark = True if use_cuda else False # 通过如上设置让内置的cuDNN的auto-tuner自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题
    train_set = IntegerSortDataset(num_samples=args.train_samples,high=args.high,
                                   min_len=args.min_length, max_len=args.max_length, seed=1) # 获取数据集
    train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, collate_fn=sparse_seq_collate_fn)
    
    test_set = IntegerSortDataset(num_samples=args.test_samples, high=args.high,
                                  min_len=args.min_length, max_len=args.max_length, seed=2)
    test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, collate_fn=sparse_seq_collate_fn)
    
    model = PtrNetwork(input_dim=args.high, embedding_dim=args.emb_dim, hidden_size=args.emb_dim).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    train_loss = AverageMeter()
    train_accuracy = AverageMeter()
    test_loss = AverageMeter()
    test_accuracy = AverageMeter()

    for epoch in range(args.epochs):
        model.train()
        for batch_idx, (seq, length, target) in enumerate(train_loader):
            seq, length, target = seq.to(device), length.to(device), target.to(device)

            optimizer.zero_grad()
            log_pointer_score, argmax_pointer, mask = model(seq, length)
            unrolled = log_pointer_score.view(-1, log_pointer_score.size(-1))
            loss = F.nll_loss(unrolled, target.view(-1), ignore_index=-1)
            # nll_loss 负对数似然损失 (Negative Log Likelihood Loss)，直接将最大化似然取负数，就是最小化损失了。
            assert not np.isnan(loss.item()), 'Model diverged with loss = NaN'
            loss.backward()
            optimizer.step()
            train_loss.update(loss.item(), seq.size(0))
            mask = mask[:,0,:] # (3,4,5)->(3,5)
            train_accuracy.update(masked_accuracy(argmax_pointer, target, mask).item(), mask.int().sum().item())
             
            # output
            if batch_idx % 20 == 0:
                print('Epoch {}: Train [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {:.6f}'
                      .format(epoch, batch_idx * len(seq), len(train_loader.dataset),
                            100. * batch_idx / len(train_loader), train_loss.avg, train_accuracy.avg))

        # Test
        model.eval()
        for seq, length, target in test_loader:
            seq, length, target = seq.to(device), length.to(device), target.to(device)
            log_pointer_score, argmax_pointer, mask = model(seq, length)

            unrolled = log_pointer_score.view(-1, log_pointer_score.size(-1))
            loss = F.nll_loss(unrolled, target.view(-1), ignore_index=-1)
            assert not np.isnan(loss.item()), 'Model diverged with loss = NaN'
            test_loss.update(loss.item(), seq.size(0))
            mask = mask[:,0,:]
            test_accuracy.update(masked_accuracy(argmax_pointer, target, mask).item(), mask.int().sum().item())
        print('Epoch {}: Test\tLoss: {:.6f}\tAccuracy: {:.6f}'.format(epoch, test_loss.avg, test_accuracy.avg))

if __name__ == "__main__":
    main()