import string

import torch
from collections import Counter
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm

from dataloader_text import TextDataset, Collate


class HMM(torch.nn.Module):
    """
    Hidden Markov Model with discrete observations.
    """

    def __init__(self, M, N):
        super(HMM, self).__init__()
        self.M = M  # number of possible observations
        self.N = N  # number of states

        # A
        self.transition_model = TransitionModel(self.N)

        # b(x_t)
        self.emission_model = EmissionModel(self.N, self.M)

        # pi
        self.unnormalized_state_priors = torch.nn.Parameter(torch.randn(self.N))

        # use the GPU
        self.is_cuda = torch.cuda.is_available()
        if self.is_cuda: self.cuda()

    def sample(self, T=10):
        state_priors = torch.nn.functional.softmax(self.unnormalized_state_priors, dim=0)
        transition_matrix = torch.nn.functional.softmax(self.transition_model.unnormalized_transition_matrix, dim=0)
        emission_matrix = torch.nn.functional.softmax(self.emission_model.unnormalized_emission_matrix, dim=1)

        # sample initial state
        z_t = torch.distributions.categorical.Categorical(state_priors).sample().item()
        z = []
        x = []
        z.append(z_t)
        for t in range(0, T):
            # sample emission
            x_t = torch.distributions.categorical.Categorical(emission_matrix[z_t]).sample().item()
            x.append(x_t)

            # sample transition
            z_t = torch.distributions.categorical.Categorical(transition_matrix[:, z_t]).sample().item()
            if t < T - 1: z.append(z_t)

        return x, z

    def forward(self, x, T):
        """
        Compute log p(x) for each example in the batch
        :param x: (batch_size, T) tensor of observations
        :param T: (batch_size) tensor of sequence lengths
        :return:
        """
        if self.is_cuda:
            x = x.cuda()
            T = T.cuda()

        batch_size = x.shape[0]
        T_max = x.shape[1]
        log_state_priors = torch.nn.functional.log_softmax(self.unnormalized_state_priors, dim=0)  # ?
        log_alpha = torch.zeros(batch_size, T_max, self.N)
        log_alpha = log_alpha.to(x.device)
        log_alpha[:, 0, :] = self.emission_model(x[:, 0]) + log_state_priors
        for t in range(1, T_max):
            log_alpha[:, t, :] = self.emission_model(x[:, t]) + self.transition_model(log_alpha[:, t - 1, :])

        # Select the sum for the final timestep (each x may have different length).
        log_sums = log_alpha.logsumexp(dim=2)
        log_probs = torch.gather(log_sums, 1, T.view(-1, 1) - 1)
        return log_probs

    def viterbi(self, x, T):
        """
        Find argmax_z log p(x|z) for each (x) in the batch.
        :param x: (batch size, T_max)
        :param T: (batch size)
        :return:
        """
        if self.is_cuda:
            x = x.cuda()
            T = T.cuda()

        batch_size = x.shape[0]
        T_max = x.shape[1]
        log_state_priors = torch.nn.functional.log_softmax(self.unnormalized_state_priors, dim=0)
        log_delta = torch.zeros(batch_size, T_max, self.N).float()
        psi = torch.zeros(batch_size, T_max, self.N).long()
        if self.is_cuda:
            log_delta = log_delta.cuda()
            psi = psi.cuda()
        log_delta[:, 0, :] = self.emission_model(x[:, 0]) + log_state_priors
        for t in range(1, T_max):
            max_val, argmax_val = self.transition_model.forward_maxmul(log_delta[:, t - 1, :])
            log_delta[:, t, :] = self.emission_model(x[:, t]) + max_val
            psi[:, t, :] = argmax_val

        # Get the log probability of the best path
        log_max = log_delta.max(dim=2)[0]
        best_path_scores = torch.gather(log_max, 1, T.view(-1, 1) - 1)

        # This next part is a bit tricky to parallelize across the batch, so we will do it separately for each example.
        z_start = []
        for i in range(0, batch_size):
            z_start_i = [log_delta[i, T[i] - 1, :].max(dim=0)[1].item()]
            for t in range(T[i] - 1, 0, -1):
                z_t = psi[i, t, z_start_i[0]].item()
                z_start_i.insert(0, z_t)
            z_start.append(z_start_i)
        return z_start, best_path_scores  # return both the best path and its log probability


class TransitionModel(torch.nn.Module):
    def __init__(self, N):
        super(TransitionModel, self).__init__()
        self.N = N
        self.unnormalized_transition_matrix = torch.nn.Parameter(torch.randn(N, N))

    def forward(self, log_alpha):
        """
        Multiply previous timestep's alphas by transition matrix (in log domain)
        :param log_alpha: Tensor of shape (batch size, N)
        :return:
        """
        log_transition_matrix = torch.nn.functional.log_softmax(self.unnormalized_transition_matrix, dim=0)
        # matrix multiplication in the log domain
        out = self.log_domain_matmul(log_transition_matrix, log_alpha.transpose(0, 1)).transpose(0, 1)
        return out

    def forward_maxmul(self, log_alpha):
        log_transition_matrix = torch.nn.functional.log_softmax(self.unnormalized_transition_matrix, dim=0)
        out1, out2 = self.maxmul(log_transition_matrix, log_alpha.transpose(0, 1))
        return out1.transpose(0, 1), out2.transpose(0, 1)

    @staticmethod
    def maxmul(log_A, log_B):
        """
        log_A : m x n
        log_B : n x p
        output : m x p matrix

        Similar to the log domain matrix multiplication,
        this computes out_{i,j} = max_k log_A_{i,k} + log_B_{k,j}
        """
        m = log_A.shape[0]
        n = log_A.shape[1]
        p = log_B.shape[1]

        log_A_expanded = torch.stack([log_A] * p, dim=2)
        log_B_expanded = torch.stack([log_B] * m, dim=0)

        elementwise_sum = log_A_expanded + log_B_expanded
        out1, out2 = torch.max(elementwise_sum, dim=1)

        return out1, out2

    @staticmethod
    def log_domain_matmul(log_A, log_B):
        """
        Normally, a matrix multiplication
        computes out_{i,j} = sum_k A_{i,k} * B_{k,j}
        A log domain matrix multiplication
        :param log_A: m*n
        :param log_B: n*p
        :return: m*p matrix
        """
        m = log_A.shape[0]
        n = log_A.shape[1]
        p = log_B.shape[1]

        # log_A_expanded = torch.stack([log_A] * p, dim=2)
        # log_B_expanded = torch.stack([log_B] * m, dim=0)
        # fix for PyTorch > 1.5 by egaznep on Github:
        log_A_expanded = torch.reshape(log_A, (m, n, 1))
        log_B_expanded = torch.reshape(log_B, (1, n, p))

        elementwise_sum = log_A_expanded + log_B_expanded
        out = torch.logsumexp(elementwise_sum, dim=1)

        return out


class EmissionModel(torch.nn.Module):
    def __init__(self, N, M):
        super(EmissionModel, self).__init__()
        self.N = N
        self.M = M
        self.unnormalized_emission_matrix = torch.nn.Parameter(torch.randn(N, M))

    def forward(self, x_t):
        log_emission_matrix = torch.nn.functional.log_softmax(self.unnormalized_emission_matrix, dim=1)
        if x_t.shape[-1] == self.M:
            out = torch.mm(log_emission_matrix, x_t.transpose(0, 1)).transpose(0, 1)# matrix multiplication
        else:
            out = log_emission_matrix[:, x_t].transpose(0, 1)  # change 0 dim to 1 dim
        return out


class CharTokenizer:
    def __init__(self, alphabet):
        self.alphabet = alphabet

    def encode(self, s):
        """
        Convert a string into a list of integers
        """
        x = [self.alphabet.index(ss) for ss in s if ss in self.alphabet]
        return x

    def decode(self, x):
        """
        Convert list of ints to string
        """
        s = "".join([self.alphabet[xx] for xx in x])
        return s


class Trainer:
    def __init__(self, model, lr, tokenizer:CharTokenizer):
        self.model = model
        self.lr = lr
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=0.00001)
        self.tokenizer = tokenizer
    def train(self, dataset):
        train_loss = 0
        num_samples = 0
        self.model.train()
        print_interval = 500

        for idx, batch in enumerate(tqdm(dataset.loader)):
            x, T = batch
            batch_size = len(x)
            num_samples += batch_size
            log_probs = self.model(x, T)
            loss = -log_probs.mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            train_loss += loss.cpu().data.numpy().item() * batch_size
            if idx % print_interval == 0:
                print("loss:", loss.item())
                for _ in range(5):
                    sampled_x, sampled_z = self.model.sample()
                    print(self.tokenizer.decode(sampled_x))
                    print(sampled_z)
        train_loss /= num_samples
        return train_loss

    def test(self, dataset):
        test_loss = 0
        num_samples = 0
        self.model.eval()
        print_interval = 50
        for idx, batch in enumerate(dataset.loader):
            x, T = batch
            batch_size = len(x)
            num_samples += batch_size
            log_probs = self.model(x, T)
            loss = -log_probs.mean()
            test_loss += loss.cpu().data.numpy().item() * batch_size
            if idx % print_interval == 0:
                print("loss:", loss.item())
                sampled_x, sampled_z = self.model.sample()
                print(self.tokenizer.decode(sampled_x))
                print(sampled_z)
        test_loss /= num_samples
        return test_loss


def main_single():
    # Initialize the model
    tokenizer = CharTokenizer(string.ascii_lowercase)
    alphabet = tokenizer.alphabet

    model = HMM(M=len(alphabet), N=2)

    # Hard-wiring the parameters!
    # Let state 0 = consonant, state 1 = vowel
    for p in model.parameters():
        p.requires_grad = False  # needed to do lines below
    print("State priors:", torch.nn.functional.softmax(model.unnormalized_state_priors, dim=0))

    # In state 0, only allow consonants; in state 1, only allow vowels
    vowel_indices = torch.tensor([alphabet.index(letter) for letter in "aeiou"])
    consonant_indices = torch.tensor([alphabet.index(letter) for letter in "bcdfghjklmnpqrstvwxyz"])
    model.emission_model.unnormalized_emission_matrix[0, vowel_indices] = -100
    model.emission_model.unnormalized_emission_matrix[1, consonant_indices] = -100
    print("Emission matrix:", torch.nn.functional.softmax(model.emission_model.unnormalized_emission_matrix, dim=1))

    # Only allow vowel -> consonant and consonant -> vowel
    model.transition_model.unnormalized_transition_matrix[0, 0] = -100  # consonant -> consonant
    model.transition_model.unnormalized_transition_matrix[0, 1] = 0.  # vowel -> consonant
    model.transition_model.unnormalized_transition_matrix[1, 0] = 0.  # consonant -> vowel
    model.transition_model.unnormalized_transition_matrix[1, 1] = -100  # vowel -> vowel
    print("Transition matrix:",
          torch.nn.functional.softmax(model.transition_model.unnormalized_transition_matrix, dim=0))

    # start forward by model
    x = torch.stack([torch.tensor(tokenizer.encode("caa"))])
    x = torch.eye(model.M)[x]
    T = torch.tensor([3])
    print(model(x, T))

    x = torch.stack([torch.tensor(tokenizer.encode("aba")), torch.tensor(tokenizer.encode("abb"))])
    x = torch.eye(model.M)[x]
    T = torch.tensor([3, 3])
    print(model.forward(x, T))
    """
    tensor([[-10.8889],
        [    -inf]], device='cuda:0')
    When using the vowel <-> consonant HMM from above, notice that the forward algorithm returns -inf for x ="abb".
    That's because our transition matrix says the probability of vowel -> vowel and consonant -> consonant is 0, so the 
    probability of "abb" happening is 0, and thus the log probability is -inf.
    """

    x = torch.stack([torch.tensor(tokenizer.encode("aba")), torch.tensor(tokenizer.encode("abb"))])
    x = torch.eye(model.M)[x]
    T = torch.tensor([3, 3])
    print(model.viterbi(x, T))

def main_train():
    filename = "./datasets/training.txt"

    with open(filename, "r") as f:
        lines = f.readlines()  # each line of lines will have one word

    alphabet = list(Counter(("".join(lines))).keys())
    tokenizer = CharTokenizer(alphabet=alphabet)

    train_lines, valid_lines = train_test_split(lines, test_size=0.1, random_state=42)
    train_dataset = TextDataset(train_lines, tokenizer.encode)
    valid_dataset = TextDataset(valid_lines, tokenizer.encode)

    M = len(alphabet)

    # initialize model
    model = HMM(N=64, M=M)
    # Train model
    num_epochs = 10
    trainer = Trainer(model, lr=0.01, tokenizer=tokenizer)
    for epoch in range(num_epochs):
        print("========= Epoch %d of %d =========" % (epoch + 1, num_epochs))
        train_loss = trainer.train(train_dataset)
        valid_loss = trainer.test(valid_dataset)

        print("========= Results: epoch %d of %d =========" % (epoch + 1, num_epochs))
        print("train loss: %.2f| valid loss: %.2f\n" % (train_loss, valid_loss))


if __name__ == '__main__':
    # main_train()
    main_single()
