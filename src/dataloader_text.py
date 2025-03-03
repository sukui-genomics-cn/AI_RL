import torch.utils.data

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, lines, encode):
        self.lines = lines  # list of strings
        collate = Collate(encode)  # function for generating a minibatch from strings
        self.loader = torch.utils.data.DataLoader(self, batch_size=1024, num_workers=1, shuffle=True,
                                                  collate_fn=collate)

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx].lstrip(" ").rstrip("\n").rstrip(" ").rstrip("\n")
        return line


class Collate:
    def __init__(self, encode):
        self.encode = encode

    def __call__(self, batch):
        """
        Returns a minibatch of strings, padded to have the same length.
        """
        x = []
        batch_size = len(batch)
        for index in range(batch_size):
            x_ = batch[index]

            # convert letters to integers
            x.append(self.encode(x_))

        # pad all sequences with 0 to have same length
        x_lengths = [len(x_) for x_ in x]
        T = max(x_lengths)
        for index in range(batch_size):
            x[index] += [0] * (T - len(x[index]))
            x[index] = torch.tensor(x[index])

        # stack into single tensor
        x = torch.stack(x)
        x_lengths = torch.tensor(x_lengths)
        return (x, x_lengths)
