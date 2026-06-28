from fastai.text.all import *  # noqa: F403

path = untar_data(URLs.HUMAN_NUMBERS)  # noqa: F405

print(path.ls())

lines = L()  # noqa: F405
with open(path / "train.txt") as f:
    lines += L(*f.readlines())  # noqa: F405
with open(path / "valid.txt") as f:
    lines += L(*f.readlines())  # noqa: F405

print(type(lines))

print(lines[:10])

text = "  .  ".join([l.strip() for l in lines])  # noqa: E741
print(text[:100])

# Токенизация
tokens = text.split("  ")
print(tokens[:10])

# Словарь
vocab = L(*tokens).unique()  # noqa: F405
print(vocab)

word2ind = {w: i for i, w in enumerate(vocab)}
print(word2ind)
nums = L(word2ind[i] for i in tokens)  # noqa: F405
print(nums[:20])

seq = L((tokens[i : i + 3], tokens[i + 3]) for i in range(0, len(tokens) - 4, 3))  # noqa: F405
print(seq[:20])

seq = L(  # noqa: F405
    (tensor(nums[i : i + 3]), tensor(nums[i + 3]))  # noqa: F405
    for i in range(0, len(tokens) - 4, 3)  # noqa: F405
)
print(seq[:20])

bs = 64
cut = int(len(seq) * 0.8)
dls = DataLoaders.from_dsets(seq[:cut], seq[cut:], bs=bs, shuffle=False)  # noqa: F405


class Model1(Module):  # noqa: F405
    def __init__(self, vocab_size, bs):
        self.i_h = nn.Embedding(vocab_size, bs)  # noqa: F405
        self.h_h = nn.Linear(bs, bs)  # noqa: F405
        self.h_o = nn.Linear(bs, vocab_size)  # noqa: F405

    def forward(self, x):
        # h = F.relu(self.h_h(self.i_h(x[:, 0])))
        # h = h + self.i_h(x[:, 1])
        # h = F.relu(self.h_h(h))
        # h = h + self.i_h(x[:, 2])
        # h = F.relu(self.h_h)
        h = 0
        for i in range(3):
            h = h + self.i_h(x[:, i])
            h = F.relu(self.h_h(h))  # noqa: F405
        return self.h_o(h)


if __name__ == "__main__":
    learn = Learner(  # noqa: F405
        dls,
        Model1(len(vocab), bs),
        loss_func=F.cross_entropy,  # noqa: F405
        metrics=accuracy,  # noqa: F405
    )

    learn.fit_one_cycle(3)

    # thousand - 0.15
