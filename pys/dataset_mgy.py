import gzip
from torch.utils.data import IterableDataset


class StreamingFastaDataset(IterableDataset):

    def __init__(self, fasta_path, max_len=70):

        self.fasta_path = fasta_path
        self.max_len = max_len

    def parse_fasta(self):

        if self.fasta_path.endswith(".gz"):
            f = gzip.open(self.fasta_path, "rt")
        else:
            f = open(self.fasta_path)

        pid = None
        seq = []

        for line in f:

            if line.startswith(">"):

                if pid:
                    yield pid, "".join(seq)[:self.max_len]

                pid = line.split()[0][1:]
                seq = []

            else:
                seq.append(line.strip())

        if pid:
            yield pid, "".join(seq)[:self.max_len]

        f.close()

    def __iter__(self):

        for pid, seq in self.parse_fasta():

            yield pid, seq