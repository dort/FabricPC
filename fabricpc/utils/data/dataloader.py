import re
import numpy as np
from fabricpc.utils.data.data_utils import one_hot, split_np_seed


def _parse_mnist_split(split: str, train_n: int, test_n: int):
    """Parse a tfds-style split string for the numpy fallback.

    Supports 'train', 'test', and percentage slices like 'train[:80%]' or
    'train[80%:]'. Returns (base_split, start_idx, end_idx).
    """
    m = re.fullmatch(r"(train|test)(?:\[(\d*)%?:(\d*)%?\])?", split.replace(" ", ""))
    if m is None:
        raise ValueError(
            f"Unsupported split '{split}' for MNIST fallback loader "
            "(expected 'train', 'test', or percentage slice like 'train[:80%]')."
        )
    base, start_s, end_s = m.group(1), m.group(2), m.group(3)
    total = train_n if base == "train" else test_n
    # When no slice is provided, groups 2 and 3 are both None.
    if start_s is None and end_s is None:
        return base, 0, total
    start = int(round(int(start_s) / 100 * total)) if start_s else 0
    end = int(round(int(end_s) / 100 * total)) if end_s else total
    return base, start, end


class MnistLoader:
    """JAX-compatible MNIST data loader.

    Primary path uses TensorFlow Datasets for C++-parallel, GIL-free reads
    (avoids os.fork warnings with JAX). If TFDS is unavailable or fails
    (missing deps, protobuf/TFDS incompatibilities, no network), falls back
    to a numpy loader that downloads raw MNIST via the helpers in
    `fabricpc.continual.data`.

    Args:
        split: Dataset split to load. Use 'train' for training data or
               'test' for test data. Also supports slicing syntax like
               'train[:80%]' or 'train[80%:]' for custom splits.
        batch_size: Number of samples per batch.
        shuffle: Whether to shuffle the data each epoch.
        seed: Random seed for reproducibility. When set, ensures deterministic
              shuffling across runs and machines. If None, shuffling is random.
        normalize_mean: Mean for normalization (default: MNIST mean).
        normalize_std: Std for normalization (default: MNIST std).
    """

    def __init__(
        self,
        split: str,
        batch_size: int,
        shuffle: bool = True,
        seed: int = None,
        tensor_format: str = "NHWC",  # image tensor 'flat' or 'NHWC' batch-height-width-channels
        normalize_mean: float = 0.1307,
        normalize_std: float = 0.3081,
    ):
        self.split = split
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.tensor_format = tensor_format
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std
        self._use_tfds = False

        try:
            self._init_tfds(split)
            self._use_tfds = True
        except Exception as e:
            print(
                f"TFDS MNIST load failed ({type(e).__name__}: {e}); "
                "falling back to numpy loader."
            )
            self._init_numpy(split)

    def _init_tfds(self, split: str):
        import tensorflow_datasets as tfds
        import tensorflow as tf

        # Disable GPU for TensorFlow (we only use it for data loading)
        tf.config.set_visible_devices([], "GPU")

        # Split seed into two independent seeds for file and buffer shuffling
        file_seed, buffer_seed = split_np_seed(self.seed, n=2)

        # Configure read options for reproducibility
        read_config = tfds.ReadConfig(
            shuffle_seed=file_seed,
            interleave_cycle_length=1,  # Sequential reading for determinism
        )

        # Load dataset with pinned version for cross-machine reproducibility
        ds, info = tfds.load(
            "mnist:3.0.1",
            split=split,
            with_info=True,
            as_supervised=True,
            read_config=read_config,
            shuffle_files=self.shuffle and self.seed is not None,
        )
        self.num_examples = info.splits[split].num_examples
        self._num_batches = (self.num_examples + self.batch_size - 1) // self.batch_size

        # Build pipeline
        if self.shuffle:
            ds = ds.shuffle(
                buffer_size=self.num_examples, seed=buffer_seed
            )  # mnist fits in memory (~60MB) so the buffer is the full dataset
        ds = ds.batch(self.batch_size, drop_remainder=False)
        ds = ds.prefetch(tf.data.AUTOTUNE)

        self.ds = ds

    def _init_numpy(self, split: str):
        # Reuse the robust Keras/manual download path already used by the
        # continual-learning Split-MNIST loader.
        from fabricpc.continual.data import _load_mnist_keras, _load_mnist_manual

        result = _load_mnist_keras()
        if result is None:
            result = _load_mnist_manual("./data")
        train_images, train_labels, test_images, test_labels = result

        base, start, end = _parse_mnist_split(
            split, train_n=len(train_images), test_n=len(test_images)
        )
        if base == "train":
            images, labels = train_images[start:end], train_labels[start:end]
        else:
            images, labels = test_images[start:end], test_labels[start:end]

        # Match TFDS shape: (N, 28, 28, 1) uint8-equivalent float in [0, 255]
        # before the per-batch /255 normalisation done in __iter__.
        self._np_images = images[..., np.newaxis].astype(np.float32)
        self._np_labels = labels.astype(np.int64)
        self.num_examples = len(self._np_images)
        self._num_batches = (self.num_examples + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        if self._use_tfds:
            yield from self._iter_tfds()
        else:
            yield from self._iter_numpy()

    def _iter_tfds(self):
        for images, labels in self.ds:
            # Convert to numpy, normalize, and flatten
            images = images.numpy().astype(np.float32) / 255.0
            images = (images - self.normalize_mean) / self.normalize_std

            # images shape is (Batch, 28, 28, 1)
            if self.tensor_format == "flat":
                images = images.reshape(images.shape[0], -1)  # Flatten to (Batch, 784)

            # One-hot encode labels
            labels = one_hot(labels.numpy(), num_classes=10)

            yield images, labels

    def _iter_numpy(self):
        indices = np.arange(self.num_examples)
        if self.shuffle:
            # Derive a fresh epoch seed the same way the TFDS path's buffer
            # shuffle does, so that seeded runs are reproducible across epochs.
            epoch_seed = None
            if self.seed is not None:
                epoch_seed = int(self.seed) + getattr(self, "_epoch", 0)
            rng = np.random.default_rng(epoch_seed)
            rng.shuffle(indices)
            self._epoch = getattr(self, "_epoch", 0) + 1

        for start in range(0, self.num_examples, self.batch_size):
            batch_idx = indices[start : start + self.batch_size]
            images = self._np_images[batch_idx] / 255.0
            images = (images - self.normalize_mean) / self.normalize_std
            if self.tensor_format == "flat":
                images = images.reshape(images.shape[0], -1)
            labels = one_hot(self._np_labels[batch_idx], num_classes=10)
            yield images, labels

    def __len__(self):
        return self._num_batches


class CharDataLoader:
    """JAX-compatible character-level dataloader using TFDS.

    Loads the tiny_shakespeare dataset from TensorFlow Datasets and
    yields batches of (x_indices, y_onehot) for next-character prediction.

    The vocabulary is always built from the train split to ensure consistent
    char-to-index mappings across all splits.

    Args:
        split: Dataset split ('train', 'validation', or 'test').
        seq_len: Number of characters per input sequence.
        batch_size: Number of sequences per batch.
        shuffle: Whether to shuffle sequence start positions each epoch.
        seed: Random seed for reproducible shuffling.
        max_samples: If set, cap the number of sequences to this value.
            Useful for fast hyperparameter tuning on a subset of data.
    """

    # Class-level cache for vocabulary (built once from train split)
    _vocab = None

    def __init__(
        self,
        split: str,
        seq_len: int,
        batch_size: int,
        shuffle: bool = True,
        seed: int = None,
        max_samples: int = None,
    ):
        import tensorflow_datasets as tfds
        import tensorflow as tf

        tf.config.set_visible_devices([], "GPU")

        self.seq_len = seq_len
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self._epoch = 0

        # Build vocabulary from the train split (cached across instances)
        if CharDataLoader._vocab is None:
            train_ds = tfds.load("tiny_shakespeare", split="train")
            train_text = next(iter(train_ds))["text"].numpy().decode("utf-8")
            chars = sorted(set(train_text))
            CharDataLoader._vocab = {
                "chars": chars,
                "vocab_size": len(chars),
                "char_to_idx": {ch: i for i, ch in enumerate(chars)},
                "idx_to_char": {i: ch for i, ch in enumerate(chars)},
            }

        self.chars = CharDataLoader._vocab["chars"]
        self.vocab_size = CharDataLoader._vocab["vocab_size"]
        self.char_to_idx = CharDataLoader._vocab["char_to_idx"]
        self.idx_to_char = CharDataLoader._vocab["idx_to_char"]

        # Load the requested split and encode to indices
        ds = tfds.load("tiny_shakespeare", split=split)
        text = next(iter(ds))["text"].numpy().decode("utf-8")
        self.data = np.array([self.char_to_idx[ch] for ch in text], dtype=np.int32)

        # Each sequence needs seq_len input chars + 1 target char
        self.num_sequences = len(self.data) - seq_len
        if max_samples is not None:
            self.num_sequences = min(self.num_sequences, max_samples)
        self._num_batches = self.num_sequences // batch_size

    def __iter__(self):
        indices = np.arange(self.num_sequences)
        if self.shuffle:
            epoch_seed = self.seed + self._epoch if self.seed is not None else None
            rng = np.random.default_rng(epoch_seed)
            rng.shuffle(indices)
        self._epoch += 1

        for start in range(0, len(indices), self.batch_size):
            batch_idx = indices[start : start + self.batch_size]
            if len(batch_idx) < self.batch_size:
                continue  # drop incomplete last batch

            x = np.stack(
                [self.data[i : i + self.seq_len] for i in batch_idx]
            )  # (batch, seq_len) int32
            y_idx = np.stack(
                [self.data[i + 1 : i + self.seq_len + 1] for i in batch_idx]
            )  # (batch, seq_len)
            y_onehot = np.eye(self.vocab_size, dtype=np.float32)[y_idx]

            yield x, y_onehot

    def __len__(self):
        return self._num_batches

    def decode(self, indices) -> str:
        """Convert an array of character indices back to a string."""
        return "".join(self.idx_to_char[int(i)] for i in indices)
