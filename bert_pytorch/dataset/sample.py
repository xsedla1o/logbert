from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split


def generate_pairs(line, window_size):
    line = np.array(line)
    line = line[:, 0]

    seqs = []
    for i in range(0, len(line), window_size):
        seq = line[i : i + window_size]
        seqs.append(seq)
    seqs += []
    seq_pairs = []
    for i in range(1, len(seqs)):
        seq_pairs.append([seqs[i - 1], seqs[i]])
    return seqs


def fixed_window_data(data, window_size, adaptive_window, seq_len=None, min_len=0):
    """
    Process a session represented as a list of tokens.

    Each token is expected to be either:
      - A single element (log key) if no time is available, or
      - A two-element list/tuple [log_key, timestamp] if a time duration is provided.

    :param data: List of tokens.
    :param window_size: Fixed window size for segmenting the tokens.
    :param adaptive_window: If True, use the full session length as the window size.
    :param seq_len: Optional maximum number of tokens to consider.
    :param min_len: Minimum required length for a session.
    :return: Tuple (logkey_seqs, time_seq) where each is a list of windows.
    """
    if len(data) < min_len:
        return [], [], 0

    if seq_len is not None:
        data = data[:seq_len]

    if adaptive_window:
        window_size = len(data)

    # Convert to a NumPy array for easy slicing.
    data_arr = np.array(data)

    # If each token has two elements, assume the second is a timestamp.
    if data_arr.ndim > 1 and data_arr.shape[1] == 2:
        # Extract time stamps and convert to float.
        tim = data_arr[:, 1].astype(float)
        log_keys = data_arr[:, 0]
        # Ensure the first time stamp is 0.
        tim[0] = 0
    else:
        # When no time information is available, create a zero time array.
        log_keys = data_arr.squeeze()
        tim = np.zeros(log_keys.shape, dtype=float)

    logkey_seqs = []
    time_seq = []
    splits = 0
    for e, i in enumerate(range(0, len(log_keys), window_size)):
        logkey_seqs.append(log_keys[i : i + window_size])
        time_seq.append(tim[i : i + window_size])
        splits = e + 1

    return logkey_seqs, time_seq, splits


def fixed_window(line, window_size, adaptive_window, seq_len=None, min_len=0):
    line = [ln.split(",") for ln in line.split()]

    # filter the line/session shorter than 10
    if len(line) < min_len:
        return [], []

    # max seq len
    if seq_len is not None:
        line = line[:seq_len]

    if adaptive_window:
        window_size = len(line)

    line = np.array(line)

    # if time duration exists in data
    if line.shape[1] == 2:
        tim = line[:, 1].astype(float)
        line = line[:, 0]

        # the first time duration of a session should be 0, so max is window_size(mins) * 60
        tim[0] = 0
    else:
        line = line.squeeze()
        # if time duration doesn't exist, then create a zero array for time
        tim = np.zeros(line.shape)

    logkey_seqs = []
    time_seq = []
    for i in range(0, len(line), window_size):
        logkey_seqs.append(line[i : i + window_size])
        time_seq.append(tim[i : i + window_size])

    return logkey_seqs, time_seq


def generate_train_valid(
    data_path,
    window_size=20,
    adaptive_window=True,
    sample_ratio=1,
    valid_size=0.1,
    output_path=None,
    scale=None,
    scale_path=None,
    seq_len=None,
    min_len=0,
):
    with open(data_path, "r") as f:
        data_iter = f.readlines()

    num_session = int(len(data_iter) * sample_ratio)
    # only even number of samples, or drop_last=True in DataLoader API
    # coz in parallel computing in CUDA, odd number of samples reports issue when merging the result
    # num_session += num_session % 2

    test_size = int(min(num_session, len(data_iter)) * valid_size)
    # only even number of samples
    # test_size += test_size % 2

    print("before filtering short session")
    print("train size ", int(num_session - test_size))
    print("valid size ", int(test_size))
    print("=" * 40)

    logkey_seq_pairs = []
    time_seq_pairs = []
    session = 0
    for line in tqdm(data_iter, desc="Generating train/valid pairs"):
        if session >= num_session:
            break
        session += 1

        logkeys, times = fixed_window(
            line, window_size, adaptive_window, seq_len, min_len
        )
        logkey_seq_pairs += logkeys
        time_seq_pairs += times

    logkey_seq_pairs = np.array(logkey_seq_pairs, dtype=object)
    time_seq_pairs = np.array(time_seq_pairs, dtype=object)

    logkey_trainset, logkey_validset, time_trainset, time_validset = train_test_split(
        logkey_seq_pairs, time_seq_pairs, test_size=test_size, random_state=1234
    )

    # sort seq_pairs by seq len
    train_len = list(map(len, logkey_trainset))
    valid_len = list(map(len, logkey_validset))

    train_sort_index = np.argsort(-1 * np.array(train_len))
    valid_sort_index = np.argsort(-1 * np.array(valid_len))

    logkey_trainset = logkey_trainset[train_sort_index]
    logkey_validset = logkey_validset[valid_sort_index]

    time_trainset = time_trainset[train_sort_index]
    time_validset = time_validset[valid_sort_index]

    print("=" * 40)
    print("Num of train seqs", len(logkey_trainset))
    print("Num of valid seqs", len(logkey_validset))
    print("=" * 40)

    return logkey_trainset, logkey_validset, time_trainset, time_validset


def process_data_file(
    file_path, window_size=20, adaptive_window=True, seq_len=None, min_len=0
):
    # Read the data from the file
    with open(file_path, "r") as f:
        data_iter = f.readlines()

    logkey_seq_pairs = []
    time_seq_pairs = []
    for line in tqdm(data_iter, desc=f"Processing {file_path}"):
        logkeys, times = fixed_window(
            line, window_size, adaptive_window, seq_len, min_len
        )
        logkey_seq_pairs += logkeys
        time_seq_pairs += times

    # Convert to NumPy arrays
    logkey_seq_pairs = np.array(logkey_seq_pairs, dtype=object)
    time_seq_pairs = np.array(time_seq_pairs, dtype=object)

    # Sort sequences by length (longest first)
    lengths = list(map(len, logkey_seq_pairs))
    sort_index = np.argsort(-np.array(lengths))
    logkey_seq_pairs = logkey_seq_pairs[sort_index]
    time_seq_pairs = time_seq_pairs[sort_index]

    return logkey_seq_pairs, time_seq_pairs


def generate_train_valid_from_files(
    train_data_path,
    valid_data_path,
    window_size=20,
    adaptive_window=True,
    seq_len=None,
    min_len=0,
):
    # Process training data
    logkey_trainset, time_trainset = process_data_file(
        train_data_path, window_size, adaptive_window, seq_len, min_len
    )

    # Process validation data
    logkey_validset, time_validset = process_data_file(
        valid_data_path, window_size, adaptive_window, seq_len, min_len
    )

    print("=" * 40)
    print("Num of train seqs:", len(logkey_trainset))
    print("Num of valid seqs:", len(logkey_validset))
    print("=" * 40)

    return logkey_trainset, logkey_validset, time_trainset, time_validset
