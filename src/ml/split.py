def split_shots(filenames, val_ratio=0.2):

    if len(filenames) == 1:
        return filenames, []  # всё в train

    n_val = max(1, int(len(filenames) * val_ratio))

    val_files = filenames[:n_val]
    train_files = filenames[n_val:]

    if len(train_files) == 0:
        train_files = val_files
        val_files = []

    return train_files, val_files