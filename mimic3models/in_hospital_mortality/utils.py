from mimic3models import common_utils
import numpy as np
import os


def load_data(reader, discretizer, normalizer, small_part=False, return_names=False):
    N = reader.get_number_of_examples()
    if small_part:
        N = 1000
    ret = common_utils.read_chunk(reader, N)
    # ret[14681,ts,18],因为希望是48h的数据，每小时一个，但是有的可能多于48，因此这里统一化
    data = ret["X"]
    ts = ret["t"]
    labels = ret["y"]
    names = ret["name"]
    # 离散化后，一方面ts变成了统一的48，另一方面把原来18个生理指标变成了76来描述
    # 此外，还涉及到一些填充策略，当数据缺失时或者间隔过长时

    data = [discretizer.transform(X, end=t)[0] for (X, t) in zip(data, ts)]
    if normalizer is not None:
        data = [normalizer.transform(X) for X in data]
    whole_data = (np.array(data), labels)
    if not return_names:
        return whole_data
    return {"data": whole_data, "names": names}


def save_results(names, pred, y_true, path):
    common_utils.create_directory(os.path.dirname(path))
    with open(path, 'w') as f:
        f.write("stay,prediction,y_true\n")
        for (name, x, y) in zip(names, pred, y_true):
            f.write("{},{:.6f},{}\n".format(name, x, y))
