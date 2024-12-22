import numpy as np
import argparse
import os
import torch
import imp
import re
from mimic3models.keras_models.nbeat import NBeatsNet as NBeatsPytorch
# from nbeats_pytorch.model import NBeatsNet as NBeatsPytorch
# import nbeats_keras时有问题
# from nbeats_keras.model import NBeatsNet as NBeatsKeras
import matplotlib
import matplotlib.pyplot as plt

from mimic3benchmark.readers import InHospitalMortalityReader
from mimic3models.in_hospital_mortality import utils

from mimic3models.preprocessing import Discretizer, Normalizer
from mimic3models import metrics
from mimic3models import keras_utils
from mimic3models import common_utils

def subplots(outputs: dict, backend_name: str):
    layers = [a[0] for a in outputs.items()]
    values = [a[1] for a in outputs.items()]
    assert len(layers) == len(values)
    n = len(layers)
    fig, axes = plt.subplots(nrows=1, ncols=n, figsize=(15, 3))
    plt.subplots_adjust(left=0.1, bottom=0.2, right=0.9, top=0.75, wspace=0.4, hspace=0.4)
    for i in range(n):
        axes[i].plot(values[i], color='deepskyblue')
        axes[i].set_title(layers[i])
        axes[i].set_xlabel('t')
        axes[i].grid(axis='x')
        axes[i].grid(axis='y')
    fig.suptitle(f'{backend_name} - Outputs of the generic and interpretable configurations', fontweight='bold')
    plt.draw()


def plot(target, generic_predictions, interpretable_predictions, backend_name):
    plt.figure()
    plt.plot(target, color='gold', linewidth=2)
    # 这里预测是结果需要合并i和g，因为i在前g在后，他们是一个整体
    plt.plot(interpretable_predictions + generic_predictions, color='r', linewidth=2)
    plt.plot(interpretable_predictions, color='orange')
    plt.plot(generic_predictions, color='green')
    plt.grid()
    plt.legend(['ACTUAL', 'FORECAST-PRED', 'FORECAST-I', 'FORECAST-G'])
    plt.xlabel('t')
    plt.title(f'{backend_name} - Forecast - Actual vs Predicted')
    plt.draw()

from keras.callbacks import ModelCheckpoint, CSVLogger
# ArgumentParser: 创建一个命令行参数解析器
parser = argparse.ArgumentParser()
# 添加通用（默认）配置信息，因此有些没有在main或模型中明写
common_utils.add_common_arguments(parser)
parser.add_argument('--target_repl_coef', type=float, default=0.0)
parser.add_argument('--data', type=str, help='Path to the data of in-hospital mortality task',
                    default=os.path.join(os.path.dirname(__file__), '../../data/in-hospital-mortality/'))
parser.add_argument('--output_dir', type=str, help='Directory relative which all output files are stored',
                    default='.')
parser.add_argument('--task', type=str, help='select the specific task',
                    default='ihm')
args = parser.parse_args()
print(args)

if args.small_part:
    args.save_every = 2**30
# 默认是0
target_repl = (args.target_repl_coef > 0.0 and args.mode == 'train')

# Build readers, discretizers, normalizers
train_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'train'),
                                         # listfile存放的是每个样本中的17个生理指标情况
                                         listfile=os.path.join(args.data, 'train_listfile.csv'),
                                         #表示时间跨度是48h
                                         period_length=48.0)

val_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'train'),
                                       listfile=os.path.join(args.data, 'val_listfile.csv'),
                                       period_length=48.0)
# 下面创建了离散化器和标准化器，在read data环节中处理数据
# Discretizer: 用于将连续时间序列数据离散化为固定的时间步（例如小时或者分钟）。
# timestep：设置每个时间步的长度。
# store_masks=True：存储缺失值的掩码。
# impute_strategy='previous'：对于缺失值，使用上一时刻的值来填补。
# start_time='zero'：将起始时间设置为零。
discretizer = Discretizer(timestep=float(args.timestep),
                          store_masks=True,
                          impute_strategy='previous',
                          start_time='zero')

discretizer_header = discretizer.transform(train_reader.read_example(0)["X"])[1].split(',')
cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]
# Normalizer: 用于标准化数据，通常会对连续特征进行均值和方差标准化。
# fields=cont_channels：传入要标准化的特征列。
# 加载预先保存的标准化参数，文件路径由 normalizer_state 指定。
# 下午稍微看看Normalizer和Discretizer，看他把属性变成tensor
# 此外再看看处理完的数据为什么是3维的
normalizer = Normalizer(fields=cont_channels)  # choose here which columns to standardize
normalizer_state = args.normalizer_state
if normalizer_state is None:
    normalizer_state = 'ihm_ts{}.input_str-{}.start_time-zero.normalizer'.format(args.timestep, args.imputation)
    normalizer_state = os.path.join(os.path.dirname(__file__), normalizer_state)
normalizer.load_params(normalizer_state)
# 将args中的参数转成字典的格式，并添加必要值，送入network当中
args_dict = dict(args._get_kwargs())
args_dict['header'] = discretizer_header
args_dict['task'] = 'ihm'
args_dict['target_repl'] = target_repl

# Read data  同时离散化和标准化
# 里面存放的是train和test中的csv，可能处理掉了小部分
train_raw = utils.load_data(train_reader, discretizer, normalizer, args.small_part)
val_raw = utils.load_data(val_reader, discretizer, normalizer, args.small_part)

#因为这里预测的是生理指标，因此y不是01，而是后16h的数据情况，因此这里要重新划分数据集 physiological 表示

# Build the model
print("==> using model {}".format(args.network))

# model_module = imp.load_source(os.path.basename(args.network), args.network)
# model = model_module.Network(**args_dict)
# 直接搭建模型，不适用传参的方式
# 这里以住院死亡的数据为例，前32 as train 后16 as predict

SEASONALITY_BLOCK = 'seasonality'
TREND_BLOCK = 'trend'
GENERIC_BLOCK = 'generic'

sample_idx = 10
if args.task== 'ihm':
    backcast_length = 48 * 76
    forecast_length = 1
    sample_x = val_raw[0][sample_idx:sample_idx + 1]
    sample_y = val_raw[1][sample_idx]
    phy_type=18
    # tmp_x = torch.tensor((10000, 48*76))
    # tmp_y = torch.tensor((10000, 1))
elif args.task== 'phy':
    # train_raw = torch.rand(14000, 48, 76)
    # val_raw = torch.rand(32000, 48, 76)
    backcast_length = 32 * 76
    forecast_length = 16 * 76
    # train_phy = train_raw[0]
    # val_phy = val_raw[0]
    # 这里有点问题，数据处理完是48*76，但是y应该是bs，16，18
    train_phy = torch.rand(14000, 48, 76)
    val_phy = torch.rand(3200, 48, 18)
    split_timesteps = 32
    # 划分训练集和测试集
    phy_train_x = train_phy[:, :split_timesteps, :]
    phy_train_y = train_phy[:, split_timesteps:, :]
    phy_val_x = val_phy[:, :split_timesteps, :]
    phy_val_y = val_phy[:, split_timesteps:, :]
    sample_x = val_phy[sample_idx:sample_idx + 1]
    sample_y = val_phy[sample_idx]
    phy_type=18
model = NBeatsPytorch(
        backcast_length=backcast_length, forecast_length=forecast_length,
        stack_types=(GENERIC_BLOCK, TREND_BLOCK, SEASONALITY_BLOCK),
        nb_blocks_per_stack=2, thetas_dim=(4, 4, 4), hidden_layer_units=20,phy_type=phy_type,phy_forecast=16*76
    )
backend_name = NBeatsPytorch.name()
suffix = ".bs{}{}{}.ts{}{}".format(args.batch_size,
                                   ".L1{}".format(args.l1) if args.l1 > 0 else "",
                                   ".L2{}".format(args.l2) if args.l2 > 0 else "",
                                   args.timestep,
                                   ".trc{}".format(args.target_repl_coef) if args.target_repl_coef > 0 else "")
model.final_name = args.prefix + "nbeat" + suffix
print("==> model.final_name:", model.final_name)
# Compile the model
print("==> compiling the model")
model.compile(loss='mae', optimizer='adam')

# Load model weights
n_trained_chunks = 0
if args.load_state != "":
    model.load_weights(args.load_state)
    n_trained_chunks = int(re.match(".*epoch([0-9]+).*", args.load_state).group(1))


if args.mode == 'train':
    # 现在要调整nbeat中，模型的结构，修改其输出，使得全连接层能够接受
    if args.task =='phy':
        model.fit(phy_train_x, phy_train_y, validation_data=(phy_val_x, phy_val_y), epochs=1, batch_size=32, task='phy')
    elif args.task == 'ihm':
        model.fit(train_raw[0], train_raw[1], validation_data=(val_raw[0], val_raw[1]), epochs=1, batch_size=32, task='ihm')

    model.enable_intermediate_outputs()
    model.predict(sample_x)  # load intermediary outputs into our model object.
    # NOTE: g_pred + i_pred = pred.
    g_pred, i_pred, outputs = model.get_generic_and_interpretable_outputs()
    plot(target=sample_y, generic_predictions=g_pred, interpretable_predictions=i_pred, backend_name=backend_name)
    # 输出每个stack中，各个block的情况

    subplots(outputs, backend_name)
    plt.show()


elif args.mode == 'test':

    # ensure that the code uses test_reader
    del train_reader
    del val_reader
    del train_raw
    del val_raw

    test_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'test'),
                                            listfile=os.path.join(args.data, 'test_listfile.csv'),
                                            period_length=48.0)
    ret = utils.load_data(test_reader, discretizer, normalizer, args.small_part,
                          return_names=True)

    data = ret["data"][0]
    labels = ret["data"][1]
    names = ret["names"]

    predictions = model.predict(data, batch_size=args.batch_size, verbose=1)
    predictions = np.array(predictions)[:, 0]
    metrics.print_metrics_binary(labels, predictions)

    path = os.path.join(args.output_dir, "test_predictions", os.path.basename(args.load_state)) + ".csv"
    utils.save_results(names, predictions, labels, path)

else:
    raise ValueError("Wrong value for args.mode")

