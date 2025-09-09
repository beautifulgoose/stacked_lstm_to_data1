from dataset import build_kfold_datasets
from model import run_with_your_kfold

def run_cross_validation(n_splits: int = 5):
    # 按照用户标签划分五折交叉验证数据集
    """
    运行 n_splits 折交叉验证，逐折训练+验证，最后给出平均指标
    """
    # 一次性构建所有折的数据
    datasets, selected= build_kfold_datasets(n_splits=n_splits)

    run_with_your_kfold(datasets)
    print("hello")


if __name__ == "__main__":
    run_cross_validation(n_splits=5)

