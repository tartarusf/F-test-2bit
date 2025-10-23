# 本程序用于在F泄露分析结果中统计出
# 单个子密钥的特征点
import re
import h5py
import numpy as np
import shutil
import os
# 1、找到文件中没有泄露的功耗数量
def count_relevant_bits(filepath): # 查找的函数
    """
    Counts the number of occurrences of "Relevant bits:" in a text file.

    Args:
        filepath: The path to the text file.

    Returns:
        The number of times "Relevant bits:" appears in the file,
        or -1 if the file is not found.
    """
    try:
        with open(filepath, 'r') as f:
            count = 0
            for line in f:
                if "Relevant bits:" in line:
                    count += 1
            return count
    except FileNotFoundError:
        print(f"Error: File not found at path: {filepath}")
        return -1
    except Exception as e:
        print(f"An error occurred: {e}")
        return -1

# 2、（单密钥）找到泄露相关项及特征点
def analyze_bit_samples_with_features(filepath,key_start,key_end):
    """
    分析文本文件，并返回每个子密钥关联的唯一特征点及其数量。

    Args:
        filepath: 文本文件的路径。

    Returns:
        一个字典，其中：
            - 键: 子密钥 (e.g., "bit 0")
            - 值: 一个字典，包含：
                - 'features': 一个列表，包含与该子密钥关联的唯一特征点 (Sample 编号)
                - 'count': 与该子密钥关联的唯一特征点的数量

        如果文件未找到或发生错误，则返回 None。
    """
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"错误：未找到文件路径：{filepath}")
        return None
    except Exception as e:
        print(f"发生错误：{e}")
        return None

    bit_sample_counts = {}
    # for i in range(16):
    for i in range(key_start,key_end):
        bit_sample_counts[f"bit {i}"] = set()  # 使用集合存储唯一的样本编号

    samples = {}
    for t, line in enumerate(lines):
        sample_match = re.search(r"Sample (\d+)", line)
        if sample_match:
            samples[t] = sample_match.group(1)

    # for i in range(16):
    for i in range(key_start, key_start + 16):
        for t, line in enumerate(lines):
            bit_string = f"bit {i}"  
            txt = line.strip()
            if t > 0 and lines[t-2].strip() == "Relevant bits:":
                    continue  # 跳过标题行之后的行
            if txt == bit_string:  # 使用 == 比较去除空格后的行内容和 bit_string，确保完全匹配
            # if bit_string in line:
                for k in range(t - 1, -1, -1):
                    if k in samples:
                        bit_sample_counts[bit_string].add(samples[k])
                        break
        print(f"bit{i}处理完成")

    # 整理结果，包含特征点列表和数量
    # result_summary = {}
    # for bit, unique_samples in bit_sample_counts.items():
    #     result_summary[bit] = {
    #         'features': sorted(list(unique_samples)),  # 将集合转换为列表并排序
    #         'count': len(unique_samples)
    #     }
    result_summary = {}
    for bit, unique_samples in bit_sample_counts.items():
        result_summary[bit] = {
            'features': sorted(list(unique_samples), key=lambda x: int(x)),  # 按照数值大小排序
            'count': len(unique_samples)
        }

    return result_summary

# 3、（单密钥）保存特征点
def add_list_to_mat_new_file(original_file_path, new_file_path, list_data, dataset_name):
    """
    将 Python 列表添加到新的 .mat 文件中，存储为一个新的矩阵，原文件不进行改动。
    参数:
        original_file_path (str): 原 .mat 文件的路径。
        new_file_path (str): 新 .mat 文件的路径。
        list_data (list): 要存储的特征点 (Python 列表)。
        dataset_name (str): 在新 .mat 文件中创建的矩阵的名称。
    """
    # 没有新文件则复制原文件到新文件
    if not os.path.exists(new_file_path):
        # 如果新文件不存在，复制原文件到新文件
        shutil.copyfile(original_file_path, new_file_path)
        print(f"新文件 {new_file_path} 已创建。")

    # 打开新的 .mat 文件（以追加模式）
    with h5py.File(new_file_path, 'a') as f:  # 'a' 模式表示追加
        # 将列表转换为 NumPy 数组
        data_array = np.array(list_data)

        # 检查是否已存在同名数据集
        if dataset_name in f:
            print(f"警告：'{dataset_name}' 已存在于新文件中，将被覆盖。")
            del f[dataset_name]  # 删除已存在的数据集

        # 创建一个新的矩阵并存储列表数据
        f.create_dataset(dataset_name, data=data_array)

    print(f"列表数据已成功添加到新文件 {new_file_path} 中的 '{dataset_name}' 数据集。")


# main:
filepath = 'E:\PZH\code\F-Test-Analysis\Ftest_key_6802点_32bitpv10_后半.txt' # 输入文本
filewrite = 'E:\PZH\code\F-Test-Analysis\子密钥泄露情况_32bitpv10_后半.txt' # 输出文本
fo = open(filewrite, "w") # 覆写文本

# 设置子密钥开始和结束位置（记得修改）
key_start = 0 # 密钥开始的地方，一定记得修改！！！！！
key_end = 64 # 密钥结束的地方，一定记得修改！！！！！

# 计算泄露功耗点数
occurrences = count_relevant_bits(filepath)
if occurrences != -1:
    print(f"有泄露的功耗点数为 {occurrences} 个")
    fo.write(f"有泄露的功耗点数为 {occurrences} 个\r\n") # \r\n是win的换行方式
# 统计子密钥相关泄露功耗点
results = analyze_bit_samples_with_features(filepath,key_start,key_end)
if results:
    for bit, data in results.items():
        print(f"子密钥: {bit}")
        fo.write(f"******子密钥: {bit}******\r\n")
        print(f"  特征点: {', '.join(data['features'])}")  # 打印特征点列表
        fo.write(f"  特征点: {', '.join(data['features'])}\r\n")
        print(f"  特征点数量: {data['count']}\n")
        fo.write(f"  特征点数量: {data['count']}\r\n")

        features = data['features']

        try:
            data_list = [int(s) for s in features]  # 使用列表推导式进行转换
        except ValueError:
            print("Error: 列表中包含无法转换为整数的字符串")
            exit()

        # 文件名
        original_file_path = 'E:\PZH\code\F-Test-Analysis\mat\RS_8K_6802_2025_01_31_13h00m_0.99717_byte32.mat'  # .mat 文件路径
        new_file_path = 'E:\PZH\code\F-Test-Analysis\mat\RS_8K_6802_2025_01_31_13h00m_0.99717_byte32_Ftest.mat'  # .mat 文件路径
        dataset_name = bit.replace(" ", "_")  # 新矩阵的名称

        # 调用函数
        add_list_to_mat_new_file(original_file_path, new_file_path, data_list, dataset_name)





