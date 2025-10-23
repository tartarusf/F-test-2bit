# 本程序用于在F泄露分析结果中统计出
# 联合2个子密钥的特征点
import re
import shutil
import os
import numpy as np
import scipy.io as sio
import h5py
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

# 2、（多密钥）找到泄露相关项及特征点
def analyze_bit_samples_with_features(filepath,key_start,key_end,threshold):
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

    # bit_sample_counts = {}
    # # for i in range(16):
    # for i in range(key_start,key_end):
    #     for j in range(i+1,key_end):
    #         bit_sample_counts[f"bit {i}\tbit {j}"] = set()  # 使用集合存储唯一的样本编号

    # 修改初始化部分(统计pv值）25.2.26
    bit_sample_counts = {}
    for i in range(key_start, key_end):
        for j in range(i + 1, key_end):
            bit_pair = f"bit {i}\tbit {j}"
            bit_sample_counts[bit_pair] = {
                'samples': set(),  # 原有字段，存储Sample编号
                'total_pv': 0,  # 新增：总pv值数量
                'valid_pv': 0,  # 新增：有效pv值数量（非inf）
                'sum_valid_pv': 0.0,
                'valid_pv_list': []
            }

    samples = {}
    for t, line in enumerate(lines):
        sample_match = re.search(r"Sample (\d+)", line)
        if sample_match:
            samples[t] = sample_match.group(1)

    # for i in range(16):
    for i in range(key_start, key_end):
        for j in range(i + 1, key_end):
            for t, line in enumerate(lines): # （统计pv值）25.2.26修改
                bit_string = f"bit {i}\tbit {j}"
                txt = line.strip()

                # 新增：跳过标题行逻辑（保持原有代码）
                if t > 0 and lines[t - 2].strip() == "Relevant bits:":
                    continue

                if txt == bit_string:
                    if t >= 2:  # 2025.04.02
                        # 检查前两行是否包含 "pv="
                        pv_line = lines[t - 2].strip()
                        if "pv=" in pv_line:
                            # 提取pv值（例如 "pv=56.49624076622083" 或 "pv=inf"）
                            pv_match = re.search(r"pv=([^,]+)", pv_line)
                            pv_value = pv_match.group(1)
                            if pv_value=='inf' or float(pv_value)>threshold: # 如果大于设置的阈值pv，才加入进去
                                # 找到Sample行
                                sample_line_index = None
                                # 逆向查找Sample
                                for k in range(t - 1, -1, -1):
                                    if k in samples:
                                        sample_line_index = k
                                        bit_sample_counts[bit_string]['samples'].add(samples[k])
                                        break


                                # 查找点的pv值
                                # if sample_line_index is not None:
                                #     # 确保Sample行后有两行
                                #     if sample_line_index + 2 < len(lines):
                                #         pv_line = lines[sample_line_index + 2].strip()
                                #         # 检查是否包含 "当前功耗点的pv为："  (或者更通用的 "pv为：")
                                #         if "当前功耗点的pv为：" in pv_line:
                                #             # 提取pv值
                                #             pv_match = re.search(r"当前功耗点的pv为：([^,]+)", pv_line)  # 修改了正则表达式
                                #             if pv_match:
                                #                 pv_value = pv_match.group(1)
                                #         elif "pv为：" in pv_line:  # 更通用的匹配
                                #             pv_match = re.search(r"pv为：([^,]+)", pv_line)
                                #             if pv_match:
                                #                 pv_value = pv_match.group(1)

                    # 统计pv值
                    if pv_value is not None and float(pv_value) >= threshold:
                        bit_sample_counts[bit_string]['total_pv'] += 1
                        if pv_value.lower() != 'inf':  # 检查是否为非inf
                            bit_sample_counts[bit_string]['valid_pv'] += 1
                            try:
                                pv_float = float(pv_value)
                                bit_sample_counts[bit_string]['sum_valid_pv'] += pv_float
                                bit_sample_counts[bit_string]['valid_pv_list'].append(pv_float)

                            except ValueError:
                                pass

            print(f"bit {i}\tbit {j}统计完成")

    # 整理结果，包含特征点列表和数量
    result_summary = {}
    for bit, data in bit_sample_counts.items():
        valid_pv_list = data['valid_pv_list']
        # 计算平均值
        sum_pv = data['sum_valid_pv']
        avg_pv = sum_pv / len(valid_pv_list) if len(valid_pv_list) > 0 else 0.0
        pv_ratio = data['valid_pv'] / data['total_pv'] if data['total_pv'] > 0 else 0.0
        # 计算中位数
        if len(valid_pv_list) > 0:
            sorted_pv = sorted(valid_pv_list)
            median_pv = sorted_pv[len(sorted_pv) // 2]
        else:
            median_pv = 0.0

        result_summary[bit] = {
            'features': sorted(list(data['samples']), key=lambda x: int(x)),
            'count': len(data['samples']),
            # 新增：计算有效pv比例
            'pv_ratio': pv_ratio,
            'avg_pv': avg_pv,
            'median_pv': median_pv,
            'avg_pv_ratio': avg_pv * pv_ratio,
            'sum_pv': sum_pv
        }

    return result_summary

# 3、（多密钥）保存特征点
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

# 4 (多密钥)保存标签（密钥组）
# 定义 Ftest_label2 函数`
# def Ftest_label2(data_matrix, sub_key_indices, operation):
#     """
#     生成标签矩阵
#
#     输入参数：
#         data_matrix:  原始数据矩阵 (16xN)
#         sub_key_indices: 子密钥索引向量，例如 [1, 2] 表示使用第一个和第二个子密钥
#         operation:    运算类型，例如 'bitand' 表示与运算，'bitor' 表示或运算，'bitxor' 表示异或运算，
#                       'concat' 表示二进制拼接
#
#     输出参数：
#         label_matrix: 标签矩阵 (1xN)
#     """
#     N = data_matrix.shape[1]  # 获取列数
#     num_sub_keys = len(sub_key_indices)
#     label_matrix = np.zeros(N, dtype=int)  # 初始化标签矩阵
#
#     for i in range(N):
#         # 提取子密钥（注意：MATLAB索引从1开始，Python索引从0开始）
#         sub_keys = data_matrix[np.array(sub_key_indices) - 1, i]
#
#         # 执行运算
#         if num_sub_keys > 0:
#             if operation == 'concat':
#                 # 二进制拼接
#                 binary_str = ''
#                 for j in range(num_sub_keys):
#                     binary_str += format(sub_keys[j], '04b')  # 将每个子密钥转换为4位二进制并拼接
#                 label_matrix[i] = int(binary_str, 2)  # 将拼接后的二进制字符串转换为十进制
#             else:
#                 # 位运算（bitand、bitor、bitxor）
#                 result = sub_keys[0]
#                 for j in range(1, num_sub_keys):
#                     if operation == 'bitand':
#                         result = result & sub_keys[j]
#                     elif operation == 'bitor':
#                         result = result | sub_keys[j]
#                     elif operation == 'bitxor':
#                         result = result ^ sub_keys[j]
#                 label_matrix[i] = result
#         else:
#             label_matrix[i] = 0  # 或者其他默认值
#
#     return label_matrix

# main:

# 5、set转换成int列表
def set_to_list(int_list):
    try:
        int_list = [int(item) for item in point_set]  # 使用列表推导式将字符串转换为整数
        int_list.sort(key = int)
        return int_list

    except ValueError as e:
        print(f"错误: 集合中的元素无法转换为整数。{e}")

# 6、裁剪后的数据根据leak重新映射原有的点（已经转换int列表）
def setleak_to_list(point_set,mat_file,leak_name):
    point_list = []
    try:
        # 使用 h5py 打开 .mat 文件
        with h5py.File(mat_file, 'r') as f:
            # 获取 'leak' 数组
            if leak_name not in f:
                raise KeyError(f"文件 '{mat_file}' 中未找到数据集{leak_name}。")
            # leak 是 HDF5 数据集对象，需要读取
            leak_f = f[leak_name][:]
            leak = np.sort(leak_f, axis=0)
            # leak = leak_f[sorted_indices]
            # leak = f['random_leak'][:]
            # print(f"'leak' 数组的形状: {leak.shape}")
    except Exception as e:
        print(f"加载文件失败: {e}")
        raise
    try:
        # 创建一个新的集合以存储转换后的值
        converted_point_set = set()

        # 逐个处理 point_set 中的元素
        print(f"转换前索引{point_set}")
        for point in point_set:
            try:
                # 将字符串类型元素转换为整数，匹配矩阵列索引 (MATLAB 索引通常从 1 开始)
                column_index = int(point)  # 将字符串转换为整数列索引
                # print(f"数字是：{int(leak[column_index,0])}")
                point_list.append(int(leak[column_index,0]))

                # if column_index < 1 or column_index > leak.shape[0]:  # 检查索引范围
                #     raise IndexError(
                #         f"point_set 中的元素值 '{point}' 超出 leak 数组的列范围。"
                #         f"(有效索引范围: 1 到 {leak.shape[0]})"
                #     )

                # MATLAB 是 1-based 索引；Python 是 0-based 索引
                # leak_value = leak[0, column_index - 1]  # 转换为 Python 的 0-based 索引
                # converted_point_set.add(leak_value)  # 将转换后的值添加到新的集合
            except ValueError:
                print(f"point_set 中的元素 '{point}' 不是有效的数字，跳过处理。")
            except IndexError as e:
                print(e)
        point_list.sort(key = int)
        return point_list
    except Exception as e:
        print(f"转换过程失败: {e}")


filepath = 'E:\\PZH\\code\\F-Test-Analysis_2bit\\Ftest_key_6802点_2bitpv0.1_noise10_1.txt' # 输入文本
# filepath = 'E:\PZH\code\F-Test-Analysis_2bit\Ftest_key_6802点_2bitpv0.1_noise10_1.txt' # 输入文本
# filewrite = 'E:\\PZH\\code\\F-Test-Analysis_2bit\\联合子密钥泄露情况_random490点_2bitpv0.1_1.txt' # 输出文本
# filewrite = 'E:\\PZH\\code\\F-Test-Analysis_2bit\\联合子密钥泄露情况_2bitpv2.6_noise10_1.txt' # 输出文本
# filewrite = 'E:\\PZH\\code\\F-Test-Analysis_2bit\\联合子密钥泄露情况_2bitpv500_1.txt' # 输出文本
# filewrite = 'E:\\PZH\\code\\F-Test-Analysis_2bit\\联合子密钥泄露情况_2bitpv2.5_noise10_1.txt' # 输出文本
filewrite = 'E:\\PZH\\code\\F-Test-Analysis_2bit\\test.txt' # 输出文本
fo = open(filewrite, "w") # 覆写文本

# leak_file = 'E:/PZH/code/F-Test-Analysis_2bit/leak/4_6802_2100_mRMRleak_noise10.mat'
leak_file = 'E:/PZH/code/F-Test-Analysis_2bit/leak/4_6802_200_leak.mat'
# 子密钥开始和结束的位置（一定记得修改！！！）
key_start = 0
key_end = key_start + 16
threshold = 1.5  # 自己定阈值，注释则默认0

# 计算泄露功耗点数
# occurrences = count_relevant_bits(filepath)
# if occurrences != -1:
#     print(f"有泄露的功耗点数为 {occurrences} 个")

#     fo.write(f"有泄露的功耗点数为 {occurrences} 个\r\n") # \r\n是win的换行方式
# 统计子密钥相关泄露功耗点
results = analyze_bit_samples_with_features(filepath,key_start,key_end,threshold)
point = 1 # 点数限制，超过这个点数的子密钥组才被选择
num = 0
point_set = set() # 点数矩阵
if results:
    all_unique_features = set()
    for bit, data in results.items():
        if data['count'] >= point:
            num+=1
            print(f"******联合子密钥: {bit}******")
            fo.write(f"******联合子密钥: {bit}******\r\n")
            print(f"  特征点: {', '.join(data['features'])}")  # 打印特征点列表
            fo.write(f"  特征点: {', '.join(data['features'])}\r\n")
            print(f"  特征点数量: {data['count']}\n")
            fo.write(f"  特征点数量: {data['count']}\r\n")
            all_unique_features.update(data['features'])  # 将当前 features 列表中的所有特征点添加到集合中

            # 新增：输出有效pv比例（已经无用）
            print(f"  PV累计值为： {data['sum_pv']:.2f}")
            fo.write(f"  PV累计值为： {data['sum_pv']:.2f}\n")
            print(f"  有效PV比例: {data['pv_ratio']:.2%}")  # 格式化为百分比，保留两位小数
            fo.write(f"  有效PV比例: {data['pv_ratio']:.2%}\n")  # 同步写入文件
            print(f"  PV平均值: {data['avg_pv']:.2f}")
            fo.write(f"  PV平均值: {data['avg_pv']:.2f}\n")
            print(f"  PV中位数: {data['median_pv']:.2f}")
            print(f"  PV平均占比值: {data['avg_pv_ratio']:.2f}")
            fo.write(f"  PV平均占比值: {data['avg_pv_ratio']:.2f}\r\n")


            features = data['features']
            point_set.update(features)

            # try:
            #     data_list = [int(s) for s in features]  # 使用列表推导式进行转换
            # except ValueError:
            #     print("Error: 列表中包含无法转换为整数的字符串")
            #     exit()
            #
            # # 文件名
            # # original_file_path = r'E:\PZH\code\F-Test-Analysis_2bit\mat\2bit噪声\RR_20W_6802_2025_2_13_11h20m23s_0.99687_byte64_noise10_keyset0_16_noise10.mat'  # .mat 文件路径
            # # new_file_path = r'E:\PZH\code\F-Test-Analysis_2bit\outputmat\RR_20W_6802_2025_2_13_11h20m23s_0.99687_byte64_noise10_keyset0_16_noise10pv0.5.mat'  # .mat 文件路径
            # original_file_path = r'E:\PZH\code\F-Test-Analysis_2bit\mat\2bit噪声\RS_8K_6802_2025_01_31_13h00m_0.99717_byte64_noise10_keyset0_16_noise10.mat'  # .mat 文件路径
            # new_file_path = r'E:\PZH\code\F-Test-Analysis_2bit\outputmat\RS_8K_6802_2025_01_31_13h00m_0.99717_byte64_noise10_keyset0_16_noise10pv2.5.mat'  # .mat 文件路径
            # dataset_name1 = bit.replace(" ", "_")  # 貌似直接将空格替换为空，matlab还是会检测到空格F
            # dataset_name = dataset_name1.replace("\t","_")  # 最终示例格式bit_0_bit_1
            # # # 调用函数
            # add_list_to_mat_new_file(original_file_path, new_file_path, data_list, dataset_name)

    point_set = sorted(point_set, key=int)

    # 05.12新增，用于转换和映射裁剪前的点
    point_list = set_to_list(point_set) # 方法1：没有经过FSLA和random的裁剪
    # point_list = setleak_to_list(point_set,leak_file,leak_name='leak') # 方法2：经过裁剪

    # 计算泄露功耗点数
    num_unique_features = len(all_unique_features)
    print(f"总共有 {num_unique_features} 个不重复的特征点")
    fo.write(f"总共有 {num_unique_features} 个不重复的特征点\r\n")
    print(f"特征点汇总：{point_list}")
    fo.write(f"特征点汇总：{point_list}\r\n")


            # # 加载数据
            # # 使用 h5py 加载 MATLAB v7.3 文件
            # with h5py.File(original_file_path, 'r') as file:
            #     # 提取数据
            #     key = np.array(file['key']).T  # 假设 'key' 是文件中的变量名，转置以匹配 MATLAB 的默认存储方式
            #     plaintext = np.array(file['plaintext']).T  # 假设 'plaintext' 是文件中的变量名，转置以匹配 MATLAB 的默认存储方式
            #
            # # 设置子密钥索引和运算类型
            # numbers = re.findall(r'\d+', dataset_name)  # 提取所有连续的数字
            # # 将提取的字符串数字转换为整数
            # sub_indices = [int(num) for num in numbers]
            # operation = 'bitand'  # 使用“二进制位置”拼接运算
            #
            # # 生成密钥标签
            # k_label = Ftest_label2(key, sub_indices, operation)
            # k_name = dataset_name.replace("bit","k")
            # # data[k_name] = k_label  # 将标签保存到数据中
            # add_list_to_mat_new_file(original_file_path, new_file_path, k_label, k_name)
            #
            # # 生成明文标签
            # p_label = Ftest_label2(plaintext, sub_indices, operation)
            # p_name = dataset_name.replace("bit", "p")
            # # data[p_name] = p_label  # 将标签保存到数据中
            # add_list_to_mat_new_file(original_file_path, new_file_path, p_label, p_name)




    # print(f"特征点多于{str(point)}的联合密钥有{str(num)}个")





