import h5py
import numpy as np
# 加载 MATLAB 文件
mat_file = 'random490_leak.mat'  # 替换为你的文件路径



# 假设原始 point_set 是一个集合
point_set = {'1', '13', '101', '2', '105'}  # 示例点集合 (可以动态填充)
print(f"原始 point_set: {point_set}")

leak_name = 'random_leak'
def setleak_to_list(point_set,mat_file,leak_name):
    point_list = []
    try:
        # 使用 h5py 打开 .mat 文件
        with h5py.File(mat_file, 'r') as f:
            # 获取 'leak' 数组
            if leak_name not in f:
                raise KeyError(f"文件 '{mat_file}' 中未找到数据集{leak_name}。")
            # leak 是 HDF5 数据集对象，需要读取
            leak = f[leak_name][:]
            # leak = f['random_leak'][:]
            print(f"'leak' 数组的形状: {leak.shape}")
    except Exception as e:
        print(f"加载文件失败: {e}")
        raise
    try:
        # 创建一个新的集合以存储转换后的值
        converted_point_set = set()

        # 逐个处理 point_set 中的元素
        for point in point_set:
            try:
                # 将字符串类型元素转换为整数，匹配矩阵列索引 (MATLAB 索引通常从 1 开始)
                column_index = int(point)  # 将字符串转换为整数列索引
                print(f"数字是：{int(leak[column_index,0])}")
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

def set_to_list(int_list):
    try:
        int_list = [int(item) for item in point_set]  # 使用列表推导式将字符串转换为整数
        int_list.sort(key = int)
        return int_list

    except ValueError as e:
        print(f"错误: 集合中的元素无法转换为整数。{e}")

point_list1 = set_to_list(point_set)
point_list2 = setleak_to_list(point_set,mat_file,leak_name)

print(f"转换后的整数列表: {point_list1}")
print(f"转换后的 point_set: {point_list2}")
