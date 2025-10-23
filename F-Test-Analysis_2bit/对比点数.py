import ast # 用于安全地将字符串形式的列表转换为实际列表

def parse_feature_points_from_file(filepath):
    """
    从指定的文件中读取并解析特征点。
    文件每行格式应为 "特征点为：['item1', 'item2', ...]"
    返回一个包含所有唯一特征点的集合。
    """
    all_points = set()
    try:
        with open(filepath, 'r', encoding='gbk') as f: # 指定编码以防中文问题
            for line in f:
                line = line.strip()
                if line.startswith("特征点汇总："):
                    # 提取列表字符串部分
                    list_str_part = line.replace("特征点汇总：", "").strip()
                    try:
                        # 安全地将字符串转换为实际的 Python 列表
                        points_list = ast.literal_eval(list_str_part)
                        if isinstance(points_list, list):
                            all_points.update(points_list) # 将列表中的所有元素添加到集合
                        else:
                            print(f"警告: 在文件 {filepath} 中解析行 '{line}' 时未得到列表。")
                    except (ValueError, SyntaxError) as e:
                        print(f"警告: 无法解析文件 {filepath} 中的行: '{line}'. 错误: {e}")
                # elif line: # 如果行不为空且不是期望的格式
                #     print(f"警告: 文件 {filepath} 中发现非标准格式行: '{line}'")
    except FileNotFoundError:
        print(f"错误: 文件 {filepath} 未找到。")
        return None
    except Exception as e:
        print(f"读取或解析文件 {filepath} 时发生错误: {e}")
        return None
    return all_points

def compare_points_occurrence(points_set1, points_set2, set1_name="集合1", set2_name="集合2"):
    """
    计算 points_set1 中的每个点在 points_set2 中出现的总次数。
    由于 points_set2 是一个集合，每个点在其中只会出现一次或零次。
    所以这个函数实际上是计算 points_set1 中有多少个点也存在于 points_set2 中。
    """
    if points_set1 is None or points_set2 is None:
        print("错误：一个或两个输入集合为空，无法比较。")
        return 0

    occurrence_count = 0
    common_points = []

    for point in points_set1:
        if point in points_set2:
            occurrence_count += 1
            common_points.append(point)

    print(f"\n比较 {set1_name} 与 {set2_name}:")
    print(f"重复点数（即共同点的数量）: {occurrence_count}")
    if occurrence_count > 0:
        print(f"共同点列表: {common_points[:]}")
    else:
        print("两个集合之间没有共同点。")
    return occurrence_count, common_points

# --- 主程序逻辑 ---
if __name__ == "__main__":
    # 替换为你的实际文件名
    # file1_path = '联合子密钥泄露情况_FSLA490点_2bit_1.txt'
    file1_path = '联合子密钥泄露情况_2bitpv500_1.txt'
    file2_path = '联合子密钥泄露情况_FSLA490点_2bit_1.txt'

    print(f"正在从文件 '{file1_path}' 解析特征点...")
    points_from_file1 = parse_feature_points_from_file(file1_path)

    print(f"\n正在从文件 '{file2_path}' 解析特征点...")
    points_from_file2 = parse_feature_points_from_file(file2_path)

    if points_from_file1 is not None:
        points_from_file1 = sorted(points_from_file1,key=int)
        print(f"\n从 '{file1_path}' 提取的唯一特征点: {list(points_from_file1)[:]}")
        print(f"总共 {len(points_from_file1)}\n")
    if points_from_file2 is not None:
        points_from_file2 = sorted(points_from_file2,key=int)
        print(f"从 '{file2_path}' 提取的唯一特征点: {list(points_from_file2)[:]}")
        print(f"总共 {len(points_from_file2)}")

    # 比较文件1的点在文件2中出现的次数
    if points_from_file1 and points_from_file2: # 确保两个集合都成功加载
        occurrence_count, common_points = compare_points_occurrence(points_from_file1, points_from_file2, f"文件 '{file1_path}'", f"文件 '{file2_path}'")
        print(f"\n{file1_path}中的出现率为{int(occurrence_count)/len(points_from_file1)}")
        print(f"{file2_path}中的出现率为{int(occurrence_count)/len(points_from_file2)}")

        # 如果需要双向比较，可以再调用一次，交换参数
        # compare_points_occurrence(points_from_file2, points_from_file1, f"文件 '{file2_path}'", f"文件 '{file1_path}'")