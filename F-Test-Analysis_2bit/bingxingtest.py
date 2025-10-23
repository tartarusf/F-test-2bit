import multiprocessing

def process_data(data, results_dict):
    """
    处理整个数据集，并将结果添加到结果字典中。
    Args:
        data: 要处理的数据列表。
        results_dict: 用于存储结果的共享字典，键为原始数据的索引，值为每个索引对应的写入内容列表。
    """
    for i, num in enumerate(data):
        # 模拟每次循环写入内容数量不同
        write_data = [f"循环{i + 1}次: {num**2 + j} " for j in range(num % 5 + 1)]
        results_dict[i] = write_data  # 将结果添加到共享字典


def parallel_process_list(data, output_filename="output.txt"):
    """
    并行处理列表，保证写入文件的顺序。
    Args:
        data: 要处理的数据列表。
        output_filename: 输出文件名。
    """
    manager = multiprocessing.Manager()  # 创建一个Manager对象，用于创建共享内存对象
    results_dict = manager.dict()  # 创建一个共享字典，用于在多个进程之间共享结果

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:  # 创建一个进程池，进程数为CPU核心数
        pool.starmap(process_data, [(data, results_dict)])  # 使用starmap并行处理数据，直接处理整个列表

    with open(output_filename, 'w') as outfile:  # 打开文件准备写入
        for i in range(len(data)):  # 按照顺序遍历每个索引
            try:
                for item in results_dict[i]:  # 遍历每个索引对应的写入内容列表
                    outfile.write(item + '\n')  # 写入内容到文件
            except KeyError as e:
                print(f"警告: 索引 {i} 出现 KeyError. 可能的原因：数据处理过程中出现错误。")


if __name__ == "__main__":
    my_list = list(range(20))  # 测试数据
    output_filename = "output.txt"  # 输出文件名
    parallel_process_list(my_list, output_filename)  # 执行并行处理
    print(f"结果已写入 {output_filename}")  # 打印完成信息
