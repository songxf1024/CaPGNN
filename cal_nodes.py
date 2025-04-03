import re

'''
请输入数据：
part 0 has 663181 nodes and 358107 are inside the partition
part 0 has 10258591 edges and 6949100 are inside the partition
part 1 has 663876 nodes and 358740 are inside the partition
part 1 has 10315210 edges and 7005719 are inside the partition


总节点数: 1327057
总内节点数: 716847
**************************************************
'''


while True:
    # 输入的文本数据
    data = ""
    print("请输入数据：")
    while True:
        line = input().strip()  # 去除输入行的前后空白字符
        if line == "":  # 如果输入是空白行，则结束
            break
        data += line + "\n"  # 将每行数据添加到完整数据中
    
    
    # 正则表达式匹配节点和内节点、边和内边
    node_pattern = re.compile(r"part \d+ has (\d+) nodes and (\d+) are inside the partition")
    edge_pattern = re.compile(r"part \d+ has (\d+) edges and (\d+) are inside the partition")
    
    # 初始化总节点数、内节点数、总边数和内边数
    total_nodes = 0
    total_inside_nodes = 0
    total_edges = 0
    total_inside_edges = 0
    
    # 解析节点信息
    node_matches = node_pattern.findall(data)
    for match in node_matches:
        total_nodes += int(match[0])  # 总节点数
        total_inside_nodes += int(match[1])  # 内节点数
    
    # 解析边信息
    edge_matches = edge_pattern.findall(data)
    for match in edge_matches:
        total_edges += int(match[0])  # 总边数
        total_inside_edges += int(match[1])  # 内边数

    
    # 输出结果
    print()
    print(f"总节点数: {total_nodes}")
    print(f"总内点数: {total_inside_nodes}")
    print(f"总的边数: {total_edges}")
    print(f"总内边数: {total_inside_edges}")
    print("*" * 50)
    print()
