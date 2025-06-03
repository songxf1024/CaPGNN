import re

'''
Please enter data:
part 0 has 663181 nodes and 358107 are inside the partition
part 0 has 10258591 edges and 6949100 are inside the partition
part 1 has 663876 nodes and 358740 are inside the partition
part 1 has 10315210 edges and 7005719 are inside the partition


Total number of nodes: 1327057 
Total number of nodes: 716847
**************************************************
'''


while True:
    # Entered text dataEntered text data
    data = ""
    print("Please enter data:")
    while True:
        line = input().strip()  # Remove the whitespace characters before and after the input line
        if line == "":  # If the input is a blank line, end
            break
        data += line + "\n"  # Add each row of data to the full data
    
    
    # Regular expressions match nodes and inner nodes, edges and inner edges
    node_pattern = re.compile(r"part \d+ has (\d+) nodes and (\d+) are inside the partition")
    edge_pattern = re.compile(r"part \d+ has (\d+) edges and (\d+) are inside the partition")
    
    # Initialize the total number of nodes, number of internal nodes, total number of edges and number of inner edges
    total_nodes = 0
    total_inside_nodes = 0
    total_edges = 0
    total_inside_edges = 0
    
    # Analyze node information
    node_matches = node_pattern.findall(data)
    for match in node_matches:
        total_nodes += int(match[0])  # Total number of nodes
        total_inside_nodes += int(match[1])  # Number of internal nodes
    
    # Analyze edge information
    edge_matches = edge_pattern.findall(data)
    for match in edge_matches:
        total_edges += int(match[0])  # Total number of edges
        total_inside_edges += int(match[1])  # Number of inner edges

    
    # Output result
    print()
    print(f"Total number of nodes: {total_nodes}")
    print(f"Total number of points: {total_inside_nodes}")
    print(f"Total number of edges: {total_edges}")
    print(f"Total inner edge number: {total_inside_edges}")
    print("*" * 50)
    print()
