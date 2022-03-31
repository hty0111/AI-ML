def my_search(maze):
    """
    任选深度优先搜索算法、最佳优先搜索（A*)算法实现其中一种
    :param maze: 迷宫对象
    :return :到达目标点的路径 如：["u","u","r",...]
    """

    path = []
    
    # -----------------请实现你的算法代码--------------------------------------
    start = maze.sense_robot()
    root = SearchTree(loc=start)
    queue = [root]
    h, w, _ = maze.maze_data.shape
    is_visit_m = np.zeros((h, w), dtype=np.int32)

    while True:                
        current_node = queue[-1]            # 当前位置为栈顶元素

        if current_node.loc == maze.destination:
            path = back_propagation(current_node)
            break
        # print(current_node.loc)

        # 如果已经搜索过，弹栈
        if is_visit_m[current_node.loc]:
            queue.pop()
            continue

        # 如果是叶节点则扩展子节点
        if current_node.is_leaf():
            expand(maze, is_visit_m, current_node)

        # 如果没有子节点，弹栈
        if not current_node.children:
            queue.pop()
        else:
            for child in current_node.children:
                queue.append(child)

        is_visit_m[current_node.loc] = 1    # 标记该位置已被搜索
        
    # -----------------------------------------------------------------------
    return path
