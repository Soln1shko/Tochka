import sys
import collections
from typing import Dict, Set, List, Tuple, Optional

def solve(edges: List[Tuple[str, str]]) -> List[str]:
    """
    Решение задачи об изоляции вируса.

    Args:
        edges: список коридоров в формате (узел1, узел2)

    Returns:
        список отключаемых коридоров в формате "Шлюз-узел"
    """
    graph = collections.defaultdict(set)
    gateways = set()
    for u, v in edges:
        graph[u].add(v)
        graph[v].add(u)
        if u.isupper():
            gateways.add(u)
        if v.isupper():
            gateways.add(v)

    virus_pos = 'a'
    result = []

    def bfs(start_node: str, current_graph: Dict[str, Set[str]]) -> Dict[str, int]:
        """Поиск в ширину для нахождения кратчайших расстояний."""
        if start_node not in current_graph:
            return {}
        queue = collections.deque([(start_node, 0)])
        distances = {start_node: 0}
        while queue:
            current, dist = queue.popleft()
            for neighbor in sorted(list(current_graph[current])):
                if neighbor not in distances:
                    distances[neighbor] = dist + 1
                    queue.append((neighbor, dist + 1))
        return distances

    def get_virus_target_and_path(start_pos: str, current_graph: Dict[str, Set[str]], current_gateways: Set[str]) -> Tuple[Optional[str], Optional[str]]:
        """Определяет цель вируса и критический узел на его пути."""
        distances_from_virus = bfs(start_pos, current_graph)
        
        reachable_gateways = []
        for gw in current_gateways:
            if gw in distances_from_virus:
                reachable_gateways.append((distances_from_virus[gw], gw))
        
        if not reachable_gateways:
            return None, None

        reachable_gateways.sort()
        target_gateway = reachable_gateways[0][1]

        distances_to_gateway = bfs(target_gateway, current_graph)
        
        path_node = start_pos
        while distances_to_gateway.get(path_node, -1) > 1:
            next_step_candidates = []
            current_dist = distances_to_gateway.get(path_node)
            for neighbor in sorted(list(current_graph[path_node])):
                if distances_to_gateway.get(neighbor) == current_dist - 1:
                    next_step_candidates.append(neighbor)
            
            if not next_step_candidates: 
                return None, None 
            
            path_node = next_step_candidates[0]
            
        return target_gateway, path_node

    while True:
        target_gateway, critical_node = get_virus_target_and_path(virus_pos, graph, gateways)

        if target_gateway is None:
            break 

        result.append(f"{target_gateway}-{critical_node}")
        graph[target_gateway].remove(critical_node)
        graph[critical_node].remove(target_gateway)

        distances_from_virus_after_cut = bfs(virus_pos, graph)
        
        new_reachable_gateways = []
        for gw in gateways:
            if gw in distances_from_virus_after_cut:
                new_reachable_gateways.append((distances_from_virus_after_cut[gw], gw))

        if not new_reachable_gateways:
            continue 

        new_reachable_gateways.sort()
        new_target_gateway = new_reachable_gateways[0][1]
        
        distances_to_new_gateway = bfs(new_target_gateway, graph)
        
        next_move = None
        current_dist_to_target = distances_to_new_gateway.get(virus_pos)
        if current_dist_to_target is not None:
            for neighbor in sorted(list(graph[virus_pos])):
                if distances_to_new_gateway.get(neighbor) == current_dist_to_target - 1:
                    next_move = neighbor
                    break
        
        if next_move:
            virus_pos = next_move

    return result


def main():
    """Основная функция для чтения ввода и вывода результата."""
    edges = []
    try:
        while True:
            line = input().strip()
            if not line:
                break
            node1, sep, node2 = line.partition('-')
            if sep:
                edges.append((node1, node2))
    except EOFError:
        pass

    result = solve(edges)
    for edge in result:
        print(edge)


if __name__ == "__main__":
    main()


