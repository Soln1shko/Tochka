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

    def get_next_move(pos: str, target_gw: str) -> Optional[str]:
        """Определяет следующий шаг вируса к цели."""
        distances_to_target = bfs(target_gw, graph)
        current_dist = distances_to_target.get(pos)
        
        if current_dist is None or current_dist <= 1:
            return None
            
        for neighbor in sorted(list(graph[pos])):
            if distances_to_target.get(neighbor) == current_dist - 1:
                return neighbor
        return None

    while True:
        distances = bfs(virus_pos, graph)
        
        reachable_gateways = []
        for gw in gateways:
            if gw in distances:
                reachable_gateways.append((distances[gw], gw))
        
        if not reachable_gateways:
            break
        
        reachable_gateways.sort()
        target_gateway = reachable_gateways[0][1]
    
        next_virus_pos = get_next_move(virus_pos, target_gateway)
        
        if next_virus_pos is None:
            result.append(f"{target_gateway}-{virus_pos}")
            graph[target_gateway].remove(virus_pos)
            graph[virus_pos].remove(target_gateway)
            continue
        
        gateways_at_next = []
        for gw in sorted(gateways):
            if gw in graph.get(next_virus_pos, set()):
                gateways_at_next.append(gw)
        
        if gateways_at_next:
            gw_to_cut = gateways_at_next[0]
            result.append(f"{gw_to_cut}-{next_virus_pos}")
            graph[gw_to_cut].remove(next_virus_pos)
            graph[next_virus_pos].remove(gw_to_cut)
        else:
            distances_to_target = bfs(target_gateway, graph)
            path_node = virus_pos
            
            while distances_to_target.get(path_node, -1) > 1:
                current_dist = distances_to_target.get(path_node)
                for neighbor in sorted(list(graph[path_node])):
                    if distances_to_target.get(neighbor) == current_dist - 1:
                        path_node = neighbor
                        break
            
            result.append(f"{target_gateway}-{path_node}")
            graph[target_gateway].remove(path_node)
            graph[path_node].remove(target_gateway)
        
        distances_after = bfs(virus_pos, graph)
        reachable_after = []
        for gw in gateways:
            if gw in distances_after:
                reachable_after.append((distances_after[gw], gw))
        
        if reachable_after:
            reachable_after.sort()
            new_target = reachable_after[0][1]
            virus_pos = get_next_move(virus_pos, new_target) or virus_pos

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
