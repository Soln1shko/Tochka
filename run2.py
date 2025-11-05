import sys
import collections
from typing import Dict, Set, List, Tuple, Optional
import copy

def solve(edges: List[Tuple[str, str]]) -> List[str]:
    """
    Решение задачи об изоляции вируса.

    Args:
        edges: список коридоров в формате (узел1, узел2)

    Returns:
        список отключаемых коридоров в формате "Шлюз-узел"
    """
    graph: Dict[str, Set[str]] = collections.defaultdict(set)
    gateways: Set[str] = set()

    for u, v in edges:
        graph[u].add(v)
        graph[v].add(u)
        if u.isupper():
            gateways.add(u)
        if v.isupper():
            gateways.add(v)

    virus_pos = 'a'
    result: List[str] = []

    def bfs(start_node: str, current_graph: Dict[str, Set[str]]) -> Dict[str, int]:
        if start_node not in current_graph:
            return {}
        queue = collections.deque([(start_node, 0)])
        distances = {start_node: 0}
        visited = {start_node}
        
        while queue:
            current, dist = queue.popleft()
            for neighbor in sorted(list(current_graph.get(current, set()))):
                if neighbor not in visited:
                    visited.add(neighbor)
                    distances[neighbor] = dist + 1
                    queue.append((neighbor, dist + 1))
        return distances

    def get_virus_move(pos: str, current_graph: Dict[str, Set[str]], current_gateways: Set[str]) -> Optional[str]:
        distances_from_virus = bfs(pos, current_graph)
        
        reachable_gateways = []
        for gw in sorted(list(current_gateways)):
            if gw in distances_from_virus:
                reachable_gateways.append((distances_from_virus[gw], gw))

        if not reachable_gateways:
            return None

        reachable_gateways.sort()
        target_gateway = reachable_gateways[0][1]

        distances_to_target = bfs(target_gateway, current_graph)
        current_dist_to_target = distances_to_target.get(pos)

        if current_dist_to_target is None or current_dist_to_target <= 1:
            return pos

        for neighbor in sorted(list(current_graph.get(pos, set()))):
            if distances_to_target.get(neighbor) == current_dist_to_target - 1:
                return neighbor
        
        return pos

    while True:
        # Проверка, может ли вирус двигаться. Если нет, игра окончена.
        if get_virus_move(virus_pos, graph, gateways) is None:
            break

        possible_cuts = []
        for gw in sorted(list(gateways)):
            for neighbor in sorted(list(graph.get(gw, set()))):
                if not neighbor.isupper():
                    possible_cuts.append(f"{gw}-{neighbor}")
        
        # Если не осталось коридоров для отключения, выходим.
        if not possible_cuts:
            break
            
        # Инициализируем best_cut первым возможным ходом.
        best_cut = possible_cuts[0]

        for cut_edge_str in possible_cuts:
            gw_to_cut, node_to_cut = cut_edge_str.split('-')
            
            temp_graph = copy.deepcopy(graph)
            temp_graph[gw_to_cut].remove(node_to_cut)
            temp_graph[node_to_cut].remove(gw_to_cut)
            
            next_virus_pos = get_virus_move(virus_pos, temp_graph, gateways)
            
            is_safe = False
            # Если вирус заблокирован, ход точно безопасен
            if next_virus_pos is None:
                is_safe = True
            # Иначе проверяем, не окажется ли он рядом со шлюзом
            else:
                is_safe_check = True
                for neighbor in temp_graph.get(next_virus_pos, set()):
                    if neighbor in gateways:
                        is_safe_check = False
                        break
                is_safe = is_safe_check
            
            if is_safe:
                best_cut = cut_edge_str
                break
        
        gw, node = best_cut.split('-')
        result.append(best_cut)
        graph[gw].remove(node)
        graph[node].remove(gw)

        new_virus_pos = get_virus_move(virus_pos, graph, gateways)
        if new_virus_pos:
            virus_pos = new_virus_pos

    return result

def main():
    edges = []
    try:
        # Используем sys.stdin.readlines() для более простого чтения всех строк
        lines = sys.stdin.readlines()
        for line in lines:
            line = line.strip()
            if line:
                node1, sep, node2 = line.partition('-')
                if sep:
                    edges.append((node1, node2))
    except (IOError, EOFError):
        pass

    result = solve(edges)
    for edge in result:
        print(edge)

if __name__ == "__main__":
    main()

