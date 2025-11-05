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
    # Построение графа
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
        """BFS для нахождения кратчайших расстояний от start_node"""
        if start_node not in current_graph:
            return {}
        
        queue = collections.deque([(start_node, 0)])
        distances = {start_node: 0}
        visited = {start_node}
        
        while queue:
            current, dist = queue.popleft()
            for neighbor in current_graph.get(current, set()):
                if neighbor not in visited:
                    visited.add(neighbor)
                    distances[neighbor] = dist + 1
                    queue.append((neighbor, dist + 1))
        
        return distances
    
    def get_virus_target_and_move(pos: str, current_graph: Dict[str, Set[str]], 
                                   current_gateways: Set[str]) -> Tuple[Optional[str], Optional[str]]:
        """
        Определяет целевой шлюз и следующий ход вируса.
        Возвращает (target_gateway, next_position) или (None, None) если вирус заблокирован
        """
        # Проверяем, достиг ли вирус шлюза
        if pos in current_gateways:
            return None, None
        
        # Находим расстояния от текущей позиции вируса
        distances_from_virus = bfs(pos, current_graph)
        
        # Находим достижимые шлюзы
        reachable_gateways = []
        for gw in current_gateways:
            if gw in distances_from_virus:
                reachable_gateways.append((distances_from_virus[gw], gw))
        
        if not reachable_gateways:
            return None, None
        
        # Сортируем по расстоянию, затем лексикографически
        reachable_gateways.sort()
        target_gateway = reachable_gateways[0][1]
        
        # Находим соседей, которые ближе к целевому шлюзу
        distances_to_target = bfs(target_gateway, current_graph)
        current_dist = distances_to_target.get(pos)
        
        if current_dist is None:
            return None, None
        
        # Если вирус рядом со шлюзом, он переходит в него
        if current_dist == 1:
            return target_gateway, target_gateway
        
        # Выбираем следующий узел - лексикографически наименьший среди оптимальных
        candidates = []
        for neighbor in current_graph.get(pos, set()):
            neighbor_dist = distances_to_target.get(neighbor)
            if neighbor_dist is not None and neighbor_dist == current_dist - 1:
                candidates.append(neighbor)
        
        if candidates:
            candidates.sort()
            return target_gateway, candidates[0]
        
        return None, None
    
    def simulate_full_game(start_pos: str, start_graph: Dict[str, Set[str]], 
                          start_gateways: Set[str], first_cut: str) -> Optional[List[str]]:
        """
        Полная симуляция игры с заданным первым ходом.
        Возвращает последовательность отключений или None, если вирус достигает шлюза.
        """
        # Копируем состояние
        sim_graph = collections.defaultdict(set)
        for node, neighbors in start_graph.items():
            sim_graph[node] = neighbors.copy()
        
        sim_gateways = start_gateways.copy()
        sim_pos = start_pos
        sim_result = []
        
        # Применяем первый ход
        gw, node = first_cut.split('-')
        sim_graph[gw].discard(node)
        sim_graph[node].discard(gw)
        sim_result.append(first_cut)
        
        # Двигаем вирус
        _, next_pos = get_virus_target_and_move(sim_pos, sim_graph, sim_gateways)
        if next_pos is None:
            return sim_result  # Вирус изолирован
        if next_pos in sim_gateways:
            return None  # Вирус достиг шлюза
        sim_pos = next_pos
        
        # Продолжаем симуляцию
        max_iterations = 200  # Защита от бесконечного цикла
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            
            # Проверяем, изолирован ли вирус
            target, next_move = get_virus_target_and_move(sim_pos, sim_graph, sim_gateways)
            if target is None:
                return sim_result  # Победа!
            
            # Собираем возможные отключения
            possible_cuts = []
            for gw in sorted(sim_gateways):
                for neighbor in sorted(sim_graph.get(gw, set())):
                    if not neighbor.isupper():
                        possible_cuts.append(f"{gw}-{neighbor}")
            
            if not possible_cuts:
                return None  # Не можем отключить больше коридоров
            
            # Выбираем первый возможный ход (лексикографический порядок)
            cut = possible_cuts[0]
            gw, node = cut.split('-')
            sim_graph[gw].discard(node)
            sim_graph[node].discard(gw)
            sim_result.append(cut)
            
            # Двигаем вирус
            _, next_pos = get_virus_target_and_move(sim_pos, sim_graph, sim_gateways)
            if next_pos is None:
                return sim_result  # Вирус изолирован
            if next_pos in sim_gateways:
                return None  # Вирус достиг шлюза
            sim_pos = next_pos
        
        return None  # Не удалось изолировать за разумное время
    
    # Основной цикл игры
    while True:
        # Проверяем, изолирован ли вирус
        target, next_move = get_virus_target_and_move(virus_pos, graph, gateways)
        if target is None:
            break  # Вирус изолирован, игра окончена
        
        # Собираем все возможные отключения
        possible_cuts = []
        for gw in sorted(gateways):
            for neighbor in sorted(graph.get(gw, set())):
                if not neighbor.isupper():
                    possible_cuts.append(f"{gw}-{neighbor}")
        
        if not possible_cuts:
            break
        
        # Пробуем каждое отключение и выбираем первое, которое приводит к победе
        best_cut = None
        for cut in possible_cuts:
            sim_result = simulate_full_game(virus_pos, graph, gateways, cut)
            if sim_result is not None:
                # Это отключение приводит к победе
                best_cut = cut
                break
        
        if best_cut is None:
            # Если симуляция не нашла безопасный ход, используем первый возможный
            best_cut = possible_cuts[0]
        
        # Применяем выбранное отключение
        gw, node = best_cut.split('-')
        graph[gw].discard(node)
        graph[node].discard(gw)
        result.append(best_cut)
        
        # Двигаем вирус
        _, next_pos = get_virus_target_and_move(virus_pos, graph, gateways)
        if next_pos is None:
            break  # Вирус изолирован
        if next_pos in gateways:
            break  # Вирус достиг шлюза (не должно произойти в корректном решении)
        virus_pos = next_pos
    
    return result


def main():
    edges = []
    try:
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
