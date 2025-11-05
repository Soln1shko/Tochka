import sys
import heapq
from typing import List, Tuple, Dict, Set, Optional
from dataclasses import dataclass


@dataclass(frozen=True)
class State:
    """Неизменяемое состояние лабиринта"""
    hallway: Tuple[str, ...]  
    rooms: Tuple[Tuple[str, ...], ...]  
    
    def __lt__(self, other):
        return False  


class AmphipodSolver:
    """Решатель задачи с использованием A* алгоритма"""
    
    EMPTY = '.'
    AMPHIPOD_TYPES = ('A', 'B', 'C', 'D')
    ENERGY_COST = {'A': 1, 'B': 10, 'C': 100, 'D': 1000}
    ROOM_POSITIONS = {'A': 2, 'B': 4, 'C': 6, 'D': 8}  
    FORBIDDEN_STOPS = {2, 4, 6, 8}  
    
    def __init__(self, room_depth: int):
        self.room_depth = room_depth
    
    def parse_input(self, lines: List[str]) -> State:
        """Парсинг входных данных"""
        hallway = [self.EMPTY] * 11
        rooms = [[] for _ in range(4)]
        
        for i in range(2, len(lines) - 1):
            line = lines[i]
            room_chars = []
            for j, char in enumerate(line):
                if char in self.AMPHIPOD_TYPES:
                    room_chars.append(char)
            
            for room_idx, char in enumerate(room_chars):
                if room_idx < 4:
                    rooms[room_idx].insert(0, char)  
        
        return State(
            hallway=tuple(hallway),
            rooms=tuple(tuple(room) for room in rooms)
        )
    
    def is_goal(self, state: State) -> bool:
        """Проверка достижения целевого состояния"""
        for room_idx, amphipod_type in enumerate(self.AMPHIPOD_TYPES):
            room = state.rooms[room_idx]
            if len(room) != self.room_depth:
                return False
            if any(pod != amphipod_type for pod in room):
                return False
        return True
    
    def heuristic(self, state: State) -> int:
        """
        Эвристическая функция для A*.
        Оценивает минимальную стоимость достижения целевого состояния.
        """
        total_cost = 0
        
        for hall_pos, amphipod in enumerate(state.hallway):
            if amphipod != self.EMPTY:
                target_room = ord(amphipod) - ord('A')
                target_pos = self.ROOM_POSITIONS[amphipod]
                distance = abs(hall_pos - target_pos) + 1
                total_cost += distance * self.ENERGY_COST[amphipod]
        
        for room_idx, room in enumerate(state.rooms):
            target_type = self.AMPHIPOD_TYPES[room_idx]
            room_pos = self.ROOM_POSITIONS[target_type]
            
            for depth, amphipod in enumerate(room):
                if amphipod != target_type:
                    target_room_idx = ord(amphipod) - ord('A')
                    target_room_pos = self.ROOM_POSITIONS[amphipod]
                    distance = (depth + 1) + abs(room_pos - target_room_pos) + 1
                    total_cost += distance * self.ENERGY_COST[amphipod]
        
        return total_cost
    
    def can_enter_room(self, state: State, room_idx: int, amphipod: str) -> bool:
        """Проверка возможности входа в комнату"""
        target_type = self.AMPHIPOD_TYPES[room_idx]
        if amphipod != target_type:
            return False
        
        room = state.rooms[room_idx]
        return all(pod == target_type for pod in room)
    
    def can_leave_room(self, state: State, room_idx: int) -> bool:
        """Проверка необходимости покинуть комнату"""
        target_type = self.AMPHIPOD_TYPES[room_idx]
        room = state.rooms[room_idx]
        
        if not room or all(pod == target_type for pod in room):
            return False
        return True
    
    def is_path_clear(self, state: State, from_pos: int, to_pos: int) -> bool:
        """Проверка свободности пути в коридоре"""
        start = min(from_pos, to_pos)
        end = max(from_pos, to_pos)
        
        for pos in range(start, end + 1):
            if pos != from_pos and state.hallway[pos] != self.EMPTY:
                return False
        return True
    
    def get_next_states(self, state: State) -> List[Tuple[State, int]]:
        """Генерация всех возможных следующих состояний"""
        next_states = []
        
        for hall_pos, amphipod in enumerate(state.hallway):
            if amphipod == self.EMPTY:
                continue
            
            room_idx = ord(amphipod) - ord('A')
            if not self.can_enter_room(state, room_idx, amphipod):
                continue
            
            room_pos = self.ROOM_POSITIONS[amphipod]
            if not self.is_path_clear(state, hall_pos, room_pos):
                continue
            
            room = state.rooms[room_idx]
            depth = self.room_depth - len(room) - 1
            distance = abs(hall_pos - room_pos) + depth + 1
            cost = distance * self.ENERGY_COST[amphipod]
            
            new_hallway = list(state.hallway)
            new_hallway[hall_pos] = self.EMPTY
            
            new_rooms = [list(room) for room in state.rooms]
            new_rooms[room_idx] = list(room) + [amphipod]
            
            new_state = State(
                hallway=tuple(new_hallway),
                rooms=tuple(tuple(room) for room in new_rooms)
            )
            next_states.append((new_state, cost))
        
        for room_idx, room in enumerate(state.rooms):
            if not room or not self.can_leave_room(state, room_idx):
                continue
            
            amphipod = room[-1]  
            depth = self.room_depth - len(room)
            room_pos = list(self.ROOM_POSITIONS.values())[room_idx]
            
            for hall_pos in range(11):
                if hall_pos in self.FORBIDDEN_STOPS:
                    continue
                if state.hallway[hall_pos] != self.EMPTY:
                    continue
                if not self.is_path_clear(state, room_pos, hall_pos):
                    continue
                
                distance = depth + 1 + abs(room_pos - hall_pos)
                cost = distance * self.ENERGY_COST[amphipod]
                
                new_hallway = list(state.hallway)
                new_hallway[hall_pos] = amphipod
                
                new_rooms = [list(r) for r in state.rooms]
                new_rooms[room_idx] = list(room[:-1])
                
                new_state = State(
                    hallway=tuple(new_hallway),
                    rooms=tuple(tuple(r) for r in new_rooms)
                )
                next_states.append((new_state, cost))
        
        return next_states
    
    def solve(self, initial_state: State) -> int:
        """
        Очень классный и сложный A* алгоритм для поиска минимальной энергии
        """
        counter = 0
        initial_h = self.heuristic(initial_state)
        frontier = [(initial_h, counter, 0, initial_state)]
        counter += 1
        
        g_score: Dict[State, int] = {initial_state: 0}
        
        visited: Set[State] = set()
        
        while frontier:
            f, _, current_g, current = heapq.heappop(frontier)
            
            if current in visited:
                continue
            
            if self.is_goal(current):
                return current_g
            
            visited.add(current)
            
            for next_state, move_cost in self.get_next_states(current):
                if next_state in visited:
                    continue
                
                tentative_g = current_g + move_cost
                
                if next_state not in g_score or tentative_g < g_score[next_state]:
                    g_score[next_state] = tentative_g
                    h = self.heuristic(next_state)
                    f = tentative_g + h
                    heapq.heappush(frontier, (f, counter, tentative_g, next_state))
                    counter += 1
        
        return -1  


def solve(lines: List[str]) -> int:
    """
    Решение задачи о сортировке в лабиринте

    Args:
        lines: список строк, представляющих лабиринт

    Returns:
        минимальная энергия для достижения целевой конфигурации
    """
    room_depth = len(lines) - 3
    
    solver = AmphipodSolver(room_depth)
    initial_state = solver.parse_input(lines)
    result = solver.solve(initial_state)
    
    return result


def main():
    lines = []
    for line in sys.stdin:
        lines.append(line.rstrip('\n'))

    result = solve(lines)
    print(result)


if __name__ == "__main__":
    main()
