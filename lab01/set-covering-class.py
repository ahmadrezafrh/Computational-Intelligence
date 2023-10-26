from random import random
from math import ceil
from functools import reduce
from collections import namedtuple, deque
from queue import PriorityQueue
import numpy as np
from tqdm.auto import tqdm


class set_covering:
    
    def __init__(self, num_sets, problem_size):
        self.State = namedtuple('State', ['taken', 'not_taken'])
        self.num_sets = num_sets
        self.problem_size = problem_size
        self.sets = tuple(np.array([random() < 0.2 for _ in range(self.problem_size)]) for _ in range(self.num_sets))
        assert self.goal_check(self.State(set(range(self.num_sets)), set())), "Probelm not solvable"
        
        
    def __covered(self, state):
        return reduce(
            np.logical_or,
            [self.sets[i] for i in state.taken],
            np.array([False for _ in range(self.problem_size)]),
        )
        
        
    def __goal_check(self, state):
        return np.all(self.__covered(state))


    def __init_state(self):
        return self.State(set(), set(range(self.num_sets)))
    
    def __missing_size(self, state):
        return self.problem_size - sum(self.__covered(state))
    
    def __f(self, state):
        cost = self.__missing_size(state)
        return cost
    
    
    
    
    def Depth_first(self):
        frontier = deque()
        state = self.__init_state()
        frontier.append(state)
        
        counter = 0
        current_state = frontier.pop()
        with tqdm(total=None) as pbar:
            while not self.__goal_check(current_state):
                counter += 1
                for action in current_state[1]:
                    new_state = self.State(
                        current_state.taken ^ {action},
                        current_state.not_taken ^ {action},
                    )
                    frontier.append(new_state)
                current_state = frontier.pop()
                pbar.update(1)
                
        print(f"Depth approach: Solved in {counter:,} steps ({len(current_state.taken)} tiles)")
        
        
        
    def breadth_first(self):
        frontier = deque()
        state = self.__init_state()
        frontier.append(state)

        counter = 0
        current_state = frontier.popleft()

        while not self.goal_check(current_state):
            counter += 1
            for action in current_state[1]:
                new_state = self.State(
                    current_state.taken ^ {action},
                    current_state.not_taken ^ {action},
                )
                frontier.append(new_state)
            current_state = frontier.popleft()


        print(f"Breadth approach: Solved in {counter:,} steps ({len(current_state.taken)} tiles)")




    def greedy_best_first(self):
        frontier = PriorityQueue()
        state = self.State(set(), set(range(self.num_sets)))
        frontier.put((self.__f(state), state))

        counter = 0
        _, current_state = frontier.get()

        while not self.goal_check(current_state):
            counter += 1
            for action in current_state[1]:
                new_state = self.State(
                    current_state.taken ^ {action},
                    current_state.not_taken ^ {action},
                )
                frontier.put((self.__f(new_state), new_state))
            a, current_state = frontier.get()
            
        print(f"Greedy Best First approach: Solved in {counter:,} steps ({len(current_state.taken)} tiles)")


    def greedy_custom(self, cost_func):
        frontier = PriorityQueue()
        state = self.State(set(), set(range(self.num_sets)))
        frontier.put((cost_func(state, self.__covered, self.problem_size, self.num_sets, self.sets), state))

        counter = 0
        _, current_state = frontier.get()

        while not self.goal_check(current_state):
            counter += 1
            for action in current_state[1]:
                new_state = self.State(
                    current_state.taken ^ {action},
                    current_state.not_taken ^ {action},
                )
                frontier.put((cost_func(new_state), new_state))
            a, current_state = frontier.get()
            
        print(f"Greedy Best First approach with optimum function: Solved in {counter:,} steps ({len(current_state.taken)} tiles)")



def h_by_prof(state, covered, problem_size, num_sets, sets):
    already_covered = covered(state)
    if np.all(already_covered):
        return 0
    missing_size = problem_size - sum(already_covered)
    candidates = sorted((sum(np.logical_and(s, np.logical_not(already_covered))) for s in sets), reverse=True)
    taken = 1
    while sum(candidates[:taken]) < missing_size:
        taken += 1
    return taken


def f_by_prof(state):
    return len(state.taken) + h_by_prof(state)



def h_proposed(state, covered, problem_size, num_sets, sets):
    coverd_problems = covered()
    not_covered_indexes = np.where(np.logical_not(coverd_problems))[0]
    covered_indexes = np.where(coverd_problems)[0]
    not_taken_states = state.not_taken
    
    tot_false_not_covered = 0
    tot_truth_covered = 0
    for ind_state in not_taken_states:
        tot_false_not_covered += sum(np.logical_not([covered(single_state(ind_state, num_sets))[n] for n in not_covered_indexes]))
        tot_truth_covered += sum([covered(single_state(ind_state, num_sets))[n] for n in covered_indexes])
        
    return len(state.taken) + tot_truth_covered + tot_false_not_covered



def single_state(num, num_sets):
    State = namedtuple('State', ['taken', 'not_taken'])
    state = State(set({num}), set(range(num_sets)))
    state.not_taken.remove(num)
    return state