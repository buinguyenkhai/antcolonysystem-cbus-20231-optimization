import random
import numpy as np

def calculate_route_cost(route):
    return c[0, route[0]] + np.sum(c[route[:-1], route[1:]]) + c[route[-1], 0]

def move_to_next_point(random_proportion, unvisited_pickup, possible_dropoff):
    # pseudorandom proportional rule
    q = random.random()
    if q <= Q0:
        # make best possible move according to choice_weights
        next_p = np.argmax(random_proportion)
    else:
        # biased exploration based on choice_weights
        normalized_weights = random_proportion / np.sum(random_proportion)
        next_p = np.random.choice(2*n+1, p=normalized_weights)
    # if we picked up new passenger, update unvisited_pickup and possible_dropoff
    if next_p <= n:
        unvisited_pickup[next_p] = 0
        possible_dropoff[next_p-1] = 1
    # if we dropped off a passenger, update possible_dropoff
    else:
        possible_dropoff[next_p-n-1] = 0
    
    return next_p

def local_pheromone_update(phero, choice_info, route):
    '''
    Update pheromone on an ant's route
    1. Evaporate pheromone by (1-ZETA) times
    2. Deposit pheromone with amount ZETA*TAU0
    3. Update choice_info accordingly
    '''
    for i in range(2*n-1):
        phero[route[i], route[i+1]] *= (1-ZETA)
        phero[route[i], route[i+1]] += ZETA*TAU0
        choice_info[route[i], route[i+1]] = phero[route[i], route[i+1]] * HEURISTIC_POWER_BETA[route[i], route[i+1]]

    phero[route[-1], 0] *= (1-ZETA)
    phero[route[-1], 0] += ZETA*TAU0
    choice_info[route[-1], 0] = phero[route[-1], 0] * HEURISTIC_POWER_BETA[route[-1], 0]

    phero[0, route[0]] *= (1-ZETA)
    phero[0, route[0]] += ZETA*TAU0
    choice_info[0, route[0]] = phero[0, route[0]] * HEURISTIC_POWER_BETA[0, route[0]]

def traverse_route(phero, choice_info):
    # pick-up points array, 1 if unvisted else 0
    unvisited_pickup = np.ones(n+1)
    unvisited_pickup[0] = 0 # start at point 0
    # drop-off points array, 1 if we can visit else 0
    possible_dropoff = np.zeros(n)
    route = []
    cur_p = 0 # start from point 0
    # visit all 2n points (except point 0)
    for _ in range(2*n):
        # if bus capacity is not ful;
        if np.count_nonzero(possible_dropoff) < k:
            # choice_points array, 1 if we can visit else 0
            choice_points = np.concatenate((unvisited_pickup, possible_dropoff))
        else: # need to drop passenger if bus full
            # choice_points array, 1 if we can visit else 0
            choice_points = np.concatenate((np.zeros(n+1), possible_dropoff))
        # weights array of visiting each point from current point
        choice_weights = choice_info[cur_p] * choice_points
        # choose next point based on pseudorandom proportional rule and visit next point
        cur_p = move_to_next_point(choice_weights, unvisited_pickup, possible_dropoff)
        route.append(cur_p)
    # update pheromone locally for each ant
    local_pheromone_update(phero, choice_info, route)

    route_cost = calculate_route_cost(route)
    return route, route_cost


def global_pheromone_update(phero, choice_info, best_route, best_cost):
    '''
    Only update on the best ant so far's route
    1. Evaporate pheromone by (1-RHO) times
    2. Deposit pheromone with amount RHO/best_cost
    3. Update choice_info accordingly
    '''
    for i in range(2*n-1):
        phero[best_route[i], best_route[i+1]] *= (1-RHO)
        phero[best_route[i], best_route[i+1]] += RHO/best_cost
        choice_info[best_route[i], best_route[i+1]] = phero[best_route[i], best_route[i+1]] * HEURISTIC_POWER_BETA[best_route[i], best_route[i+1]]

    phero[best_route[-1], 0] *= (1-RHO)
    phero[best_route[-1], 0] += RHO/best_cost
    choice_info[best_route[-1], 0] = phero[best_route[-1], 0] * HEURISTIC_POWER_BETA[best_route[-1], 0]

    phero[0, best_route[0]] *= (1-RHO)
    phero[0, best_route[0]] += RHO/best_cost
    choice_info[0, best_route[0]] = phero[0, best_route[0]] * HEURISTIC_POWER_BETA[0, best_route[0]]

def ant_colony_system():
    # Pheromone matrix
    phero = np.full_like(c, TAU0, dtype=np.float64)
    np.fill_diagonal(phero, 0)
    # Choice info matrix
    choice_info = phero * HEURISTIC_POWER_BETA

    best_route_so_far = None
    best_cost_so_far = float('inf')

    for iter in range(MAX_ITER):
        for _ in range(NUM_ANTS):
            # each ant traverse a route and return the route and cost
            route, cost = traverse_route(phero, choice_info)
            # update best route and best cost
            if cost < best_cost_so_far:
                best_route_so_far = route
                best_cost_so_far = cost
        # update pheromone globally only for the best ant so far        
        global_pheromone_update(phero, choice_info, best_route_so_far, best_cost_so_far)
    
    return best_route_so_far, best_cost_so_far


def read_test_case(text):
    with open(text, 'r') as f:
        lines = f.readlines()
        n, k = map(int, lines[0].split())
        c = [list(map(int, line.split())) for line in lines[1:]]
        c = np.array(c).astype(dtype=np.float64)
        np.fill_diagonal(c, np.inf)
        return c, n, k

def nearest_neighbor_route_cost(n, k, c):
    # pick-up points from 1 to n
    unvisited_pickup = set(i for i in range(1, n+1))
    # drop-off points that we can visit if already picked up the passenger for that point
    possible_dropoff = set()
    route = []
    cur_p = 0 # start from point 0
    # visit all 2n points
    for _ in range(2*n):
        nearest_p = cur_p
        # if the bus capacity is not full
        if len(possible_dropoff) < k:
            # can either pick-up new passenger or drop-off passenger(s) on the bus
            for p in unvisited_pickup.union(possible_dropoff):
                # update nearest point from current point
                if c[cur_p, nearest_p] > c[cur_p, p]: 
                    nearest_p = p
        # if the bus capacity is full
        else:
            # visit drop-off point to drop-off passenger on the bus
            for p in possible_dropoff:
                # update nearest point from current point
                if c[cur_p, nearest_p] > c[cur_p, p]:
                    nearest_p = p
        # if we picked up new passenger, update unvisited_pickup and possible_dropoff
        if nearest_p <= n:
            unvisited_pickup.remove(nearest_p)
            possible_dropoff.add(nearest_p + n)
        # if we dropped off a passenger, update possible_dropoff
        else:
            possible_dropoff.remove(nearest_p)
        route.append(nearest_p)
    # return nearest neighbor route cost
    return calculate_route_cost(route)

if __name__ == '__main__':
    # cost matrix, number of passengers, bus capacity
    text = input('Enter test case: ')
    c, n, k = read_test_case(text)
    # number of ants 
    NUM_ANTS = 10
    # evaporation rate rho
    RHO = 0.1
    # pheromone and heuristic parameters
    BETA = 5
    # number of iter
    MAX_ITER = 500
    # Initial tau for phero matrix
    TAU0 = 1/((2*n+1)*nearest_neighbor_route_cost(n, k, c))
    # zeta for local phero
    ZETA = 0.1
    # Q0 for pseudorandom proportional
    Q0 = 0.9
    # Heuristic matrix
    C_NONZEROS = np.where(c == 0, 0.35, c) #0.05 is optional
    HEURISTIC = 1 / C_NONZEROS
    HEURISTIC_POWER_BETA = (HEURISTIC ** BETA)
    # Output
    route, cost = ant_colony_system()
    print('Best route:', route)
    print('Best cost:', cost)