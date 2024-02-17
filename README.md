## Problem description
#### CBUS Large-Size
There are n passengers 1, 2, …, n. The passenger i want to travel from point i to point i + n (i = 1,2,…,n). There is a bus located at point 0 and has k places for transporting the passengers (it means at any time, there are at most k passengers on the bus). You are given the distance matrix c in which c(i,j) is the traveling distance from point i to point j (i, j = 0,1,…, 2n). Compute the shortest route for the bus, serving n passengers and coming back to point 0.

#### Input

Line 1 contains n and k (1≤n≤1000,1≤k≤50)
Line i+1 (i=1,2,…,2n+1) contains the (i−1)th line of the matrix c (rows and columns are indexed from 0,1,2,..,2n).

#### Output

Line 1: write the value n
Line 2: Write the sequence of points (pickup and drop-off) of passengers (separated by a SPACE character)


## Ant Colony Optimization
Ant colony optimization algorithms have been applied to many combinatorial optimization problems, including Capacitated Vehicle Routing Problem. They have an advantage over simulated annealing and genetic algorithm approaches of similar problems when the graph may change dynamically; the ant colony algorithm can be run continuously and adapt to changes in real time.

Ant Colony System (ACS) is one of the most used and often
the best-performing of ACO algorithms.

Pseudo-code of the ACS algorithm:
```
procedure ACSMetaheuristicStatic
    Set parameters, initialize pheromone trails
    while (termination condition not met) do
        ConstructAntsSolutions % update local pheromones after each ant's solution
        UpdateGlobalPheromones
    end
end
```
#### Running the Code

The code uploaded to this GitHub Repository is a Python implementation of the Ant Colony System algorithm for the CBUS Large-size problem.

You need to have Python installed on your machine. The project uses Numpy module, to install the module you can use a package manager like pip:
```
pip install numpy
```

The main script of the algorithm is CBUS_ACS.py It can be run from the command line using the following command:
```
python CBUS_ACS.py
```

Enter the test case (from the given test cases) you want to solve for:
```
n_k.txt
```
(n is the number of passengers, k is the bus capacity)

## Authors
Bui Nguyen Khai


## Resources
[Ant Colony Optimization](https://web2.qatar.cmu.edu/~gdicaro/15382/additional/aco-book.pdf) book by Marco Dorigo and Thomas Stützle.
