
# Constructive knapsack

This directory contains instances of the constructive variant of the 0/1
knapsack problem, i.e. the classic knapsack (maximise the sum of item costs
while maintaining the weighted sum of item counts below the knapsack capacity)
where each item is included either once or zero times.

### Format

All input tokens are natural numbers. One problem instance per line.

#### `_inst.dat` files

Each line of the format:

```plain
ID n M w_1 c_1 ... w_n c_n
```

where

| entry | description       |
|-------|-------------------|
|  ID   | unique identifier |
|   n   | number of items   |
|   M   | knapsack capacity |
|  w_i  | weight of item i  |
|  c_i  | cost of item i    |

#### `_sol.dat` files

Each line of the format:

```plain
ID n C a_1 ... a_n
```

where

| entry | description                           |
|-------|---------------------------------------|
|  ID   | unique identifier                     |
|   n   | number of items                       |
|   C   | total cost of the solution            |
|  a_i  | count of item i in the solution (0/1) |
