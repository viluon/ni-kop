
# Decision knapsack

This directory contains instances of the decision variant of the 0/1 knapsack
problem, i.e. decide whether a solution to the [constructive 0/1
knapsack](../constructive/README.md) exists with total cost of at least B.

### Format

All input tokens are natural numbers. One problem instance per line.
Each line of the format:

```plain
ID n M B w_1 c_1 ... w_n c_n
```

where

| entry | description       |
|-------|-------------------|
|  ID   | unique identifier |
|   n   | number of items   |
|   M   | knapsack capacity |
|   B   | minimal cost      |
|  w_i  | weight of item i  |
|  c_i  | cost of item i    |
