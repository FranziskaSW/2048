301790515
777934738
*****
Comments:
--------------------
Evaluation function
--------------------
We came up with 3 features for evaluation.

(1) smoothness - we define smoothness as the sum of "derivations" of rows
                 and columns. we "derive" the board by subtracting an element from its neighbor.
                 smaller smoothness means neighbors elements are close to each other, so that
                 there are more chances that they will be combined, so we evaluate boards with small smoothness.
(2) empty tiles - a board with more empty tiles allows more flexibility of movement and less likely to get blocking.
                  we simply sum the number of empty tiles.
(3) merges - we will prefer a state in which there are more potentials merges.

After some experimentation with those 3 features, the linear combination:
10000 - smoothness + 100 * empty + 20 * merges
led to quiet good results.