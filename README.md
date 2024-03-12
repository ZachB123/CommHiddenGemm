# Comm Hidden Gemm

## Installation
- pip install -r requirements.txt

## GEMM1d
- NOTE: The number of processes must be greater than 1 and divide m,n,k equally
- This program executes C = AB + C where A (m x k), B (k x n), C (m x n)
- mpirun -n <num_processes> python driver.py arguments

### Arguments
- -m 
- Specify the M dimension of arrays A and C.
- Type: Integer
- Default: None

<br>

- -k
- Specify the K dimension of array A and the N dimension of array B.
- Type: Integer
- Default: None

<br>

- -n, 
- Specify the N dimension of arrays B and C.
- Type: Integer
- Default: None

<br>

- -s, --strategy
- Specify the algorithm for computation.
- Type: String
- Options: 'allgather_A_col', 'allgather_A_row', 'allgather_B_col', 'allgather_B_row', 'reducescatter_C_col', 'reducescatter_C_row'
- Default: None

<br>

- If no dimension arguments are provided, default dimensions (m=16, k=8, n=4) are used.
- If no strategy is provided, the best option based on dimensions is automatically selected defaulting to allgather_A_col.
- If a strategy is provided, it must be one of the preset options.
- All dimension arguments (m, k, n) must be specified together.