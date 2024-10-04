
# QP_Moire_OBC

`QP_Moire_OBC` is a Julia-based project aimed at studying quasi-periodic (QP) systems with moiré patterns under open boundary conditions (OBC). The project includes two main files: `Functions.jl` and `Run_and_eigs.jl`. These scripts provide tools to construct and analyze such systems, particularly focusing on obtaining eigenvalues and eigenstates.

## Repository Structure

- **Functions.jl**: Contains the core functions used for constructing the Hamiltonian, generating quasi-periodic potentials, and setting up open boundary conditions.
  
- **Run_and_eigs.jl**: Main script to run the simulations, compute the eigenvalues and eigenstates of the system, and perform other calculations. This file uses the functions defined in `Functions.jl`.

## Requirements

- **Julia**: This project requires Julia to run. You can download Julia [here](https://julialang.org/downloads/).
- **Dependencies**: The scripts may depend on some external Julia packages. You can install them via Julia's package manager (e.g., `Pkg.add()`).

## Installation

1. Clone this repository to your local machine:
    ```bash
    git clone https://github.com/yourusername/QP_Moire_OBC.git
    ```

2. Install any required dependencies:
    ```julia
    using Pkg
    Pkg.add("YourPackageList")
    ```

## Usage

1. Open the Julia REPL and navigate to the project directory:
    ```bash
    cd path/to/QP_Moire_OBC
    ```

2. Load the functions by including the `Functions.jl` file:
    ```julia
    include("Functions.jl")
    ```

3. Run the main script to perform simulations and compute eigenvalues/eigenstates:
    ```julia
    include("Run_and_eigs.jl")
    ```

   This will execute the main operations, leveraging the functions in `Functions.jl`.

## Example

Here’s a basic example of how to use the code:

```julia
include("Functions.jl")
include("Run_and_eigs.jl")

# Run simulations or compute eigenstates here
# Example function calls and outputs
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/yourusername/QP_Moire_OBC/issues).
