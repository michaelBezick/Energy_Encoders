import polytensor
import torch

vector = torch.randn(64)

num_vars = 64

num_per_degree = [num_vars, num_vars * (num_vars - 1) // 2]
sample_fn = lambda: torch.randn(1, device='cuda')
terms = polytensor.generators.coeffPUBORandomSampler(
        n=num_vars, num_terms=num_per_degree, sample_fn=sample_fn
        )

terms = polytensor.generators.denseFromSparse(terms, num_vars)
energy_fn = polytensor.polynomial.DensePolynomial(terms)

print(f"Energy before saving {energy_fn(vector)}")

data = energy_fn.coefficients
print(f"Data before saving {data}")
torch.save(data, "coefficients.pt")

data = torch.load("coefficients.pt")
print(f"Data after saving {data}")

energy_fn.coefficients = data
print(f"Energy after saving {energy_fn(vector)}")

terms2 = polytensor.generators.coeffPUBORandomSampler(
        n=num_vars, num_terms=num_per_degree, sample_fn=sample_fn
        )
terms2 = polytensor.generators.denseFromSparse(terms2, num_vars)
energy_fn_2 = polytensor.polynomial.DensePolynomial(terms2)
energy_fn.coefficients = energy_fn_2.coefficients
print(f"Energy after new random terms {energy_fn(vector)}")
