"""Small numerical primitives with explicit approximation and scope checks."""
import math
import numpy as np
import torch


def detrended_score(tokens, losses, degree: int = 2) -> dict:
    """Sample SD of residuals after a predeclared polynomial trend in token exposure.

    Normalize token coordinates before fitting. With only 16 observations, this
    mixes slow learning-curve mismatch and short fluctuations; report linear
    sensitivity too, rather than claiming a seed uncertainty or spectral test.
    """
    x = np.asarray(tokens, dtype=np.float64)
    y = np.asarray(losses, dtype=np.float64)
    if len(x) != len(y) or len(y) <= degree + 1 or not np.isfinite(y).all() or not np.all(np.diff(x) > 0):
        raise ValueError('Need increasing coordinates and enough finite observations')
    x = (x - x.mean()) / (x.max() - x.min())
    trend = np.polyval(np.polyfit(x, y, degree), x)
    residual = y - trend
    return {'degree': degree, 'residual_sample_std': float(np.std(residual, ddof=1)),
            'trend': trend.tolist(), 'residuals': residual.tolist(), 'observations': len(y)}


def directional_slice(parameters, direction: dict, loss_fn, alphas, epsilons) -> dict:
    """Evaluate theta + alpha * normalized recent update; always restore weights.

    alpha is an absolute whole-parameter L2 distance. Two finite-difference
    epsilons expose numerical sensitivity; this is one direction on one probe,
    not a maximum-curvature estimate or a visualization of all parameter space.
    """
    parameters = dict(parameters)
    if set(parameters) != set(direction):
        raise ValueError('Update direction must cover exactly the named parameters')
    norm = math.sqrt(sum(float(v.double().square().sum()) for v in direction.values()))
    if not math.isfinite(norm) or norm <= 0 or 0 not in alphas:
        raise ValueError('Need a finite nonzero direction and an alpha=0 baseline')
    base = {name: value.detach().clone() for name, value in parameters.items()}
    unit = {name: value.to(parameters[name].device) / norm for name, value in direction.items()}
    values = {}
    try:
        for alpha in alphas:
            with torch.no_grad():
                for name, parameter in parameters.items():
                    parameter.copy_(base[name] + alpha * unit[name])
            with torch.no_grad():
                values[float(alpha)] = float(loss_fn())
            if not math.isfinite(values[float(alpha)]):
                raise ValueError('Nonfinite directional loss')
    finally:
        with torch.no_grad():
            for name, parameter in parameters.items():
                parameter.copy_(base[name])
    curvatures = []
    for eps in epsilons:
        if eps <= 0 or eps not in values or -eps not in values:
            raise ValueError('Both signs of each positive epsilon must be evaluated')
        numerator = values[eps] + values[-eps] - 2 * values[0.]
        curvatures.append({'epsilon': eps, 'kappa': numerator / eps**2,
                           'second_difference': numerator})
    return {'direction_norm': norm, 'points': [{'alpha': alpha, 'loss': values[float(alpha)]} for alpha in alphas],
            'curvatures': curvatures}


def power_curvature(loss_fn, parameters, *, steps: int = 12, seed: int = 17) -> dict:
    """Use exact autograd Hessian-vector products and bounded power iteration.

    Power iteration targets the largest-MAGNITUDE eigenvalue. Retain its sign
    and residual. A negative dominant value must not be mislabeled lambda_max;
    a positive value is a provisional largest-positive estimate, not a proof of
    convergence or of the full-corpus Hessian. Rebuild the graph each iteration
    so forward activations do not accumulate. No dense Hessian is constructed.
    """
    parameters = tuple(parameters)
    if steps < 2:
        raise ValueError('Need at least two power iterations')
    generator = torch.Generator(device=parameters[0].device).manual_seed(seed)
    vector = [torch.randn(p.shape, dtype=p.dtype, device=p.device, generator=generator) for p in parameters]
    def dot(a, b):
        return sum((u*v).sum() for u,v in zip(a,b))
    def normalize(v):
        norm = torch.sqrt(dot(v,v))
        if not torch.isfinite(norm) or float(norm) == 0:
            raise ValueError('Hessian iteration reached a zero/nonfinite vector')
        return [t / norm for t in v]
    vector = normalize(vector)
    history = []
    for iteration in range(steps):
        loss = loss_fn()
        gradients = torch.autograd.grad(loss, parameters, create_graph=True)
        product = dot(gradients, vector)
        hvp = [v.detach() for v in torch.autograd.grad(product, parameters)]
        rayleigh = dot(vector, hvp)
        residual = torch.sqrt(sum((h-rayleigh*v).square().sum() for h,v in zip(hvp,vector)))
        relative = float(residual / torch.sqrt(dot(hvp,hvp)).clamp_min(1e-12))
        history.append({'iteration': iteration+1, 'rayleigh': float(rayleigh), 'relative_residual': relative})
        vector = normalize(hvp)
        del loss, gradients, product, hvp
    eigenvalue = history[-1]['rayleigh']
    return {'dominant_eigenvalue_estimate': eigenvalue,
            'lambda_max_estimate': eigenvalue if eigenvalue > 0 else None,
            'positive_dominant': eigenvalue > 0, 'converged_residual_5pct': relative < .05,
            'relative_residual': relative, 'iterations': history,
            'scope': 'Fixed training probe, data-loss Hessian; no optimizer, clipping or decay Jacobian.'}


def lanczos_curvature(loss_fn, parameters, *, steps: int = 12, seed: int = 17) -> dict:
    """Estimate the algebraically largest Hessian eigenvalue when |lambda_min| wins.

    Lanczos builds a small symmetric tridiagonal projection from HVPs, then
    selects its largest algebraic Ritz value. Full reorthogonalization prevents
    repeated eigenvectors from roundoff. Store at most `steps` parameter-size
    vectors, not a dense Hessian. One extra HVP measures the final Ritz residual.
    """
    parameters = tuple(parameters)
    generator = torch.Generator(device=parameters[0].device).manual_seed(seed)
    dimension = sum(p.numel() for p in parameters)
    q = torch.randn(dimension,device=parameters[0].device,dtype=parameters[0].dtype,generator=generator)
    q = q/q.norm()
    basis = []; diagonal = []; off_diagonal = []
    previous = None; beta_previous = 0.
    def hvp(vector):
        loss = loss_fn()
        gradients = torch.autograd.grad(loss,parameters,create_graph=True)
        start = 0; dot = loss.new_zeros(())
        for gradient in gradients:
            size = gradient.numel()
            dot = dot + (gradient.reshape(-1)*vector[start:start+size]).sum()
            start += size
        product = torch.autograd.grad(dot,parameters)
        return torch.cat([p.detach().reshape(-1) for p in product])
    for _ in range(min(steps,dimension)):
        basis.append(q)
        z = hvp(q)
        alpha = torch.dot(q,z)
        z = z-alpha*q
        if previous is not None:
            z = z-beta_previous*previous
        # Two passes are inexpensive compared with an HVP and reduce FP32 loss
        # of orthogonality when the positive and negative spectra differ greatly.
        for _ in range(2):
            for vector in basis:
                z = z-torch.dot(vector,z)*vector
        beta = z.norm()
        diagonal.append(float(alpha))
        if not torch.isfinite(beta):
            raise ValueError('Nonfinite Lanczos vector')
        if float(beta)<1e-7:
            break
        off_diagonal.append(float(beta))
        previous,q = q,z/beta
        beta_previous = beta
    projection = np.diag(diagonal)
    if len(diagonal)>1:
        projection += np.diag(off_diagonal[:len(diagonal)-1],1)+np.diag(off_diagonal[:len(diagonal)-1],-1)
    eigenvalues,eigenvectors = np.linalg.eigh(projection)
    coefficients = eigenvectors[:,-1]
    ritz = sum(float(c)*v for c,v in zip(coefficients,basis))
    ritz = ritz/ritz.norm()
    product = hvp(ritz)
    estimate = float(torch.dot(ritz,product))
    residual = float((product-estimate*ritz).norm()/product.norm().clamp_min(1e-12))
    return {'lambda_max_estimate':estimate,'relative_residual':residual,
            'converged_residual_5pct':residual<.05,'steps':len(diagonal),
            'ritz_spectrum':eigenvalues.tolist(),'method':'Symmetric Lanczos, full reorthogonalization, one residual HVP'}
