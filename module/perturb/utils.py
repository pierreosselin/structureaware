from .community import sample_perturbed_graphs_community
from .bernoulli import sample_perturbed_graphs_bernoulli

def load_perturbation(name):
    """Load the type of perturbation to use

    Args:
        name (str): Name of the perturbation

    Returns:
        [function]: Pertubation function
    """
    if name == "bernoulli":
        return sample_perturbed_graphs_bernoulli
    
    if name == "community":
        return sample_perturbed_graphs_community

