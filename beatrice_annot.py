from absl import app
from absl import flags
import os
import scripts.trainer_annot as trainer_annot
import torch
import random
FLAGS = flags.FLAGS

#  Define Flagg names for inputs.
flags.DEFINE_string('z', 'example_data/Simulation_data0.z', 'Location of Z Score')
flags.DEFINE_string('target', 'results', 'Location to store results.')
flags.DEFINE_string('LD', 'example_data/Simulation_data0.ld', 'Location of LD matrix')
flags.DEFINE_string('annot', None, 'Location of annotations') #'example_data/Simulation_data0.v'
flags.DEFINE_string('prior_location', '', 'Location where priors of the underlying probability map of bianry concrete distribution is stored.')
flags.DEFINE_integer('N', 5000, 'Number of subjects.', lower_bound=0)
flags.DEFINE_integer('MCMC_samples', 1, 'Number of random samples for MC integration', lower_bound=1)
flags.DEFINE_integer('max_iter', 2000, 'Number of training iterations.',lower_bound=500)
flags.DEFINE_boolean('plot_loss', True, 'Plot training losses.')
flags.DEFINE_boolean('allow_dup', False, 'Allow duplicate variants across credible sets')
flags.DEFINE_boolean('get_cred', True, 'Get Credible Sets')
flags.DEFINE_float('gamma', 0.1, 'Threshold to create the reduced space of binary vectors B^R.')
flags.DEFINE_float('gamma_key', 0.2, 'Threshold for key variants.')
flags.DEFINE_float('gamma_coverage', 0.95, 'Threshold for coverage.')
flags.DEFINE_float('gamma_selection', 0.05, 'Threshold for selection probability within a credible set.')
flags.DEFINE_float('sigma_sq', 0.05, 'Variance of causal variants')
flags.DEFINE_float('temp_lower_bound', 0.01, 'Extent of continuous relaxations', lower_bound=0.005)
flags.DEFINE_integer('sparse_concrete', 50, 'Number of non zero locatons of the concrete random vector at every iteration.', lower_bound=1)
flags.DEFINE_integer('n_caus', 20, 'Number of causal variants', lower_bound=1)
flags.DEFINE_list('true_loc', '', 'Index of true causal variants.')
flags.DEFINE_float('purity', 0, 'Purity')
flags.DEFINE_list('neural_network', '10,200,10', 'Size of 3-layer neural network. First layer has 10 nodes, second layer has 200 nodes and third has 10 nodes')
flags.DEFINE_list('prior_neural_network', '10,200,10', 'Size of 3-layer prior neural network. First layer has 10 nodes, second layer has 200 nodes and third has 10 nodes')
flags.DEFINE_float('prior_regularisation', 1.0, 'Size of L2 regularisation for prior neural network.')
flags.DEFINE_string('prior_weights', '',
    'Path to pre-trained prior network weights file (.pt) to initialize from. '
    'Use this to continue training from a previous run. The weights file must have '
    'matching n_annotations and hidden_dims (set via --prior_neural_network).')
flags.DEFINE_boolean('return_weights', False,
    'If True, saves the trained prior network weights to <target>/prior_network_weights.pt. '
    'These weights can be reused via --prior_weights in subsequent runs to iteratively '
    'train the prior network across multiple genomic regions.')

# LassoNet-specific arguments
flags.DEFINE_float('lambda_l1', 0.01,
    'L1 penalty weight for LassoNet skip connections. Controls feature sparsity. '
    'Higher values = more sparse feature selection. Typical range: 0.001-1.0')
flags.DEFINE_float('hierarchy_M', 10.0,
    'Hierarchy constraint multiplier for LassoNet. Controls how much hidden layer '
    'weights can exceed skip connection weights. ||W^(1)_j|| <= M * |theta_j|. '
    'Higher M = looser constraint. Typical range: 1.0-100.0')



def run_beatrice_annot(beatrice_args: dict):
    """Run Beatrice without CLI flags (safe for imports)."""

    defaults = {
        "neural_network": ['10', '200', '10'],
        "purity": 0.0,
        "n_caus": 20,
        "allow_dup": False,
        "true_loc": [],
        "MCMC_samples": 1,
        "sigma_sq": 0.05,
        "max_iter": 2000,
        "temp_lower_bound": 0.01,
        "prior_location": '',
        "plot_loss": True,
        "gamma_coverage": 0.95,
        "gamma_selection": 0.05,
        "gamma_key": 0.2,
        "get_cred": True,
        "sparse_concrete": 50,
        "gamma": 0.1,
        "annot": None, # defaults to None (not provided)
        "prior_neural_network": ['10', '200', '10'],
        "prior_regularisation": 1.0,
        "prior_weights": '',
        "return_weights": False,
        "lambda_l1": 0.01,
        "hierarchy_M": 10.0
        }

    # Merge user args with defaults
    args = {**defaults, **beatrice_args}

    # Add annot_given flag
    if args["annot"] is None:
        args["annot_given"] = False
    else:
        args["annot_given"] = True

    # File and folder checks
    for key in ["z", "LD", "annot", "target", "N"]:
        if key not in args:
            raise ValueError(f"Missing required argument: {key}")

    if not os.path.exists(args["z"]):
        raise FileNotFoundError(f"Location of Z doesn’t exist: {args['z']}")
    if not os.path.exists(args["LD"]):
        raise FileNotFoundError(f"Location of LD doesn’t exist: {args['LD']}")
    if args["annot_given"]:
        if not os.path.exists(args["annot"]):
            raise FileNotFoundError(f"Location of annotations doesn’t exist: {args['annot']}")
    else:
        print("No annotations provided, using default prior.")

    if not os.path.exists(args["target"]):
        os.makedirs(args["target"])

    torch.manual_seed(1)
    random.seed(1)

    options = {
        'NN': [int(i) for i in args["neural_network"]],
        'purity': args["purity"],
        'n_causal': args["n_caus"],
        'allow_duplicates': args["allow_dup"],
        'target': args["target"],
        'z': args["z"],
        'LD': args["LD"],
        'annot': args["annot"],
        'n_sub': args["N"],
        'loc_true': [int(i) for i in args["true_loc"]],
        'MCMC_samples': args["MCMC_samples"],
        'sigma_sq': args["sigma_sq"],
        'max_iter': args["max_iter"],
        'temp_lower_bound': args["temp_lower_bound"],
        'prior_location': args["prior_location"],
        'plot_loss': args["plot_loss"],
        'coverage_ths': args["gamma_coverage"],
        'selection_prob': args["gamma_selection"],
        'key_thres': args["gamma_key"],
        'get_cred': args["get_cred"],
        'sparsity_cl': args["sparse_concrete"],
        'gamma': args["gamma"],
        'annot_given': args["annot_given"],
        'prior_neural_network': [int(i) for i in args["prior_neural_network"]],
        'prior_regularisation': args["prior_regularisation"],
        'prior_weights': args["prior_weights"],
        'return_weights': args["return_weights"],
        'lambda_l1': args["lambda_l1"],
        'hierarchy_M': args["hierarchy_M"]
    }

    print(options)
    trainer_annot.main(options)


def main(argv):
    args = {
        "z": FLAGS.z,
        "LD": FLAGS.LD,
        "annot": FLAGS.annot,
        "N": FLAGS.N,
        "target": FLAGS.target,
        'prior_neural_network': FLAGS.prior_neural_network,
        'prior_regularisation': FLAGS.prior_regularisation,
        'MCMC_samples': FLAGS.MCMC_samples,
        'n_caus': FLAGS.n_caus,
        'neural_network': FLAGS.neural_network,
        'sparse_concrete': FLAGS.sparse_concrete,
        'prior_weights': FLAGS.prior_weights,
        'return_weights': FLAGS.return_weights,
        'lambda_l1': FLAGS.lambda_l1,
        'hierarchy_M': FLAGS.hierarchy_M,
        'sigma_sq': FLAGS.sigma_sq,
    }
    run_beatrice_annot(args)


if __name__ == '__main__':
    app.run(main)


"""
python beatrice_annot.py --z example_data/Simulation_data0.z --LD example_data/Simulation_data0.ld --annot example_data/Simulation_data0.v --N 5000  --target results
python beatrice_annot.py --z example_data/Simulation_data0.z --LD example_data/Simulation_data0.ld --N 5000 --MCMC_samples 50 --target results

python beatrice_annot.py --z alt_methods/Funmap_main/data/data.z --LD alt_methods/Funmap_main/data/ld.txt --N 50000 --sparse_concrete 10 --target results_funmap
python beatrice_annot.py --z alt_methods/Funmap_main/data/data.z --LD alt_methods/Funmap_main/data/ld.txt --annot alt_methods/Funmap_main/data/anno.txt --sparse_concrete 10 --target results_funmap

python beatrice_annot.py --z example_data/chr1_109000001_112000001.z --LD example_data/chr1_109000001_112000001.ld --N 459324 --MCMC_samples 50 --target resultslowcaus

python beatrice_annot.py --z example_data/chr1_109000001_112000001.z --LD example_data/chr1_109000001_112000001.ld --annot example_data/chr1_109000001_112000001.annot --N 459324 --prior_regularisation 10.0 --MCMC_samples 50 --target resultshighreg
python beatrice_annot.py --z example_data/chr1_55000001_58000001.z --LD example_data/chr1_55000001_58000001.ld --annot example_data/chr1_55000001_58000001.annot --N 459324 --sparse_concrete 10 --target results_annot_mcmc50

python beatrice_annot.py --z example_data/chr1_1_3000001.z --LD example_data/chr1_1_3000001.ld  --N 459324 --annot example_data/chr1_1_3000001.annot --sparse_concrete 10 --target results_annot_mcmc50
"""
