"""
beatrice_joint.py  -  CLI for CROSS-REGION JOINT prior training (Iteration 003).

Trains ONE shared annotation prior head jointly across ALL regions listed in a
manifest, then writes per-region pip.csv / credible_set.txt (via gen_cred) into
each region's own target dir. See scripts/joint_trainer.py for the model / the
joint objective and the equal-weight-per-region decision.

Manifest (--manifest): a tab-separated file, one region per line, columns:
    z <tab> LD <tab> annot <tab> target <tab> N
An optional header line 'z\tLD\tannot\ttarget\tN' is tolerated. `annot` is
REQUIRED for every region (a joint annotation prior needs annotations).

Prior head (--prior_head):
    'lassonet' (default) -> LassoNetPrior     (user idea #2)
    'linear'             -> LinearPrior        (user idea #1, flagship)

Example:
    python beatrice_joint.py --manifest scenario.tsv --N 50000 --prior_head linear \\
        --max_iter 1500 --n_caus 5 --sigma_sq 0.05 --sparse_concrete 50
"""
from absl import app
from absl import flags
import os

FLAGS = flags.FLAGS

flags.DEFINE_string('manifest', None, 'TSV of per-region z,LD,annot,target,N (one row per region).')
flags.DEFINE_string('prior_head', 'lassonet', "Shared prior head: 'lassonet' (idea #2) or 'linear' (idea #1).")
flags.DEFINE_string('save_prior_weights', '', 'Optional path to save the trained shared prior head (.pt).')
flags.DEFINE_integer('N', 50000, 'Default sample size when a manifest row omits N.', lower_bound=0)
flags.DEFINE_integer('MCMC_samples', 1, 'MC integration samples', lower_bound=1)
flags.DEFINE_integer('max_iter', 1500, 'Number of JOINT training iterations.', lower_bound=500)
flags.DEFINE_float('gamma', 0.1, 'Threshold to create the reduced space of binary vectors.')
flags.DEFINE_float('gamma_key', 0.2, 'Threshold for key variants.')
flags.DEFINE_float('gamma_coverage', 0.95, 'Coverage threshold for credible sets.')
flags.DEFINE_float('gamma_selection', 0.05, 'Selection-probability threshold within a credible set.')
flags.DEFINE_float('sigma_sq', 0.05, 'Variance of causal variants.')
flags.DEFINE_float('temp_lower_bound', 0.01, 'Extent of continuous relaxation.', lower_bound=0.005)
flags.DEFINE_integer('sparse_concrete', 50, 'Non-zero locations of the concrete vector per iteration.', lower_bound=1)
flags.DEFINE_integer('n_caus', 5, 'Prior number of causal variants.', lower_bound=1)
flags.DEFINE_boolean('allow_dup', False, 'Allow duplicate variants across credible sets.')
flags.DEFINE_list('neural_network', '10,200,10', 'Finemapping net hidden dims.')
flags.DEFINE_list('prior_neural_network', '10,200,10', 'LassoNet prior hidden dims.')
flags.DEFINE_float('prior_regularisation', 1.0, 'L2 regularisation on the prior (p_0).')
flags.DEFINE_float('lambda_l1', 0.01, 'L1 penalty for LassoNet skip connections (ignored by linear head).')
flags.DEFINE_float('hierarchy_M', 10.0, 'LassoNet hierarchy constraint multiplier.')


def _parse_manifest(path, default_N):
    regions = []
    with open(path) as f:
        for ln in f:
            ln = ln.rstrip('\n')
            if not ln.strip():
                continue
            parts = ln.split('\t')
            if parts[0].strip().lower() == 'z':      # header
                continue
            if len(parts) < 4:
                raise ValueError(f"manifest row needs >=4 cols (z,LD,annot,target[,N]): {ln!r}")
            z, LD, annot, target = parts[0], parts[1], parts[2], parts[3]
            N = int(parts[4]) if len(parts) >= 5 and parts[4].strip() else default_N
            if not annot or annot.lower() in ('none', 'na', ''):
                raise ValueError("beatrice_joint requires annotations for every region; "
                                 f"row has no annot: {ln!r}")
            regions.append(dict(z=z, LD=LD, annot=annot, target=target, N=N))
    return regions


def main(argv):
    import scripts.joint_trainer as joint_trainer
    if FLAGS.prior_head not in ('lassonet', 'linear'):
        raise ValueError(f"--prior_head must be 'lassonet' or 'linear', got {FLAGS.prior_head!r}")
    regions = _parse_manifest(FLAGS.manifest, FLAGS.N)
    for r in regions:
        os.makedirs(r['target'], exist_ok=True)

    options = {
        'regions': regions,
        'prior_head': FLAGS.prior_head,
        'save_prior_weights': FLAGS.save_prior_weights,
        'NN': [int(i) for i in FLAGS.neural_network],
        'prior_neural_network': [int(i) for i in FLAGS.prior_neural_network],
        'MCMC_samples': FLAGS.MCMC_samples,
        'max_iter': FLAGS.max_iter,
        'sigma_sq': FLAGS.sigma_sq,
        'temp_lower_bound': FLAGS.temp_lower_bound,
        'sparse_concrete': FLAGS.sparse_concrete,
        'gamma': FLAGS.gamma,
        'prior_regularisation': FLAGS.prior_regularisation,
        'lambda_l1': FLAGS.lambda_l1,
        'hierarchy_M': FLAGS.hierarchy_M,
        'n_causal': FLAGS.n_caus,
        'coverage_ths': FLAGS.gamma_coverage,
        'selection_prob': FLAGS.gamma_selection,
        'key_thres': FLAGS.gamma_key,
        'allow_duplicates': FLAGS.allow_dup,
    }
    joint_trainer.run_joint(options)


if __name__ == '__main__':
    flags.mark_flag_as_required('manifest')
    app.run(main)
