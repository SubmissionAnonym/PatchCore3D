import os
import argparse

from utils import *
import patchcore3d.metrics
import patchcore3d.utils


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='eval')
    parser.add_argument('--pred', type=str, default='/mnt/pred', metavar='N',
                        help='Predictions path')
    args = parser.parse_args()

    test_imgs = os.listdir(os.path.join(args.pred, 'sample'))
    labels_gt = [0 if uid.startswith('IXI') else 1 for uid in test_imgs]
    
    scores = [float(load(os.path.join(args.pred, f'sample/{fname}'))) for fname in test_imgs]

    instance_results = patchcore3d.metrics.compute_imagewise_retrieval_metrics(
        scores, labels_gt
    )

    print('auroc', instance_results['auroc'])
    print('auprc', instance_results['auprc'])
        