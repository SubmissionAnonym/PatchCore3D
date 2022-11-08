python bin/run_patchcore.py --log_project project --log_group patchcore \
/path/to/results_test \
patch_core -b densenet -le features.denseblock3 --faiss_num_workers 6 \
--pretrain_embed_dimension 1024  --target_embed_dimension 128 --anomaly_scorer_num_nn 1 --patchsize 3 \
--pretrained_model_path /path/to/DenseNet121_BHB-10K_yAwareContrastive.pth \
sampler -p 0.01 approx_greedy_coreset \
dataset --test_size 0.2 /path/to/ixi/ /path/to/brats2021