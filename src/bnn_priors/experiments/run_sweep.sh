
runFunc () {
	export CUDA_VISIBLE_DEVICES=$1; python train_bnn.py with data=cifar10_stackaug model=googleresnet \
		weight_prior=gaussian inference=VerletSGLDReject warmup=45 burnin=0 skip=1 n_samples=300 \
		lr=0.01 momentum=0.994 weight_scale=1.4 cycles=60 batch_size=128 temperature=1.0 \
		save_samples=True progressbar=True log_dir=../results/conloss_tests \
		batchnorm=True likelihood=consistency
}

runFunc 1 1.0
