
seed=$(( $1 / 6 ))
varInd=$(( $1 / 5 ))
# varInd=$1

temps=(0.005 0.01 0.05 0.1 0.5 1.0)
# scale=(0.005 0.01 0.05 0.1 0.5 1.0 1.5 2.0 5.0 10.0)
echo $seed ${temps[$varInd]}

# for seed in 0 1 2 3 4;
# do
python train_bnn.py with data=cifar10_augmented model=googleresnet weight_prior=gaussian \
    inference=VerletSGLDReject warmup=45 burnin=0 skip=1 n_samples=300 lr=0.01 \
    temperature=1.0 label_smoothing=0.0 save_samples=True progressbar=True batchnorm=True \
    momentum=0.994 softmax_temp=${temps[$varInd]} weight_scale=1.0 bias_scale=1.0 cycles=60 batch_size=128 \
    seed=$seed likelihood=True log_dir=/scratch/${USER}/bnn_priors/softmax_temp_flips/
        
#    sleep 5s
# done
