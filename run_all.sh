sudo python inception.py --epochs 80 \
 --traditional_aug 0 \
 --neural_aug 1 \
 --log_save_dir ./t_0_n_1_pretrained/log \
 --model_save_dir ./t_0_n_1_pretrained/model \
 --feature_extract 1 \
 --save_step 1

sudo python inception.py --epochs 4800 \
 --traditional_aug 1 \
 --neural_aug 0 \
 --log_save_dir ./t_1_n_0_pretrained/log \
 --model_save_dir ./t_1_n_0_pretrained/model \
 --save_step 160 \
 --feature_extract 1

sudo python inception.py --epochs 4800 \
 --traditional_aug 0 \
 --neural_aug 0 \
 --log_save_dir ./t_0_n_0_pretrained/log \
 --model_save_dir ./t_0_n_0_pretrained/model \
 --save_step 160 \
 --feature_extract 1

# =========

sudo python inception.py --epochs 80 \
 --traditional_aug 0 \
 --neural_aug 1 \
 --log_save_dir ./t_0_n_1_pretrained_tuneall/log \
 --model_save_dir ./t_0_n_1_pretrained_tuneall/model \
 --feature_extract 0 \
 --save_step 1

sudo python inception.py --epochs 80 \
 --traditional_aug 1 \
 --neural_aug 1 \
 --log_save_dir ./t_1_n_1_pretrained_tuneall/log \
 --model_save_dir ./t_1_n_1_pretrained_tuneall/model \
 --feature_extract 0 \
 --save_step 1

sudo python inception.py --epochs 4800 \
 --traditional_aug 1 \
 --neural_aug 0 \
 --log_save_dir ./t_1_n_0_pretrained_tuneall/log \
 --model_save_dir ./t_1_n_0_pretrained_tuneall/model \
 --save_step 160 \
 --feature_extract 0

sudo python inception.py --epochs 4800 \
 --traditional_aug 0 \
 --neural_aug 0 \
 --log_save_dir ./t_0_n_0_pretrained_tuneall/log \
 --model_save_dir ./t_0_n_0_pretrained_tuneall/model \
 --feature_extract 0 \
 --save_step 160





