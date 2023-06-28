DATA_DIR:=/home/ninnart/datasets/super-resolution-datasets
SCALE   :=4

.PHONY: train
train:
	python -u main.py \
	  --gpus 0 \
	  --n_GPUs 1 \
	  --model EDSR_e2fif \
	  --save edsr_e2fif \
	  --res_scale 1 \
	  --binary_mode binary \
	  --dir_data $(DATA_DIR) \
	  --epochs 300 \
	  --decay 200 \
	  --lr 2e-4 \
	  --data_test Set5+Set14+Urban100 \
	  --scale $(SCALE) \
	  --res_scale 1 \
	  --n_resblocks 16 \
	  --n_feats 64 \
	  --res_scale 1 \
	  --n_colors 1

.PHONY: test
test:
	python -u main.py \
	  --gpus 0 \
	  --n_GPUs 1 \
	  --model EDSR_e2fif \
	  --load edsr_e2fif \
	  --resume -2 \
	  --test_only \
	  --binary_mode binary \
	  --dir_data $(DATA_DIR) \
	  --epochs 300 \
	  --decay 200 \
	  --lr 2e-4 \
	  --data_test Set5+Set14+Urban100+BSDS100+Manga109 \
	  --scale $(SCALE) \
	  --n_resblocks 16 \
	  --n_feats 64 \
	  --res_scale 1 \
	  --n_colors 1

.PHONY: clean
clean:
	find -iname __pycache__ | xargs rm -rf
	find -iname tensorboard_files | xargs rm -rf
	rm -rf experiment

