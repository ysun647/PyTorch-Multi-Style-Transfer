echo 'Hard-Edge-Painting'

sudo python msgnet_transfer.py \
		--mode multi \
			--src /home/ys3031/PyTorch-Multi-Style-Transfer/experiments/dataset/train2014 \
				--style /data/dataset-of-4/train/Hard-Edge-Painting \
					--tgt /data/dataset-of-4/train-after-10000/Hard-Edge-Painting \
						--model_path /home/ys3031/PyTorch-Multi-Style-Transfer/experiments/models/model-4class/Final_epoch_7_Mon_Dec__3_15:36:11_2018_1.0_5.0.model \
							--num 5000

echo 'Impressionism'

sudo python msgnet_transfer.py \
		--mode multi \
			--src /home/ys3031/PyTorch-Multi-Style-Transfer/experiments/dataset/train2014 \
				--style /data/dataset-of-4/train/Impressionism \
					--tgt /data/dataset-of-4/train-after-10000/Impressionism \
						--model_path /home/ys3031/PyTorch-Multi-Style-Transfer/experiments/models/model-4class/Final_epoch_7_Mon_Dec__3_15:36:11_2018_1.0_5.0.model \
							--num 5000

echo 'Ink-and-wash-painting'

sudo python msgnet_transfer.py \
		--mode multi \
			--src /home/ys3031/PyTorch-Multi-Style-Transfer/experiments/dataset/train2014 \
				--style /data/dataset-of-4/train/Ink-and-wash-painting \
					--tgt /data/dataset-of-4/train-after-10000/Ink-and-wash-painting \
						--model_path /home/ys3031/PyTorch-Multi-Style-Transfer/experiments/models/model-4class/Final_epoch_7_Mon_Dec__3_15:36:11_2018_1.0_5.0.model \
							--num 5000

echo 'Neo-Expressionism'

sudo python msgnet_transfer.py \
		--mode multi \
			--src /home/ys3031/PyTorch-Multi-Style-Transfer/experiments/dataset/train2014 \
				--style /data/dataset-of-4/train/Neo-Expressionism \
					--tgt /data/dataset-of-4/train-after-10000/Neo-Expressionism \
						--model_path /home/ys3031/PyTorch-Multi-Style-Transfer/experiments/models/model-4class/Final_epoch_7_Mon_Dec__3_15:36:11_2018_1.0_5.0.model \
							--num 5000
