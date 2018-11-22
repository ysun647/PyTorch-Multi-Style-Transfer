from msgnet_models import *
from msgnet_util import *
import argparse
import os

MODEL_PATH = '21styles.model'

def transfer_single_image(source_img, style_img, target_img):
	content_image = tensor_load_rgbimage(source_img, size=512, keep_asp=True).unsqueeze(0)
	style = tensor_load_rgbimage(style_img, size=512).unsqueeze(0)
	style = preprocess_batch(style)
	model_dict = torch.load(MODEL_PATH)
	model_dict_clone = model_dict.copy()  # We can't mutate while iterating

	for key, value in model_dict_clone.items():
		if key.endswith(('running_mean', 'running_var')):
			del model_dict[key]

	style_model = Net(ngf=128)
	style_model.load_state_dict(model_dict, False)

	style_v = Variable(style)
	content_image = Variable(preprocess_batch(content_image))
	style_model.setTarget(style_v)
	output = style_model(content_image)
	tensor_save_bgrimage(output.data[0], target_img, False)


def transfer_multi_image(source_dir, style_img, target_dir, style):
	if not os.path.exists(target_dir):
		os.makedirs(target_dir)
	for img in os.listdir(source_dir):
		if img.endswith(".jpg"):
			transfer_single_image(os.path.join(source_dir, img), style_img, os.path.join(target_dir, style+"-"+img))



if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--mode")
	parser.add_argument("--src")
	parser.add_argument("--style_pic")
	parser.add_argument("--tgt")
	parser.add_argument("--style")
	args = parser.parse_args()
	if args.mode == "single":
		# sample usage: $ python msgnet_transfer.py --mode single --src candy.jpg --style_pic venice-boat.jpg --tgt sym_output.jpg
		transfer_single_image(source_img=args.src, style_img=args.style_pic, target_img=args.tgt)
	elif args.mode == "multi":
		'''
		python msgnet_transfer.py \
		  --mode multi \
		  --src /Users/yiming/dev/data/pic/realism \
		  --style_pic /Users/yiming/dev/data/pic/post-imp/post-imp.jpg \
		  --tgt /Users/yiming/dev/data/pic/realism-postimp-result \
		  --style postimp
		'''
		transfer_multi_image(source_dir=args.src, style_img=args.style_pic, target_dir=args.tgt, style=args.style)
	else:
		raise ValueError("unknown mode: %s" % args.mode)