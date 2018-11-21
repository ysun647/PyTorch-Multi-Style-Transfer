from msgnet_models import *
from msgnet_util import *
import argparse

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

	output_img = Image.open(target_img)
	output_img.show()


if __name__ == "__main__":
	# sample usage: $ python msgnet_transfer.py --mode single --src candy.jpg --style venice-boat.jpg --tgt sym_output.jpg
	parser = argparse.ArgumentParser()
	parser.add_argument("--mode")
	parser.add_argument("--src")
	parser.add_argument("--style")
	parser.add_argument("--tgt")
	args = parser.parse_args()
	if args.mode == "single":
		transfer_single_image(source_img=args.src, style_img=args.style, target_img=args.tgt)
	elif args.mode == "multi":
		print("multi mode to be completed!")
	else:
		raise ValueError("unknown mode: %s" % args.mode)