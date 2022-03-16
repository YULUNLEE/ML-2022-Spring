import os
import random
from torch.utils.data import Dataset
import json
import csv
from pathlib import Path
from tqdm.notebook import tqdm
import torch
from torch.utils.data import DataLoader
import Hw4


class InferenceDataset(Dataset):
	def __init__(self, data_dir):
		testdata_path = Path(data_dir) / "testdata.json"
		metadata = json.load(testdata_path.open())
		self.data_dir = data_dir
		self.data = metadata["utterances"]
		# self.segment_len = 128
	def __len__(self):
		return len(self.data)

	def __getitem__(self, index):
		utterance = self.data[index]
		feat_path = utterance["feature_path"]
		mel = torch.load(os.path.join(self.data_dir, feat_path))
		# Segmemt mel-spectrogram into "segment_len" frames.
		# if len(mel) > self.segment_len:
		# 	# Randomly get the starting point of the segment.
		# 	start = random.randint(0, len(mel) - self.segment_len)
		# 	# Get a segment with "segment_len" frames.
		# 	mel = torch.FloatTensor(mel[start:start + self.segment_len])
		# else:
		# 	mel = torch.FloatTensor(mel)
		# print(mel.shape)
		return feat_path, mel


def inference_collate_batch(batch):
	"""Collate a batch of data."""
	feat_paths, mels = zip(*batch)

	return feat_paths, torch.stack(mels)


def parse_args():
	"""arguments"""
	config = {
		"data_dir": "./Dataset",
		"model_path1": "./model356_1_92.ckpt",
		# "model_path2": "./model3.ckpt",
		# "model_path3": "./model4.ckpt",
		"output_path": "./output_model356_1_92.csv",
	}

	return config


def main(
	data_dir,
	model_path1,
	# model_path2,
	# model_path3,
	output_path,
):
	"""Main function."""
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"[Info]: Use {device} now!")

	mapping_path = Path(data_dir) / "mapping.json"
	mapping = json.load(mapping_path.open())

	dataset = InferenceDataset(data_dir)
	dataloader = DataLoader(
		dataset,
		batch_size=1,
		shuffle=False,
		drop_last=False,
		num_workers=8,
		collate_fn=inference_collate_batch,
	)
	print(f"[Info]: Finish loading data!",flush = True)


	speaker_num = len(mapping["id2speaker"])
	# model = Hw4.Conformer(input_dim=40, num_heads=2, ffn_dim=512, num_layers=5, depthwise_conv_kernel_size=11, dropout=0.2).to(device)
	model1 = Hw4.Classifier1().to(device)
	model1.load_state_dict(torch.load(model_path1))
	# model2 = Hw4.Classifier2().to(device)
	# model2.load_state_dict(torch.load(model_path2))
	# model3 = Hw4.Classifier3().to(device)
	# model3.load_state_dict(torch.load(model_path3))

	model1.eval()
	# model2.eval()
	# model3.eval()

	print(f"[Info]: Finish creating model!",flush = True)

	results = [["Id", "Category"]]
	for feat_paths, mels in tqdm(dataloader,disable=True):
		with torch.no_grad():
			# lengths = torch.randint(200, 201, (32,)).to(device)
			# lengths = torch.randint(1, 201, (31,)).to(device)
			# lengths = torch.cat((lengths, torch.tensor([200]).to(device)), dim=0)
			# # 打亂
			# idx = torch.randperm(lengths.nelement())
			# lengths = lengths.view(-1)[idx].view(lengths.size())
			mels = mels.to(device)
			# print(mels.shape)
			outs = model1(mels)
			# outs2 = model2(mels)
			# outs3 = model3(mels)
			# outs = (outs1+outs2+outs3)/3.0
			# print(outs)
			preds = outs.argmax(1).cpu().numpy()
			for feat_path, pred in zip(feat_paths, preds):
				results.append([feat_path, mapping["id2speaker"][str(pred)]])

	with open(output_path, 'w', newline='') as csvfile:
		writer = csv.writer(csvfile)
		writer.writerows(results)


if __name__ == "__main__":
	main(**parse_args())