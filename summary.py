import transformers
import datasets
import os
import sys
import logging
import numpy as np
from dataclasses import dataclass, field
from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader
from typing import Optional
import torch
import argparse
from tqdm.auto import tqdm
from transformers import AdamW
from transformers import get_scheduler	
from transformers import (
	AutoConfig,
	AutoModelForSeq2SeqLM,
	AutoModelForCausalLM,
	AutoTokenizer,
	DataCollatorForSeq2Seq,
	HfArgumentParser,
	Seq2SeqTrainer,
	Seq2SeqTrainingArguments,
	TrainingArguments,
	Trainer,
	GPT2LMHeadModel,
	default_data_collator,
	set_seed,
)

from torch.nn import CrossEntropyLoss

from transformers.trainer_utils import get_last_checkpoint

@dataclass
class ModelArguments:
	"""
	Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
	"""

	model_name_or_path: str = field(
		default="uer/gpt2-chinese-cluecorpussmall", metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
	)
	config_name: Optional[str] = field(
		default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
	)
	tokenizer_name: Optional[str] = field(
		default="bert-base-chinese", metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
	)
	cache_dir: Optional[str] = field(
		default=None,
		metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
	)
	use_fast_tokenizer: bool = field(
		default=True,
		metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
	)
	model_revision: str = field(
		default="main",
		metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
	)
	use_auth_token: bool = field(
		default=False,
		metadata={
			"help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
			"with private models)."
		},
	)


@dataclass
class DataTrainingArguments:
	"""
	Arguments pertaining to what data we are going to input our model for training and eval.
	"""

	dataset_name: Optional[str] = field(
		default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
	)
	dataset_config_name: Optional[str] = field(
		default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
	)
	text_column: Optional[str] = field(
		default=None,
		metadata={"help": "The name of the column in the datasets containing the full texts (for summarization)."},
	)
	summary_column: Optional[str] = field(
		default=None,
		metadata={"help": "The name of the column in the datasets containing the summaries (for summarization)."},
	)
	train_file: Optional[str] = field(
		default="data/train.csv", metadata={"help": "The input training data file (a jsonlines or csv file)."}
	)
	validation_file: Optional[str] = field(
		default="data/eval.csv",
		metadata={
			"help": "An optional input evaluation data file to evaluate the metrics (rouge) on "
			"(a jsonlines or csv file)."
		},
	)
	test_file: Optional[str] = field(
		default=None,
		metadata={
			"help": "An optional input test data file to evaluate the metrics (rouge) on " "(a jsonlines or csv file)."
		},
	)
	overwrite_cache: bool = field(
		default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
	)
	preprocessing_num_workers: Optional[int] = field(
		default=None,
		metadata={"help": "The number of processes to use for the preprocessing."},
	)
	max_source_length: Optional[int] = field(
		default=1024,
		metadata={
			"help": "The maximum total input sequence length after tokenization. Sequences longer "
			"than this will be truncated, sequences shorter will be padded."
		},
	)
	max_target_length: Optional[int] = field(
		default= 1024,
		metadata={
			"help": "The maximum total sequence length for target text after tokenization. Sequences longer "
			"than this will be truncated, sequences shorter will be padded."
		},
	)
	val_max_target_length: Optional[int] = field(
		default=None,
		metadata={
			"help": "The maximum total sequence length for validation target text after tokenization. Sequences longer "
			"than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
			"This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
			"during ``evaluate`` and ``predict``."
		},
	)
	pad_to_max_length: bool = field(
		default=True,
		metadata={
			"help": "Whether to pad all samples to model maximum sentence length. "
			"If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
			"efficient on GPU but very bad for TPU."
		},
	)
	max_train_samples: Optional[int] = field(
		default=None,
		metadata={
			"help": "For debugging purposes or quicker training, truncate the number of training examples to this "
			"value if set."
		},
	)
	max_eval_samples: Optional[int] = field(
		default=None,
		metadata={
			"help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
			"value if set."
		},
	)
	max_predict_samples: Optional[int] = field(
		default=None,
		metadata={
			"help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
			"value if set."
		},
	)
	num_beams: Optional[int] = field(
		default=None,
		metadata={
			"help": "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
			"which is used during ``evaluate`` and ``predict``."
		},
	)
	ignore_pad_token_for_loss: bool = field(
		default=True,
		metadata={
			"help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
		},
	)
	source_prefix: Optional[str] = field(
		default=None, metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
	)

	def __post_init__(self):
		if self.dataset_name is None and self.train_file is None and self.validation_file is None:
			raise ValueError("Need either a dataset name or a training/validation file.")
		else:
			if self.train_file is not None:
				extension = self.train_file.split(".")[-1]
				assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
			if self.validation_file is not None:
				extension = self.validation_file.split(".")[-1]
				assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
		if self.val_max_target_length is None:
			self.val_max_target_length = self.max_target_length

def main():
	parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
	if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
		# If we pass only one argument to the script and it's the path to a json file,
		# let's parse it to get our arguments.
		model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
	else:
		model_args, data_args, training_args = parser.parse_args_into_dataclasses()
		
	print(training_args)
	
	##查看是否有先前訓練過的
	last_checkpoint = None
	if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
		last_checkpoint = get_last_checkpoint(training_args.output_dir)
		if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
			raise ValueError(
				f"Output directory ({training_args.output_dir}) already exists and is not empty. "
				"Use --overwrite_output_dir to overcome."
				)
		elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
			logger.info(
				f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
				"the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
				)
	#設置隨機種子
	set_seed(training_args.seed)
			
	if data_args.dataset_name is not None:
		raw_datasets = load_dataset(
			data_args.dataset_name, data_args.dataset_config_name, cache_dir=model_args.cache_dir
		)	
	else:
		data_files = {}
		if data_args.train_file is not None:
			data_files["train"] = data_args.train_file
			extension = data_args.train_file.split(".")[-1]
		if data_args.validation_file is not None:
			data_files["validation"] = data_args.validation_file
			extension = data_args.validation_file.split(".")[-1]
		if data_args.test_file is not None:
			data_files["test"] = data_args.test_file
			extension = data_args.test_file.split(".")[-1]
		raw_datasets = load_dataset(extension, data_files=data_files, cache_dir=model_args.cache_dir)



	# Load pretrained model and tokenizer
	config = AutoConfig.from_pretrained(
		model_args.config_name if model_args.config_name else model_args.model_name_or_path,
		cache_dir=model_args.cache_dir,
		revision=model_args.model_revision,
		use_auth_token=True if model_args.use_auth_token else None,
	)
	
	tokenizer = AutoTokenizer.from_pretrained(
		model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
		cache_dir=model_args.cache_dir,
		use_fast=model_args.use_fast_tokenizer,
		revision=model_args.model_revision,
		use_auth_token=True if model_args.use_auth_token else None,
	)
	
	#AutoModelForCausalLM 裡面有包含到 GPT2LMheadmodel
	model = AutoModelForCausalLM.from_pretrained(
		model_args.model_name_or_path,
		from_tf=bool(".ckpt" in model_args.model_name_or_path),
		config=config,
		cache_dir=model_args.cache_dir,
		revision=model_args.model_revision,
		use_auth_token=True if model_args.use_auth_token else None,
	)
	
	
	
	# Temporarily set max_target_length for training.
	max_target_length = data_args.max_target_length
	padding = "max_length" if data_args.pad_to_max_length else False

		# Preprocessing the datasets.


	def preprocess_function(examples):
		inputs = examples["article"]
		targets = examples["summarization"]
		#inputs = [prefix + inp for inp in inputs]
		model_inputs = tokenizer(inputs, targets, max_length=data_args.max_source_length, padding=padding, truncation=True)
		#print(f"model_input: {model_inputs}")
		
		label = [[word if model_inputs["token_type_ids"][batch_id][id] ==1 else 0 for id, word in enumerate(model_inputs["input_ids"][batch_id])] for batch_id, token_type_ids in enumerate(model_inputs["token_type_ids"])]
		#labels["input_ids"] = [word if model_inputs["token_type_ids"][idx] == 1 else 0 for idx, word in enumerate(model_inputs["input_ids"])]
		# # Setup the tokenizer for targets
		# with tokenizer.as_target_tokenizer():
		# 	labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)

		# # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
		# # padding in the loss.

		# if padding == "max_length" and data_args.ignore_pad_token_for_loss:
		# 	labels["input_ids"] = [
		# 		[(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
		# 	]
	
		model_inputs["labels"] = label
		return model_inputs

	if training_args.do_train:
		#為了讓處理後的dataset去掉中文文本
		column_names = raw_datasets["train"].column_names
	elif training_args.do_eval:
		column_names = raw_datasets["validation"].column_names
	elif training_args.do_predict:
		column_names = raw_datasets["test"].column_names
	else:
		#至少一定要做train, eval, predict
		print('error!!')

	if training_args.do_train:
		if "train" not in raw_datasets:
			raise ValueError("--do_train requires a train dataset")
		train_dataset = raw_datasets["train"]
		if data_args.max_train_samples is not None:
			train_dataset = train_dataset.select(range(data_args.max_train_samples))
		train_dataset = train_dataset.map(
			preprocess_function,
			batched=True,
			num_proc=data_args.preprocessing_num_workers,
			remove_columns=column_names,
			load_from_cache_file=not data_args.overwrite_cache,
			desc="Running tokenizer on train dataset",
		)

	if training_args.do_eval:
		max_target_length = data_args.val_max_target_length
		if "validation" not in raw_datasets:
			raise ValueError("--do_eval requires a validation dataset")
		eval_dataset = raw_datasets["validation"]
		if data_args.max_eval_samples is not None:
			eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))
		eval_dataset = eval_dataset.map(
			preprocess_function,
			batched=True,
			num_proc=data_args.preprocessing_num_workers,
			remove_columns=column_names,
			load_from_cache_file=not data_args.overwrite_cache,
			desc="Running tokenizer on validation dataset",
		)

	if training_args.do_predict:
		max_target_length = data_args.val_max_target_length
		if "test" not in raw_datasets:
			raise ValueError("--do_predict requires a test dataset")
		predict_dataset = raw_datasets["test"]
		if data_args.max_predict_samples is not None:
			predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))
		predict_dataset = predict_dataset.map(
			preprocess_function,
			batched=True,
			num_proc=data_args.preprocessing_num_workers,
			remove_columns=column_names,
			load_from_cache_file=not data_args.overwrite_cache,
			desc="Running tokenizer on prediction dataset",
		)
	
	
	

	def calculate_loss_and_accuracy(outputs, labels, device):
		logits = outputs[0]  
		shift_logits = logits[..., :-1, :].contiguous()
		shift_labels = labels[..., 1:].contiguous()
		loss_fct = CrossEntropyLoss(ignore_index=0, reduction='sum')
		loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),shift_labels.view(-1))
		
		#num = shift_labels.ne(0).long().sum().item()  
		#loss = loss / num
		return loss
	
	
	
	def eval_epoch(model, device, eval_dataloader, epoch):
		total_loss = 0.0
		model.eval()
		for batch in eval_dataloader:
			with torch.no_grad():
				input_ids = batch["input_ids"].to(device)
				token_type_ids = batch["token_type_ids"].to(device)
				labels = batch["labels"].to(device)
			
				outputs = model.forward(input_ids=input_ids)
				loss = calculate_loss_and_accuracy(outputs, labels, device)
				loss = loss.mean()
				
				total_loss += loss.item()
				
		epoch_mean_loss = total_loss / len(eval_dataloader)		
		return epoch_mean_loss
	
	
	
	def train_epoch(model, device, train_dataloader, optimizer, epoch, lr_scheduler):
		total_loss = 0
		##梯度積累
		gradient_accumulation_steps = 2
		max_grad_norm = 1
		
		model.train()
		for step, batch in enumerate(train_dataloader):
			input_ids = batch["input_ids"].to(device)
			token_type_ids = batch["token_type_ids"].to(device)
			labels = batch["labels"].to(device)
			
			outputs = model.forward(input_ids=input_ids)
			
			loss = calculate_loss_and_accuracy(outputs, labels, device)
			
			loss = loss.mean()
			total_loss += loss.item()
			if gradient_accumulation_steps > 1:
				loss = loss/ gradient_accumulation_steps

			loss.backward()
			
			torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
			
			if (step +1) % gradient_accumulation_steps == 0:
				optimizer.step()
				lr_scheduler.step()
				optimizer.zero_grad()
				
			progress_bar.update(1)
			
			
		epoch_mean_loss = total_loss / len(train_dataloader)
		print(f"epoch: {epoch}, loss:{epoch_mean_loss}")
		
		##儲存模型
		output_dir = os.path.join(training_args.output_dir, "checkpoint-{}".format(epoch))		
		model_to_save = model.module if hasattr(model, "module") else model
		model_to_save.save_pretrained(output_dir)
		
		return epoch_mean_loss
		
	
	
	train_dataset.set_format(type='torch')
	eval_dataset.set_format(type='torch')
	#print(train_dataset[0])
	train_dataloader = DataLoader(train_dataset, batch_size=4)
	eval_dataloader = DataLoader(eval_dataset, batch_size=4)
	##print(next(iter(eval_dataloader)))
	
	optimizer = AdamW(model.parameters(), lr=5e-5)
	num_epochs = 9
	num_training_steps = num_epochs * len(train_dataloader)
	lr_scheduler = get_scheduler(
		"linear",
		optimizer=optimizer,
		num_warmup_steps=0,
		num_training_steps=num_training_steps
	)
	device = torch.device("cuda")
	#print(torch.cuda.get_device_name(0))

	##device = torch.device("cpu")
	model.to(device)
	progress_bar = tqdm(range(num_training_steps))
	
	best_val_loss = 10000
	train_losses, validate_losses = [], []
	print("begin training...")

	for epoch in range(num_epochs):
		train_loss = train_epoch(model, device, train_dataloader, optimizer, epoch, lr_scheduler)
		train_losses.append(train_loss)
		
		eval_loss = eval_epoch(model, device, eval_dataloader, epoch)
		validate_losses.append(eval_loss)
		 
		if eval_loss < best_val_loss:
			print(f"epoch: {epoch} is currently best model")
			best_val_loss = eval_loss
			output_dir = os.path.join(training_args.output_dir, "best_model")		
			model_to_save = model.module if hasattr(model, "module") else model
			model_to_save.save_pretrained(output_dir)
	


if __name__ == '__main__':
	main()
