# coding:utf-8
import logging
import time
import os

import torch
from torch import nn, optim
import numpy as np

from cotk.metric import LanguageGenerationRecorder
from utils import Storage, cuda, BaseModel, SummaryHelper, get_mean, storage_to_list, \
	CheckpointManager
from network import Network

class LM(BaseModel):
	def __init__(self, param):
		args = param.args
		net = Network(param)
		self.optimizer = optim.Adam(net.get_parameters_by_name(), lr=args.lr)
		optimizerList = {"optimizer": self.optimizer}
		checkpoint_manager = CheckpointManager(args.name, args.model_dir, \
						args.checkpoint_steps, args.checkpoint_max_to_keep, "min")
		super().__init__(param, net, optimizerList, checkpoint_manager)

		self.create_summary()

	def create_summary(self):
		args = self.param.args
		self.summaryHelper = SummaryHelper("%s/%s_%s" % \
				(args.log_dir, args.name, time.strftime("%H%M%S", time.localtime())), \
				args)
		self.trainSummary = self.summaryHelper.addGroup(\
			scalar=["loss", "word_loss", "perplexity"],\
			prefix="train")

		scalarlist = ["word_loss", "perplexity_avg_on_batch"]
		tensorlist = []
		textlist = []
		emblist = []
		for i in self.args.show_sample:
			textlist.append("show_str%d" % i)
		self.devSummary = self.summaryHelper.addGroup(\
			scalar=scalarlist,\
			tensor=tensorlist,\
			text=textlist,\
			embedding=emblist,\
			prefix="dev")
		self.testSummary = self.summaryHelper.addGroup(\
			scalar=scalarlist,\
			tensor=tensorlist,\
			text=textlist,\
			embedding=emblist,\
			prefix="test")

	def _preprocess_batch(self, data):
		incoming = Storage()
		incoming.data = data = Storage(data)
		data.batch_size = data.sent.shape[0]
		data.sent = cuda(torch.LongTensor(data.sent.transpose(1, 0))) # length * batch_size
		return incoming

	def get_next_batch(self, dm, key, restart=True, needhash=False):
		data = dm.get_next_batch(key, needhash=needhash)
		if data is None:
			if restart:
				dm.restart(key)
				return self.get_next_batch(dm, key, False, needhash=needhash)
			else:
				return None
		return self._preprocess_batch(data)

	def get_select_batch(self, dm, key, i):
		data = dm.get_batch(key, i)
		if data is None:
			return None
		return self._preprocess_batch(data)

	def train(self, batch_num):
		args = self.param.args
		dm = self.param.volatile.dm
		datakey = 'train'

		for _ in range(batch_num):
			self.now_batch += 1
			incoming = self.get_next_batch(dm, datakey)
			incoming.args = Storage()

			self.optimizer.zero_grad()
			self.net.forward(incoming)

			loss = incoming.result.loss
			self.trainSummary(self.now_batch, storage_to_list(incoming.result))
			logging.info("batch %d : gen loss=%f", self.now_batch, loss.detach().cpu().numpy())

			loss.backward()
			nn.utils.clip_grad_norm_(self.net.parameters(), args.grad_clip)
			self.optimizer.step()

	def evaluate(self, key):
		args = self.param.args
		dm = self.param.volatile.dm

		dm.restart(key, args.batch_size, shuffle=False)

		result_arr = []
		while True:
			incoming = self.get_next_batch(dm, key, restart=False)
			if incoming is None:
				break
			incoming.args = Storage()

			with torch.no_grad():
				self.net.forward(incoming)
			result_arr.append(incoming.result)

		detail_arr = Storage()
		for i in args.show_sample:
			index = [i * args.batch_size + j for j in range(args.batch_size)]
			incoming = self.get_select_batch(dm, key, index)
			incoming.args = Storage()
			with torch.no_grad():
				self.net.detail_forward(incoming)
			detail_arr["show_str%d" % i] = incoming.result.show_str

		detail_arr.update({key:get_mean(result_arr, key) for key in result_arr[0]})
		detail_arr.perplexity_avg_on_batch = np.exp(detail_arr.word_loss)
		return detail_arr

	def train_process(self):
		args = self.param.args
		dm = self.param.volatile.dm

		while self.now_epoch < args.epochs:
			self.now_epoch += 1
			self.updateOtherWeights()

			dm.restart('train', args.batch_size)
			self.net.train()
			self.train(args.batch_per_epoch)

			self.net.eval()
			devloss_detail = self.evaluate("dev")
			self.devSummary(self.now_batch, devloss_detail)
			logging.info("epoch %d, evaluate dev", self.now_epoch)

			testloss_detail = self.evaluate("test")
			self.testSummary(self.now_batch, testloss_detail)
			logging.info("epoch %d, evaluate test", self.now_epoch)

			self.save_checkpoint(value=devloss_detail.loss.tolist())

	def test(self, key):
		args = self.param.args
		dm = self.param.volatile.dm

		dm.restart(key, args.batch_size, shuffle=False)
		metric1 = dm.get_teacher_forcing_metric()

		while True:
			incoming = self.get_next_batch(dm, key, restart=False, needhash=True)
			if incoming is None:
				break
			incoming.args = Storage()
			with torch.no_grad():
				self.net.forward(incoming)
				gen_prob = nn.functional.log_softmax(incoming.gen.w, -1)
			data = incoming.data
			data.sent = incoming.data.sent_allvocabs
			data.sent_length = incoming.data.sent_length
			data.gen_log_prob = gen_prob.detach().cpu().numpy().transpose(1, 0, 2)
			metric1.forward(data)
		res = metric1.close()

		dm.restart(key, args.batch_size, shuffle=False)
		#metric2 = dm.get_inference_metric()
		metric2 = LanguageGenerationRecorder(dm, gen_key="gen")
		while True:
			incoming = self.get_next_batch(dm, key, restart=False)
			if incoming is None:
				break
			incoming.args = Storage()
			with torch.no_grad():
				self.net.detail_forward(incoming)
			data = incoming.data
			data.sent = incoming.data.sent_allvocabs
			data.gen = incoming.gen.w_o.detach().cpu().numpy().transpose(1, 0)
			metric2.forward(data)
		res.update(metric2.close())

		if not os.path.exists(args.out_dir):
			os.makedirs(args.out_dir)
		filename = args.out_dir + "/%s_%s.txt" % (args.name, key)

		with open(filename, 'w') as f:
			logging.info("%s Test Result:", key)
			for key, value in res.items():
				if isinstance(value, float) or isinstance(value, bytes):
					logging.info("\t{}:\t{}".format(key, value))
					f.write("{}:\t{}\n".format(key, value))
			for i in range(len(res['gen'])):
				f.write("gen:\t%s\n" % " ".join(res['gen'][i]))
			f.flush()
		logging.info("result output to %s.", filename)

	def test_process(self):
		logging.info("Test Start.")
		self.net.eval()
		self.test("dev")
		self.test("test")
		logging.info("Test Finish.")
