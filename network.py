# coding:utf-8
import logging

import torch
from torch import nn

from utils import zeros,\
			BaseNetwork, MyGRU, Storage, gumbel_max, flattenSequence

# pylint: disable=W0221
class Network(BaseNetwork):
	def __init__(self, param):
		super().__init__(param)

		self.genNetwork = GenNetwork(param)

	def forward(self, incoming):
		incoming.result = Storage()

		self.genNetwork.forward(incoming)

		incoming.result.loss = incoming.result.word_loss

		if torch.isnan(incoming.result.loss).detach().cpu().numpy() > 0:
			logging.info("Nan detected")
			logging.info(incoming.result)
			raise FloatingPointError("Nan detected")

	def detail_forward(self, incoming):
		incoming.result = Storage()

		self.genNetwork.detail_forward(incoming)

class GenNetwork(nn.Module):
	def __init__(self, param):
		super().__init__()
		self.args = args = param.args
		self.param = param

		self.embWeight = nn.Parameter(torch.Tensor(param.volatile.wordvec))
		self.GRULayer = MyGRU(args.embedding_size, args.dh_size, initpara=False)
		self.wLinearLayer = nn.Linear(args.dh_size, param.volatile.dm.vocab_size)
		self.lossCE = nn.CrossEntropyLoss(ignore_index=param.volatile.dm.unk_id)
		self.start_generate_id = 2

	def teacherForcing(self, inp, gen):
		embedding = inp.embedding
		length = inp.sent_length
		gen.h, _ = self.GRULayer.forward(embedding, length-1, h_init=inp.init_h, need_h=True)
		gen.w = self.wLinearLayer(gen.h)

	def freerun(self, inp, gen, mode='max'):
		batch_size = inp.batch_size
		dm = self.param.volatile.dm

		first_emb = inp.embWeight[dm.go_id].repeat(batch_size, 1)
		gen.w_pro = []
		gen.w_o = []
		gen.emb = []
		flag = zeros(batch_size).byte()
		EOSmet = []

		next_emb = first_emb
		gru_h = inp.init_h
		for _ in range(self.args.max_sen_length):
			now = next_emb
			gru_h = self.GRULayer.cell_forward(now, gru_h)
			w = self.wLinearLayer(gru_h)
			gen.w_pro.append(w)
			if mode == "max":
				w_o = torch.argmax(w[:, self.start_generate_id:], dim=1) + self.start_generate_id
				next_emb = inp.embWeight[w_o]
			elif mode == "gumbel":
				w_onehot, w_o = gumbel_max(w[:, self.start_generate_id:], 1, 1)
				w_o = w_o + self.start_generate_id
				next_emb = torch.sum(torch.unsqueeze(w_onehot, -1) * inp.embWeight[2:], 1)
			gen.w_o.append(w_o)
			gen.emb.append(next_emb)

			EOSmet.append(flag)
			flag = flag | (w_o == dm.eos_id)
			if torch.sum(flag).detach().cpu().numpy() == batch_size:
				break

		EOSmet = 1-torch.stack(EOSmet)
		gen.w_o = torch.stack(gen.w_o) * EOSmet.long()
		gen.emb = torch.stack(gen.emb) * EOSmet.float().unsqueeze(-1)
		gen.length = torch.sum(EOSmet, 0).detach().cpu().numpy()

	def forward(self, incoming):
		inp = Storage()
		inp.sent_length = incoming.data.sent_length
		inp.embedding = incoming.sent.embedding
		inp.init_h = incoming.conn.init_h

		incoming.gen = gen = Storage()
		self.teacherForcing(inp, gen)

		w_o_f = flattenSequence(gen.w, incoming.data.sent_length-1)
		data_f = flattenSequence(incoming.data.sent[1:], incoming.data.sent_length-1)
		incoming.result.word_loss = self.lossCE(w_o_f, data_f)
		incoming.result.perplexity = torch.exp(incoming.result.word_loss)

	def detail_forward(self, incoming):
		inp = Storage()
		batch_size = inp.batch_size = incoming.data.batch_size
		inp.init_h = incoming.conn.init_h

		incoming.gen = gen = Storage()
		self.freerun(inp, gen, 'gumbel')

		dm = self.param.volatile.dm
		w_o = gen.w_o.detach().cpu().numpy()
		incoming.result.gen_str = gen_str = \
				[" ".join(dm.index_to_sen(w_o[:, i].tolist())) for i in range(batch_size)]
		incoming.result.golden_str = golden_str = \
				[" ".join(dm.index_to_sen(incoming.data.sent[:, i].detach().cpu().numpy().tolist()))\
				for i in range(batch_size)]
		incoming.result.show_str = "\n".join(["gen: " + b + "\n" + \
				"golden: " + c + "\n" \
				for a, b, c in zip(gen_str, golden_str)])
