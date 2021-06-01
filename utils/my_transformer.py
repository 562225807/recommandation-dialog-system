from transformers import GPT2PreTrainedModel, GPT2Model
from transformers.modeling_utils import SequenceSummary
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch
import numpy as np
import logging
from torch.nn import functional as F

logger = logging.getLogger(__name__)


class GPT2DoubleHeadsModel(GPT2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.device = torch.device("cuda")
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.multiple_choice_head = SequenceSummary(config)

        self.LSTM_hidden_size = self.config.n_embd//2
        self.kn_encoder = nn.LSTM(input_size=config.n_embd, hidden_size=self.LSTM_hidden_size, dropout=0.1, bidirectional=True)
        self.encoder = nn.LSTM(input_size=config.n_embd, hidden_size=self.LSTM_hidden_size, dropout=0.1, bidirectional=True)

        self.xy_x = nn.Linear(self.LSTM_hidden_size * 4, self.LSTM_hidden_size * 2)

        self.gen = nn.Sequential(nn.Dropout(0.1),
                                     nn.Linear(config.n_embd, 1, bias=False),
                                     nn.Sigmoid())

        # self.bow_fc = nn.Sequential(nn.Dropout(0.1),
        #                              nn.Linear(config.n_embd, config.n_embd),
        #                              nn.Tanh())
        #
        # self.bow_lm = nn.Sequential(nn.Dropout(0.1),
        #                             nn.Linear(config.n_embd, 29916, bias=False),
        #                             nn.Softmax())

        self.hid_dim = config.n_embd
        self.n_heads = 12
        self.w_q = nn.Linear(self.hid_dim, self.hid_dim)
        self.w_k = nn.Linear(self.hid_dim, self.hid_dim)
        self.w_v = nn.Linear(self.hid_dim, self.hid_dim)
        self.fc = nn.Linear(self.hid_dim, self.hid_dim)
        self.do = nn.Dropout(0.1)
        self.scale = torch.sqrt(torch.FloatTensor([self.hid_dim // self.n_heads])).to(self.device)

        self.my_embs = nn.Embedding(29916, config.n_embd)
        self.my_head = nn.Linear(config.n_embd, 29916, bias=False)
        self.my_head.weight = self.my_embs.weight

        self.init_weights()

        #self.input_pre_embs()

    def input_pre_embs(self):
        vocab = []
        with open("vocab.txt", encoding='utf-8') as f:
            for line in f.readlines():
                emb = line.split('\t')[-1].split(' ')
                emb = list(map(float, emb))
                vocab.append(emb)
        self.my_embs = nn.Embedding.from_pretrained(torch.FloatTensor(vocab))
        self.my_head.weight = self.my_embs.weight

    def multi_attention(self, query, key, value, mask=None):

        bsz = query.shape[0]
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)

        Q = Q.view(bsz, -1, self.n_heads, self.hid_dim //
                   self.n_heads).permute(0, 2, 1, 3)
        K = K.view(bsz, -1, self.n_heads, self.hid_dim //
                   self.n_heads).permute(0, 2, 1, 3)
        V = V.view(bsz, -1, self.n_heads, self.hid_dim //
                   self.n_heads).permute(0, 2, 1, 3)

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        # 如果没有mask，就生成一个
        if mask is not None:
            mask = mask.view_as(energy)
            energy = energy.masked_fill(mask == 0, -1e10)

        # 然后对Q,K相乘的结果计算softmax加上dropout，这是计算scaled dot product attention的第二步：
        attention = self.do(torch.softmax(energy, dim=-1))

        # 第三步，attention结果与V相乘
        x = torch.matmul(attention, V)

        # 最后将多头排列好，就是multi-head attention的结果了
        x = x.permute(0, 2, 1, 3).contiguous()

        x = x.view(bsz, -1, self.n_heads * (self.hid_dim // self.n_heads))
        x = self.fc(x)

        return x

    def dot_attention(self, query, memory, mask=None):
        """ dot attention """
        attn = torch.matmul(query, memory.transpose(-2, -1))
        if mask is not None:
            attn += mask.float() * -1000000000

        weight = torch.softmax(attn, -1)
        weight_memory = torch.matmul(weight, memory)

        return weight_memory, weight

    def get_output_embeddings(self):
        return self.lm_head

    def forward(
        self,
        input_ids=None,
        past=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        mc_token_ids=None,
        lm_labels=None,
        mc_labels=None,
        use_posterior=False,
        use_bow=False
    ):
        r"""
        mc_token_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, num_choices)`, `optional`, default to index of the last token of the input)
            Index of the classification token in each input sequence.
            Selected in the range ``[0, input_ids.size(-1) - 1[``.
        lm_labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`)
            Labels for language modeling.
            Note that the labels **are shifted** inside the model, i.e. you can set ``lm_labels = input_ids``
            Indices are selected in ``[-1, 0, ..., config.vocab_size]``
            All labels set to ``-100`` are ignored (masked), the loss is only
            computed for labels in ``[0, ..., config.vocab_size]``
        mc_labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size)`, `optional`, defaults to :obj:`None`)
            Labels for computing the multiple choice classification loss.
            Indices should be in ``[0, ..., num_choices]`` where `num_choices` is the size of the second dimension
            of the input tensors. (see `input_ids` above)

    Return:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.GPT2Config`) and inputs:
        lm_loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when ``lm_labels`` is provided):
            Language modeling loss.
        mc_loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`multiple_choice_labels` is provided):
            Multiple choice classification loss.
        lm_prediction_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, num_choices, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        mc_prediction_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, num_choices)`):
            Prediction scores of the multiple choice classification head (scores for each choice before SoftMax).
        past (:obj:`List[torch.FloatTensor]` of length :obj:`config.n_layers` with each tensor of shape :obj:`(2, batch_size, num_heads, sequence_length, embed_size_per_head)`):
            Contains pre-computed hidden-states (key and values in the attention blocks).
            Can be used (see `past` input) to speed up sequential decoding. The token ids which have their past given to this model
            should not be passed as input ids as they have already been computed.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

        import torch
        from transformers import GPT2Tokenizer, GPT2DoubleHeadsModel

        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        model = GPT2DoubleHeadsModel.from_pretrained('gpt2')

        # Add a [CLS] to the vocabulary (we should train it also!)
        tokenizer.add_special_tokens({'cls_token': '[CLS]'})
        model.resize_token_embeddings(len(tokenizer))  # Update the model embeddings with the new vocabulary size
        print(tokenizer.cls_token_id, len(tokenizer))  # The newly token the last token of the vocabulary

        choices = ["Hello, my dog is cute [CLS]", "Hello, my cat is cute [CLS]"]
        encoded_choices = [tokenizer.encode(s) for s in choices]
        cls_token_location = [tokens.index(tokenizer.cls_token_id) for tokens in encoded_choices]

        input_ids = torch.tensor(encoded_choices).unsqueeze(0)  # Batch size: 1, number of choices: 2
        mc_token_ids = torch.tensor([cls_token_location])  # Batch size: 1

        outputs = model(input_ids, mc_token_ids=mc_token_ids)
        lm_prediction_scores, mc_prediction_scores = outputs[:2]

        """
        bts = input_ids[0].shape[0]
        kn_num = input_ids[2].shape[1]

        his_ids = input_ids[0]
        res_ids = input_ids[1]
        kn_ids = input_ids[2].view(bts * kn_num, -1)
        voc_ids = input_ids[3]
        choose = None
        if len(input_ids) == 5:
            choose = input_ids[-1]

        his_mask = his_ids != 0
        his_mask = his_mask.long()
        his_len = torch.sum(his_mask, -1)
        res_mask = res_ids != 0
        res_mask = res_mask.long()
        res_len = torch.sum(res_mask, -1)
        kn_mask = kn_ids != 0
        kn_mask = kn_mask.long()
        kn_len = torch.sum(kn_mask, -1)
        kn_num_mask = kn_len == 0
        kn_num_mask = kn_num_mask.long()
        kn_len[kn_len == 0] = 1

        his_embs = self.my_embs(his_ids)
        res_embs = self.my_embs(res_ids)
        kn_embs = self.my_embs(kn_ids)

        his_tmp = nn.utils.rnn.pack_padded_sequence(his_embs, his_len, batch_first=True, enforce_sorted=False)
        kn_tmp = nn.utils.rnn.pack_padded_sequence(kn_embs, kn_len, batch_first=True, enforce_sorted=False)
        his_en, his_last = self.encoder(his_tmp)
        kn_en, kn_last = self.kn_encoder(kn_tmp)

        his_en, _ = nn.utils.rnn.pad_packed_sequence(his_en, batch_first=True, padding_value=0)
        his_last = his_last[0].transpose(0, 1).reshape(bts, 1, -1)
        kn_last = kn_last[0].transpose(0, 1).reshape(bts, kn_num, -1)

        kn_num_mask = kn_num_mask.view(bts, 1, kn_num)
        x_weight_kn, x_kn_att = self.dot_attention(his_last, kn_last, mask=kn_num_mask)
        #x_weight_kn = self.multi_attention(his_last, kn_last, kn_last, mask=kn_num_mask)

        knowledge = x_weight_kn
        if use_posterior:
            res_tmp = nn.utils.rnn.pack_padded_sequence(res_embs, res_len, batch_first=True, enforce_sorted=False)
            res_en, res_last = self.encoder(res_tmp)
            res_last = res_last[0].transpose(0, 1).reshape(bts, 1, -1)
            res_last = torch.cat([res_last, his_last], -1)

            kn_num_mask = kn_num_mask.view(bts, 1, kn_num)
            xy_weight_kn, xy_kn_att = self.dot_attention(self.xy_x(res_last), kn_last, mask=kn_num_mask)
            #xy_weight_kn = self.multi_attention(self.xy_x(res_last), kn_last, kn_last, mask=kn_num_mask)
            knowledge = xy_weight_kn

        input_ids = torch.cat([his_ids, res_ids], 1)
        input_embs = self.my_embs(input_ids)
        input_embs = input_embs + knowledge

        token_type_ids = torch.cat([token_type_ids, 2 * torch.ones([bts, res_ids.shape[1]]).long().to(self.device)], 1)
        attention_mask = torch.cat([his_mask, res_mask], 1)

        transformer_outputs = self.transformer(
            past=past,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=input_embs,
        )

        hidden_states = transformer_outputs[0]

        lm_logits = self.my_head(hidden_states)
        gen_p = self.gen(hidden_states).view(bts * hidden_states.shape[1], -1)
        tmp_gen_p = gen_p.repeat(1, voc_ids.shape[-1])
        tmp_lm = lm_logits.view(bts * lm_logits.shape[1], -1)
        tmp_voc = voc_ids.view(bts, 1, voc_ids.shape[-1]).repeat(1, lm_logits.shape[1], 1).view(bts * lm_logits.shape[1], -1)
        kn_p = torch.zeros(tmp_lm.shape).to(self.device).scatter_(1, tmp_voc, 1 - tmp_gen_p)
        kn_p[:, 0] = 0.0
        lm_logits = gen_p * tmp_lm + kn_p * tmp_lm
        lm_logits = lm_logits.view(bts, hidden_states.shape[1], -1)

        mc_logits = self.multiple_choice_head(hidden_states, mc_token_ids).squeeze(-1)

        outputs = (lm_logits, mc_logits) + transformer_outputs[1:]
        if use_bow:
            bow_logits = self.bow_fc(knowledge)
            bow_logits = self.bow_lm(bow_logits).repeat(1, hidden_states.shape[1], 1)
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(bow_logits.view(-1, bow_logits.size(-1)), lm_labels.view(-1))
            outputs = (loss,) + outputs

        if use_posterior:
            prior_attn = torch.log(x_kn_att + 1e-10).view(bts, -1)
            posterior_att = torch.FloatTensor(xy_kn_att.tolist()).to(self.device).view(bts, -1)
            kl_loss = posterior_att * (torch.log(posterior_att + 1e-10) - prior_attn)
            kl_loss = torch.mean(kl_loss)
            outputs = (kl_loss,) + outputs

        if mc_labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(mc_logits.view(-1, mc_logits.size(-1)), mc_labels.view(-1))
            outputs = (loss,) + outputs
        if lm_labels is not None:
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = lm_labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            outputs = (loss,) + outputs
        if choose is not None:
            # gen_p = gen_p.view(bts, -1, gen_p.shape[-1])
            # shift_probabily = gen_p[:, :-1].contiguous().view(-1)
            # shift_labels = choose[..., 1:].contiguous().view(-1)
            # res_mask = res_mask.view(-1)
            # have = res_mask == 1
            # tmp = shift_labels != -100
            # shift_probabily = shift_probabily[tmp][have]
            # shift_labels = shift_labels[tmp][have]
            # loss = torch.sum(- torch.log(shift_probabily) * shift_labels +
            #                  - torch.log(1.0 - shift_probabily) * (1 - shift_labels))/torch.sum(res_mask, -1)
            loss = torch.Tensor([0])
            outputs = (loss,) + outputs

        return outputs, x_kn_att  # (lm loss), (mc loss), lm logits, mc logits, presents, (all hidden_states), (attentions)

    @torch.no_grad()
    def generate(
            self,
            input_ids=None,
            max_length=None,
            do_sample=True,
            num_beams=None,
            temperature=None,
            top_k=None,
            top_p=None,
            repetition_penalty=None,
            bos_token_id=None,
            pad_token_id=None,
            eos_token_ids=None,
            length_penalty=None,
            num_return_sequences=None,
            knowledges=None,
            kn_vocs=None,
            segments=None,
    ):
        # We cannot generate if the model does not have a LM head
        if self.get_output_embeddings() is None:
            raise AttributeError(
                "You tried to generate sequences with a model that does not have a LM Head."
                "Please use another model class (e.g. `OpenAIGPTLMHeadModel`, `XLNetLMHeadModel`, `GPT2LMHeadModel`, `CTRLLMHeadModel`, `T5WithLMHeadModel`, `TransfoXLLMHeadModel`)"
            )

        max_length = max_length if max_length is not None else self.config.max_length
        do_sample = do_sample if do_sample is not None else self.config.do_sample
        num_beams = num_beams if num_beams is not None else self.config.num_beams
        temperature = temperature if temperature is not None else self.config.temperature
        top_k = top_k if top_k is not None else self.config.top_k
        top_p = top_p if top_p is not None else self.config.top_p
        repetition_penalty = repetition_penalty if repetition_penalty is not None else self.config.repetition_penalty
        bos_token_id = bos_token_id if bos_token_id is not None else self.config.bos_token_id
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_ids = eos_token_ids if eos_token_ids is not None else self.config.eos_token_ids
        length_penalty = length_penalty if length_penalty is not None else self.config.length_penalty
        num_return_sequences = (
            num_return_sequences if num_return_sequences is not None else self.config.num_return_sequences
        )

        if input_ids is not None:
            batch_size = input_ids.shape[0]  # overriden by the input batch_size
        else:
            batch_size = 1
        if isinstance(eos_token_ids, int):
            eos_token_ids = [eos_token_ids]

        assert isinstance(max_length, int) and max_length > 0, "`max_length` should be a strictely positive integer."
        assert isinstance(do_sample, bool), "`do_sample` should be a boolean."
        assert isinstance(num_beams, int) and num_beams > 0, "`num_beams` should be a strictely positive integer."
        assert temperature > 0, "`temperature` should be strictely positive."
        assert isinstance(top_k, int) and top_k >= 0, "`top_k` should be a positive integer."
        assert 0 <= top_p <= 1, "`top_p` should be between 0 and 1."
        assert repetition_penalty >= 1.0, "`repetition_penalty` should be >= 1."
        assert input_ids is not None or (
                isinstance(bos_token_id, int) and bos_token_id >= 0
        ), "If input_ids is not defined, `bos_token_id` should be a positive integer."
        assert pad_token_id is None or (
                isinstance(pad_token_id, int) and (pad_token_id >= 0)
        ), "`pad_token_id` should be a positive integer."
        assert (eos_token_ids is None) or (
                isinstance(eos_token_ids, (list, tuple)) and ((isinstance(e, int) and e >= 0) for e in eos_token_ids)
        ), "`eos_token_ids` should be a positive integer or a list/tuple of positive integers."
        assert length_penalty > 0, "`length_penalty` should be strictely positive."
        assert (
                isinstance(num_return_sequences, int) and num_return_sequences > 0
        ), "`num_return_sequences` should be a strictely positive integer."

        if input_ids is None:
            assert isinstance(bos_token_id, int) and bos_token_id >= 0, (
                "you should either supply a context to complete as `input_ids` input "
                "or a `bos_token_id` (integer >= 0) as a first token to start the generation."
            )
            input_ids = torch.full(
                (batch_size, 1), bos_token_id, dtype=torch.long, device=next(self.parameters()).device
            )
        else:
            assert input_ids.dim() == 2, "Input prompt should be of shape (batch_size, sequence length)."

        if pad_token_id is None and eos_token_ids is not None:
            logger.warning(
                "Setting `pad_token_id` to {} (first `eos_token_id`) to generate sequence".format(eos_token_ids[0])
            )
            pad_token_id = eos_token_ids[0]

        # current position and vocab size
        cur_len = input_ids.shape[1]
        vocab_size = self.config.vocab_size

        if num_return_sequences != 1:
            # Expand input to num return sequences
            input_ids = input_ids.unsqueeze(1).expand(batch_size, num_return_sequences, cur_len)
            input_ids = input_ids.contiguous().view(
                batch_size * num_return_sequences, cur_len
            )  # (batch_size * num_return_sequences, cur_len)
            effective_batch_size = batch_size * num_return_sequences
        else:
            effective_batch_size = batch_size

        if num_beams > 1:
            output = self._generate_beam_search(
                input_ids,
                cur_len,
                max_length,
                do_sample,
                temperature,
                top_k,
                top_p,
                repetition_penalty,
                pad_token_id,
                eos_token_ids,
                effective_batch_size,
                length_penalty,
                num_beams,
                vocab_size,
            )
        else:
            output = self._generate_no_beam_search(
                input_ids,
                cur_len,
                max_length,
                do_sample,
                temperature,
                top_k,
                top_p,
                repetition_penalty,
                pad_token_id,
                eos_token_ids,
                effective_batch_size,
                knowledges=knowledges,
                kn_vocs=kn_vocs,
                segments=segments,
            )

        return output

    def _generate_no_beam_search(
            self,
            input_ids,
            cur_len,
            max_length,
            do_sample,
            temperature,
            top_k,
            top_p,
            repetition_penalty,
            pad_token_id,
            eos_token_ids,
            batch_size,
            knowledges=None,
            kn_vocs=None,
            segments=None,
    ):
        """ Generate sequences for each example without beam search (num_beams == 1).
            All returned sequence are generated independantly.
        """
        # current position / max lengths / length of generated sentences / unfinished sentences
        unfinished_sents = input_ids.new(batch_size).fill_(1)
        sent_lengths = input_ids.new(batch_size).fill_(max_length - input_ids.shape[1])

        past = None
        device = torch.device("cuda")
        responses = torch.LongTensor([[]] * input_ids.shape[0]).to(device)

        while cur_len < max_length:
            model_inputs = {"input_ids": (input_ids, responses, knowledges, kn_vocs), "token_type_ids": segments}
            outputs, x_att = self(**model_inputs)
            next_token_logits = outputs[0][:, -1, :]

            # if model has past, then set the past variable to speed up decoding
            if self._do_output_past(outputs):
                past = outputs[1]

            if do_sample:
                # Temperature (higher temperature => more likely to sample low probability tokens)
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature
                # Top-p/top-k filtering
                next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
                # Sample
                next_token = torch.multinomial(F.softmax(next_token_logits, dim=-1), num_samples=1).squeeze(1)
            else:
                # Greedy decoding
                next_token = torch.argmax(next_token_logits, dim=-1)

            # update generations and finished sentences
            if eos_token_ids is not None:
                # pad finished sentences if eos_token_ids exist
                tokens_to_add = next_token * unfinished_sents + (pad_token_id) * (1 - unfinished_sents)
            else:
                tokens_to_add = next_token

            responses = torch.cat([responses, tokens_to_add.unsqueeze(-1)], dim=-1)
            # print(pro)

            if eos_token_ids is not None:
                for eos_token_id in eos_token_ids:
                    eos_in_sents = tokens_to_add == eos_token_id
                    # if sentence is unfinished and the token to add is eos, sent_lengths is filled with current length
                    is_sents_unfinished_and_token_to_add_is_eos = unfinished_sents.mul(eos_in_sents.long()) != 0
                    sent_lengths.masked_fill_(is_sents_unfinished_and_token_to_add_is_eos, cur_len + 1 - input_ids.shape[-1])
                    # unfinished_sents is set to zero if eos in sentence
                    unfinished_sents.mul_((~eos_in_sents).long())

            cur_len = cur_len + 1

            # stop when there is a </s> in each sentence, or if we exceed the maximul length
            if unfinished_sents.max() == 0:
                break

        # if there are different sentences lengths in the batch, some batches have to be padded
        if sent_lengths.min().item() != sent_lengths.max().item():
            assert pad_token_id is not None, "`Pad_token_id` has to be defined if batches have different lengths"
            # finished sents are filled with pad_token
            decoded = responses.new(batch_size, sent_lengths.max().item()).fill_(pad_token_id)
        else:
            decoded = responses

        for hypo_idx, hypo in enumerate(responses):
            decoded[hypo_idx, : sent_lengths[hypo_idx]] = hypo[: sent_lengths[hypo_idx]]

        return decoded, x_att


def top_k_top_p_filtering(logits, top_k=0, top_p=1.0, filter_value=-float("Inf"), min_tokens_to_keep=1):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size, vocabulary size)
            if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
            Make sure we keep at least min_tokens_to_keep per batch example in the output
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


