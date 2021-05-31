"""
https://tangshusen.me/2019/03/09/tf-attention/
"""
import tensorflow as tf
import keras

class SeqModel():
    """
    定义seq2seq模型，训练机器人
    """
    def __init__(self):
        pass

    def model_inputs(self):
        """
        模型输入定义
        :return:

        """
        tf.compat.v1.disable_eager_execution()
        input_data = tf.compat.v1.placeholder(tf.int32, [None, None], name='input')
        targets = tf.compat.v1.placeholder(tf.int32, [None, None], name='targets')
        # 学习率
        lr = tf.compat.v1.placeholder(tf.float32, name='learning_rate')
        keep_prob = tf.compat.v1.placeholder(tf.float32, name='keep_prob')
        return input_data, targets, lr, keep_prob


    def process_encoding_input(self, target_data, vocab_to_int, batch_size):
        """
        删除每个批次中最后一个单词ID，并在每个批次开头追加<GO>标记
        :param target_data:
        :param batch_size:
        :param vocab_to_int:
        :return:
        """

        # 对张量进行切片操作
        ending = tf.strided_slice(target_data, [0, 0], [batch_size, -1], [1, 1])
        dec_input = tf.concat([tf.fill([batch_size, 1], vocab_to_int['<GO>']), ending], 1)

        return dec_input

    def encodig_layer(self, rnn_inputs, rnn_size, num_layers, keep_prod, sequence_length):
        """
        定义模型，使用LSTM神经元和双向编码器来定义Seq2Seq模型的编码层
        :return:
        """
        lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
        drop = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob=keep_prod)

        # 多层循环神经网络编码器
        enc_cell = tf.contrib.rnn.MultiRNNCell([drop] * num_layers)
        # 双向RNN
        enc_output, enc_state = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=enc_cell,
            cell_bw=enc_cell,
            sequence_length=sequence_length,
            inputs=rnn_inputs,
            dtype=tf.float32
        )

        return enc_output, enc_state

    def decoding_layer_train(self, encoder_state, dec_cell, dec_embed_input,
                             sequence_length, decoding_scope,
                             output_fn, keep_prob, batch_size):
        """
        解码器层 --- 加入attention机制
        :param encoder_state:
        :param dec_cell:
        :param dec_embed_input:
        :param sequence_length:
        :param decoding_scope:
        :param output_fn:
        :param keep_prob:
        :param batch_size:
        :return:
        """
        # attention 矩阵 U、V、K
        attention_states = tf.zeros([batch_size, 1, dec_cell.output_size])

        # tf.contrib.seq2seq.prepare_attention
        attention_mechanism = tf.contrib.seq2seq.BahdanauMonotonicAttention(
            dec_cell.output_size, dec_cell
        )
        attention_cell = tf.contrib.seq2seq.AttentionWrapper(dec_cell, attention_mechanism, batch_size,
                                                             name="attention_wrapper")

        init_state = attention_cell.zero_state(batch_size, tf.float32).clone(cell_state=encoder_state)

        train_decoder_fn = tf.contrib.seq2seq.attention_decoder_fn_train(encoder_state[0],
                                                                         att_keys,
                                                                         att_vals,
                                                                         att_score_fn,
                                                                         att_construct_fn,
                                                                         name='attn_dec_train')

        train_pred, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(dec_cell,
                                                                  train_decoder_fn,
                                                                  dec_embed_input,
                                                                  sequence_length,
                                                                  scope=decoding_scope)
        train_pred_drop = tf.nn.dropout(train_pred, keep_prob)

        return output_fn(train_pred_drop)

    def decoding_layer_infer(self, encoder_state, dec_cell,
                             dec_embeddings, start_of_seqence_id,
                             end_of_sequence_id, maximum_length,
                             vocab_size, decoding_scope,
                             output_fn, keep_prob, batch_size):
        """
        为所查询对问题创建合适对回复，
        :param encoder_state:
        :param dec_cell:
        :param dec_embeddings:
        :param start_of_seqence_id:
        :param end_of_sequence_id:
        :param maximum_length:
        :param vocab_size:
        :param decoding_scope:
        :param output_fn:
        :param keep_prod:
        :param batch_size:
        :return:
        """
        attention_states = tf.zeros([batch_size, 1, dec_cell.output_size])
        att_keys, att_vals, att_score_fn, att_construct_fn = tf.contrib.seq2seq.prepare_attention(
            attention_states,
            attention_option="bahdanau",
            num_units=dec_cell.output_size
        )

        infer_decoder_fn = tf.contrib.seq2seq.attention_decoder_fn_inference(
            output_fn,
            encoder_state[0],
            att_keys,
            att_vals,
            att_score_fn,
            att_construct_fn,
            dec_embeddings,
            start_of_seqence_id,
            end_of_sequence_id,
            maximum_length,
            vocab_size,
            name='attn_dec_inf'
        )

        infer_logits, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(
            dec_cell, infer_decoder_fn, scope=decoding_scope
        )
        return infer_logits

    def decoding_layer(self, dec_embed_input, dec_embedding, encoder_state, vocab_sise,
                       sequence_length, rnn_size, num_layers, vocab_to_int, keep_prob, batch_size):
        """
        创建推理和训练分对数, 并使用给定对标准差通过截断的正态分布来初始化权重和偏差
        :return:
        """
        with tf.variable_scope("decoding") as decoding_scope:
            lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
            drop = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob=keep_prob)
            dec_cell = tf.contrib.rnn.MultiRNNCell([drop] * num_layers)

            weights = tf.truncated_normal_initializer(stddev=0.1)
            biases = tf.zeros_initializer()

            output_fn = lambda x: tf.contrib.layers.fully_connected(
                x,
                vocab_sise,
                None,
                scope=decoding_scope,
                weights_initializer=weights,
                biases_initializer=biases
            )
        # 训练Decoder
        train_logits = self.decoding_layer_train(encoder_state,
                                                 dec_cell,
                                                 dec_embed_input,
                                                 sequence_length,
                                                 decoding_scope,
                                                 output_fn,
                                                 keep_prob,
                                                 batch_size)
        decoding_scope.reuse_varibles()
        infer_logits = self.decoding_layer_infer(encoder_state,
                                                 dec_cell,
                                                 dec_embedding,
                                                 vocab_to_int['<GO>'],
                                                 vocab_to_int['<EOS>'],
                                                 sequence_length - 1,
                                                 vocab_sise,
                                                 decoding_scope,
                                                 output_fn,
                                                 keep_prob,
                                                 batch_size)

        return train_logits, infer_logits

    def seq2seq_model(self, input_data, target_data,
                       keep_prob, batch_size,
                       sequence_length,
                       answers_vocab_size,
                       questions_vocab_size,
                       enc_embedding_size,
                       dec_embedding_size,
                       rnn_size,
                       num_layers,
                       questions_vocab_to_int):
        """
        用户将编码器等串联起来，使用随机均匀分布来初始化词嵌入过程。
        :param input_data:
        :param target_data:
        :param keep_prob:
        :param batch_size:
        :param sequence_length:
        :param answers_vocab_size:
        :param questions_vocab_size:
        :param enc_embedding_size:
        :param dec_embedding_size:
        :param rnn_size:
        :param num_layers:
        :param questions_vocab_to_int:
        :return:
        """
        # tf.contrib.layers.embed_sequence 为句子产生词嵌入
        enc_embed_input = tf.contrib.layers.embed_sequence(
            input_data,
            answers_vocab_size + 1,
            enc_embedding_size,
            initializer=tf.random_normal_initializer(0, 1)
        )

        enc_output, enc_state = self.encodig_layer(enc_embed_input,
                                       rnn_size,
                                       num_layers,
                                       keep_prob,
                                       sequence_length)

        dec_input = self.process_encoding_input(target_data,
                                                questions_vocab_to_int,
                                                batch_size)

        dec_embeddings = tf.Variable(
            tf.random_uniform([questions_vocab_size + 1, dec_embedding_size], 0, 1)
        )
        dec_embed_input = tf.nn.embedding_lookup(
            dec_embeddings,
            dec_input
        )

        train_logits, infer_logits = self.decoding_layer(dec_embed_input,
                                                         dec_embeddings,
                                                         enc_state,
                                                         questions_vocab_size,
                                                         sequence_length,
                                                         rnn_size,
                                                         num_layers,
                                                         questions_vocab_to_int,
                                                         keep_prob,
                                                         batch_size)
        return train_logits, infer_logits
