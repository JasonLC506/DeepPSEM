"""
BiGRU and user-specific attention aggregation
@inproceedings{zhang2018caden,
  title={CADEN: A Context-Aware Deep Embedding Network for Financial Opinions Mining},
  author={Zhang, Liang and Xiao, Keli and Zhu, Hengshu and Liu, Chuanren and Yang, Jingyuan and Jin, Bo},
  booktitle={2018 IEEE International Conference on Data Mining (ICDM)},
  pages={757--766},
  year={2018},
  organization={IEEE}
}
"""
import tensorflow as tf

from models import NN
Optimizer = tf.train.AdagradDAOptimizer


class Caden(NN):
    def __init__(
            self,
            data_spec,
            model_spec,
            model_name=None
    ):
        self.n_words = data_spec["n_words"]
        self.n_users = data_spec["n_users"]
        self.n_labels = data_spec["n_labels"]
        self.data_spec = data_spec
        self.model_spec = model_spec
        if model_name is None:
            self.model_name = model_spec["name"]
        else:
            self.model_name = model_name

        super(Caden, self).__init__(graph=None)

    def initialization(
            self,
            w_emb_file="../ckpt/w_emb",
            u_emb_file="../ckpt/u_emb"
    ):
        self.sess = self.setup_session(graph=self.graph)
        self.initialize(sess=self.sess)

        # load pretrained word embed #
        NN.initialize_fixed_input(
            sess=self.sess,
            fixed_input_placeholder=self.w_emb_placeholder,
            fixed_input_init=self.w_emb_init,
            fixed_input_file=w_emb_file
        )

        # load pretrained user embed #
        NN.initialize_fixed_input(
            sess=self.sess,
            fixed_input_placeholder=self.u_emb_placeholder,
            fixed_input_init=self.u_emb_init,
            fixed_input_file=u_emb_file
        )

    def _setup_placeholder(self):
        self.uid = tf.placeholder(
            dtype=tf.int64,
            shape=[None],
            name="uid"
        )
        self.wids = tf.placeholder(
            dtype=tf.int64,
            shape=[None, self.data_spec["max_length"], self.data_spec["size_local"]],  # number of context words
            name="wids"
        )
        self.label = tf.placeholder(
            dtype=tf.float32,
            shape=[None, self.n_labels],
            name="label"
        )

    def _setup_net(self):
        # word embedding #
        self.w_emb, self.w_emb_placeholder, self.w_emb_init = NN.fixed_input_load(
            input_shape=[self.n_words + 1, self.model_spec["emb_dim"]],
            trainable=False,
            name="w_emb"
        )
        # embedding lookup #
        self.ws = tf.nn.embedding_lookup(
            params=self.w_emb,
            ids=self.wids,
        )

        # user embedding #
        self.u_emb, self.u_emb_placeholder, self.u_emb_init = NN.fixed_input_load(
            input_shape=[self.n_users, self.model_spec["emb_dim"]],
            trainable=False,
            name="u_emb"
        )
        self.us = tf.nn.embedding_lookup(
            params=self.u_emb,
            ids=self.uid,
        )

        xs = tf.reshape(
            self.ws,
            shape=[
                -1,
                self.data_spec["max_length"],
                self.data_spec["size_local"] * self.model_spec["emb_dim"]
            ]
        )

        # rnn #
        rnn_cell_fw = tf.contrib.rnn.GRUBlockCell(
            num_units=self.model_spec["rnn_dim"],
            name="rnn_cell_forward"
        )
        rnn_cell_bw = tf.contrib.rnn.GRUBlockCell(
            num_units=self.model_spec["rnn_dim"],
            name="rnn_cell_backward"
        )
        rnn_outputs, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
            [rnn_cell_fw],
            [rnn_cell_bw],
            xs,
            dtype=tf.float32
        )

        # attention #
        score_hidden_rnn = tf.layers.dense(
            inputs=rnn_outputs,
            units=self.model_spec["score_hidden_dim"],
            use_bias=False
        )
        score_hidden_user = tf.layers.dense(
            inputs=self.us,
            units=self.model_spec["score_hidden_dim"],
            use_bias=True
        )
        score = tf.layers.dense(
            inputs=tf.math.tanh(
                score_hidden_rnn + tf.expand_dims(
                    score_hidden_user,
                    axis=1
                )
            ),
            units=1,
            use_bias=False
        )
        attention = tf.nn.softmax(
            tf.squeeze(
                score,
                axis=-1
            ),
            axis=-1
        )
        rnn_aggreated = tf.einsum(
            "ijk,ij->ik",
            rnn_outputs,
            attention
        )

        # fully connected layers #
        hidden = tf.layers.dense(
            inputs=rnn_aggreated,
            units=self.model_spec["hidden_dim"],
            activation=tf.nn.tanh
        )
        self.logit = tf.layers.dense(
            inputs=hidden,
            units=self.n_labels,
            use_bias=False
        )
        self.preds = tf.nn.softmax(self.logit, axis=-1)

    def _setup_loss(self):
        loss_ = tf.nn.softmax_cross_entropy_with_logits(
            labels=self.label,
            logits=self.logit
        )
        self.loss = tf.reduce_sum(loss_)

    def _setup_optim(self):
        self.optimizer = Optimizer(
            learning_rate=self.model_spec["learning_rate"],
            global_step=tf.dtypes.cast(0, dtype=tf.int64),
            l2_regularization_strength=self.model_spec["l2_regularization"]
        ).minimize(self.loss)

    def train(
            self,
            data_generator,
            data_generator_valid=None,
    ):
        if self.sess is None:
            self.initialization()
        results = self._train_w_generator(
            data_generator=data_generator,
            fn_feed_dict=self._fn_feed_dict_train,
            op_optimizer=self.optimizer,
            op_losses=[self.loss],
            session=self.sess,
            batch_size=self.model_spec["batch_size"],
            max_epoch=self.model_spec["max_epoch"],
            data_generator_valid=data_generator_valid,
            op_savers=[self.saver],
            save_path_prefixs=[self.model_name],
            log_board_dir="../summary/" + self.model_name
        )
        return results

    def _fn_feed_dict_train(
            self,
            data,
            batch_index
    ):
        feed_dict = {
            self.uid: data["uid"][batch_index],
            self.wids: data["wids_loc"][batch_index],
            self.label: data["label"][batch_index]
        }
        return feed_dict

    def predict(
            self,
            data_generator
    ):
        results = self._feed_forward_w_generator(
            data_generator=data_generator,
            fn_feed_dict=self._fn_feed_dict_predict,
            output=[self.preds],
            session=self.sess,
            batch_size=self.model_spec["batch_size"]
        )[0]
        return results

    def _fn_feed_dict_predict(
            self,
            data,
            batch_index
    ):
        feed_dict = {
            self.uid: data["uid"][batch_index],
            self.wids: data["wids_loc"][batch_index]
        }
        return feed_dict


if __name__ == "__main__":
    m = Caden(
        data_spec={
            "n_words": 10,
            "n_users": 10,
            "n_labels": 5,
            "max_length": 2,
            "size_local": 3
        },
        model_spec={
            "emb_dim": 100,
            "rnn_dim": 50,
            "score_hidden_dim": 50,
            "hidden_dim": 100,
            "learning_rate": 0.0001,
            "l2_regularization": 1e-7,
            "batch_size": 512,
            "max_epoch": 100,
            "name": "Caden"
        }
    )
