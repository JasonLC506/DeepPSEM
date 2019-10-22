"""
pretrain word and user embeddings
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
import _pickle as cPickle

from models import NN
Optimizer = tf.train.AdamOptimizer


class WordUserEmb(NN):
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

        super(WordUserEmb, self).__init__(graph=None)

    def initialization(self):
        self.sess = self.setup_session(graph=self.graph)
        self.initialize(sess=self.sess)

    def _setup_placeholder(self):
        """
        each sample is a document
        """
        with tf.name_scope("placeholder"):
            self.uid = tf.placeholder(
                dtype=tf.int64,
                shape=[None],
                name="uid"
            )
            self.wids = tf.placeholder(
                dtype=tf.int64,
                shape=[None, self.data_spec["max_length"]],       # cut-off length of documents
                name="wids"
            )
            self.wids_neg = tf.placeholder(
                dtype=tf.int64,
                shape=[None, self.data_spec["max_length"]],       # cut-off length of documents
                name="wids"
            )
            self.cids = tf.placeholder(
                dtype=tf.int64,
                shape=[None, self.data_spec["max_length"], self.data_spec["size_context"]],   # number of context words
                name="cids"
            )
            self.lengths = tf.placeholder(
                dtype=tf.int64,
                shape=[None],
                name="lengths",                                   # true lengths of documents
            )
            self.label = tf.placeholder(
                dtype=tf.float32,
                shape=[None, self.n_labels],
                name="label"
            )

    def _setup_net(self):
        # word embedding #
        self.w_emb_valid = tf.Variable(
            tf.random_normal(
                [self.n_words, self.model_spec["emb_dim"]],
                mean=0.0,
                stddev=0.01
            ),
            dtype=tf.float32,
            name="w_emb_valid"
        )
        # padding word #
        self.w_emb_padding = tf.Variable(
            tf.zeros(
                [1, self.model_spec["emb_dim"]],
            ),
            dtype=tf.float32,
            name="w_emb_padding",
            trainable=False
        )
        # -1 padding #
        self.w_emb = tf.concat([self.w_emb_valid, self.w_emb_padding], axis=0)
        # embedding lookup #
        self.ws = tf.nn.embedding_lookup(
            params=self.w_emb,
            ids=self.wids,
        )
        self.ws_neg = tf.nn.embedding_lookup(
            params=self.w_emb,
            ids=self.wids_neg,
        )
        self.cs = tf.nn.embedding_lookup(
            params=self.w_emb,
            ids=self.cids
        )

        # user embedding #
        self.u_emb = tf.Variable(
            tf.random_normal(
                [self.n_users, self.model_spec["emb_dim"]],
                mean=0.0,
                stddev=0.1
            ),
            dtype=tf.float32,
            name="u_emb"
        )
        self.us = tf.nn.embedding_lookup(
            params=self.u_emb,
            ids=self.uid,
        )

        # user label logits #
        self.u_l_logit = tf.layers.dense(
            inputs=self.us,
            units=self.n_labels,
            use_bias=False,
            name="u_l_logit"
        )

        # word label logits #
        ws_avg = tf.reduce_sum(self.ws, axis=1) / tf.dtypes.cast(tf.expand_dims(self.lengths, axis=-1), tf.float32)
        self.w_l_logit = tf.layers.dense(
            inputs=ws_avg,
            units=self.n_labels,
            use_bias=False,
            name="w_l_logit"
        )

    def _setup_loss(self):
        # user word loss #
        loss_u_w_ = tf.maximum(
            0.0,
            1.0 - tf.einsum(
                "ijk,ik->ij",
                self.ws - self.ws_neg,
                self.us
            )
        )
        self.loss_u_w = tf.reduce_sum(loss_u_w_)   # given padding positive and negative words are same

        # word word loss #
        loss_w_w_ = tf.maximum(
            0.0,
            1.0 - tf.einsum(
                "ijk,ijck->ijc",
                self.ws - self.ws_neg,
                self.cs
            )
        )
        self.loss_w_w = tf.reduce_sum(loss_w_w_)   # given padding positive and negative words are same

        # user label loss #
        loss_u_l_ = tf.nn.softmax_cross_entropy_with_logits(
            labels=self.label,
            logits=self.u_l_logit
        )
        self.loss_u_l = tf.reduce_sum(loss_u_l_)

        # word label loss #
        loss_w_l_ = tf.nn.softmax_cross_entropy_with_logits(
            labels=self.label,
            logits=self.w_l_logit
        )
        self.loss_w_l = tf.reduce_sum(loss_w_l_)

        self.loss = self.model_spec["beta"] * (
            self.loss_u_w + self.loss_w_w
        ) + (1.0 - self.model_spec["beta"]) * (
            self.loss_u_l + self.loss_w_l
        )

    def _setup_optim(self):
        self.optimizer = Optimizer(
            learning_rate=self.model_spec["learning_rate"]
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
            op_losses=[self.loss, self.loss_u_w, self.loss_w_w, self.loss_u_l, self.loss_w_l],
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
            self.wids: data["wids"][batch_index],
            self.wids_neg: data["wids_neg"][batch_index],
            self.cids: data["cids"][batch_index],
            self.lengths: data["lengths"][batch_index],
            self.label: data["label"][batch_index]
        }
        return feed_dict

    def save_embs(
            self,
            save_path=None
    ):
        if save_path is None:
            save_path = "../ckpt/" + self.model_name
        w_emb, u_emb = self.sess.run(
            [self.w_emb, self.u_emb]
        )
        with open(save_path + "_w_emb", "wb") as wf:
            cPickle.dump(w_emb, wf)
        with open(save_path + "_u_emb", "wb") as uf:
            cPickle.dump(u_emb, uf)


if __name__ == "__main__":
    m = WordUserEmb(
        n_words=10,
        n_users=10,
        n_labels=5,
        data_spec={
            "max_length": 2,
            "size_context": 2
        },
        model_spec={
            "emb_dim": 3,
            "beta": 0.4,
            "learning_rate": 0.0001,
            "batch_size": 512,
            "max_epoch": 100,
            "name": "WordUserEmb"
        }
    )
