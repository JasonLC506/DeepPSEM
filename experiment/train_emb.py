import argparse
import os
import _pickle as cPickle
import numpy as np

from experiment import DataDUELoader, json_reader, evaluate
from models import WordUserEmb
from common.readlogboard import read


RANDOM_SEED_NP = 2019


def train(
        config_file,
        meta_data_file,
        id_map,
        dataToken,
        batch_data_dir_train,
        batch_data_dir_valid=None,
        max_doc_length=30,
        size_context=2,
        model_name=None,
        restore_path=None
):
    np.random.seed(RANDOM_SEED_NP)
    data_train = DataDUELoader(
        meta_data_file=meta_data_file,
        batch_data_dir=batch_data_dir_train,
        id_map=id_map,
        dataToken=dataToken,
        max_doc_length=max_doc_length,
        size_context=size_context
    )
    if batch_data_dir_valid is not None:
        data_valid = DataDUELoader(
            meta_data_file=meta_data_file,
            batch_data_dir=batch_data_dir_valid,
            id_map=id_map,
            dataToken=dataToken,
            max_doc_length=max_doc_length,
            size_context=size_context
        )
    else:
        data_valid = None

    model_spec = json_reader(config_file)
    model = WordUserEmb(
        data_spec=data_train.data_spec,
        model_spec=model_spec,
        model_name=model_name
    )
    model.initialization()
    if restore_path is not None:
        model.restore(restore_path)

    # train #
    results = model.train(
        data_generator=data_train,
        data_generator_valid=data_valid
    )
    print("train_results: %s" % str(results))

    best_epoch = read(
        directory="../summary/" + model.model_name,
        main_indicator="epoch_losses_valid_00"
    )[0]
    print("best_epoch by validation loss: %d" % best_epoch)


class ArgParse(object):
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("-dm", "--data_name", default="foxnews_nolike")
        parser.add_argument("-mdl", "--max_doc_length", default=30, type=int)
        parser.add_argument("-sc", "--size_context", default=2, type=int)
        parser.add_argument("-cf", "--config_file", default="../models/word_user_emb_config.json")
        parser.add_argument("-rp", "--restore_path", default=None)
        parser.add_argument("-mn", "--model_name", default="test")
        parser.add_argument("-vt", "--valid_test", default="v", choices=["v", "t"])
        self.parser = parser

    def parse_args(self):
        return self.parser.parse_args()


if __name__ == "__main__":
    args = ArgParse().parse_args()

    data_dir = "C:/Users/jpz5181/Documents/GitHub/PSEM/data/CNN_foxnews/"
    data_prefix = "_CNN_foxnews_combined_K10"

    id_map_file = data_dir + "id_map" + data_prefix
    postcontent_dataW_file = data_dir + "dataW" + data_prefix
    postcontent_dataToken_file = data_dir + "dataToken" + data_prefix
    word_dictionary_file = data_dir + "word_dictionary" + data_prefix

    id_map, id_map_reverse = cPickle.load(open(id_map_file, "rb"))
    # dataW = cPickle.load(open(postcontent_dataW_file, "rb"), encoding='bytes')
    # print(dataW.nnz)
    dataToken = cPickle.load(open(postcontent_dataToken_file, "rb"))
    word_dictionary = cPickle.load(open(word_dictionary_file, "rb"))
    # print(word_dictionary)

    data_name = args.data_name
    data_dir = "C:/Users/jpz5181/Documents/GitHub/PSEM/data/" + data_name + "/"

    batch_rBp_dir = data_dir + "train/"
    batch_valid_on_shell_dir = data_dir + "on_shell/valid/"
    batch_valid_off_shell_dir = data_dir + "off_shell/valid/"
    batch_test_on_shell_dir = data_dir + "on_shell/test/"
    batch_test_off_shell_dir = data_dir + "off_shell/test/"

    meta_data_train_file = data_dir + "meta_data_train"
    meta_data_off_valid_file = data_dir + "meta_data_off_shell_valid"
    meta_data_off_test_file = data_dir + "meta_data_off_shell_test"
    meta_data_on_valid_file = data_dir + "meta_data_on_shell_valid"
    meta_data_on_test_file = data_dir + "meta_data_on_shell_test"

    train(
        config_file=args.config_file,
        meta_data_file=meta_data_train_file,
        id_map=id_map_reverse,
        dataToken=dataToken,
        batch_data_dir_train=batch_rBp_dir,
        batch_data_dir_valid=batch_valid_on_shell_dir,
        max_doc_length=args.max_doc_length,
        size_context=args.size_context,
        model_name=args.data_name if args.model_name is None else args.model_name + "_" + args.data_name,
        restore_path=args.restore_path,
    )

    # if args.restore_path is not None:
    #     if os.path.isdir(args.restore_path):
    #         file_names = list(os.listdir(args.restore_path))
    #         del file_names[file_names.index("checkpoint")]
    #         restore_paths = list(map(lambda x: x[:9], file_names))      # 9: len("epoch_%03d")
    #         restore_paths = sorted(list(set(restore_paths)))
    #         restore_paths = list(map(lambda x: os.path.join(args.restore_path, x), restore_paths))
    #     else:
    #         restore_paths = [args.restore_path]
    # else:
    #     restore_paths = None
    #
    # test(
    #     config_file=args.config_file,
    #     meta_data_file=meta_data_train_file,
    #     id_map=id_map_reverse,
    #     dataToken=dataToken,
    #     batch_data_dir=batch_valid_on_shell_dir if args.valid_test == "v" else batch_test_on_shell_dir,
    #     max_doc_length=args.max_doc_length,
    #     size_context=args.size_context,
    #     model_name=args.data_name if args.model_name is None else args.model_name + "_" + args.data_name,
    #     restore_path=restore_paths,
    # )
