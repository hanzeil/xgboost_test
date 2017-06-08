//
// Created by tangjinghao on 2017/5/26.
//

#include <sstream>
#include <cmath>
#include <map>
#include <vector>
#include "XGBoost.h"

XGBoost::XGBoost() : cur_training_set_num_(0), training_set_num_(0),
                     predict_labels(NULL), h_train_(NULL),
                     h_test_(NULL) {
}

XGBoost::~XGBoost() {
    for (int i = 0; i < cur_training_set_num_; i++) {
        XGDMatrixFree(h_train_[i]);
    }
    XGDMatrixFree(h_test_[0]);
    XGBoosterFree(h_booster_);
};

void XGBoost::set_training_set_num(std::size_t num) {
    training_set_num_ = num;
    h_train_ = new DMatrixHandle[training_set_num_];
}

bool XGBoost::add_training_set_from_csv(
        const string &in_file_path,
        const std::size_t rows,
        const std::size_t cols) {

    create_matrix_from_csv(
            in_file_path,
            rows,
            cols,
            h_train_[cur_training_set_num_++]);
    return true;
}

bool XGBoost::train() {
// create the booster and load some parameters
    XGBoosterCreate(h_train_, training_set_num_, &h_booster_);
    XGBoosterSetParam(h_booster_, "booster", "gbtree");
    XGBoosterSetParam(h_booster_, "objective", "reg:linear");
    XGBoosterSetParam(h_booster_, "max_depth", "6");
    XGBoosterSetParam(h_booster_, "eta", "0.1");
    XGBoosterSetParam(h_booster_, "min_child_weight", "1");
    XGBoosterSetParam(h_booster_, "subsample", "0.9");
    XGBoosterSetParam(h_booster_, "colsample_bytree", "1");
    XGBoosterSetParam(h_booster_, "num_parallel_tree", "1");
// perform 200 learning iterations
    for (int i = 0; i < cur_training_set_num_; i++) {
        for (int iter = 0; iter < 200; iter++)
            XGBoosterUpdateOneIter(h_booster_, iter, h_train_[i]);
    }
}

bool XGBoost::set_test_set_from_csv(const string &in_file_path, const std::size_t rows, const std::size_t cols) {
    if (h_test_) {
        XGDMatrixFree(h_test_);
        h_test_ = NULL;
    }
    h_test_ = new DMatrixHandle();
    create_matrix_from_csv(in_file_path,
                           rows,
                           cols,
                           h_test_[0]);
}

bool XGBoost::create_matrix_from_csv(const string &in_file_path, const std::size_t rows, const std::size_t cols,
                                     DMatrixHandle &h_train) {

    std::ifstream in_file(in_file_path);
    std::string item;

    float *train_datas = NULL;
    float *train_labels = NULL;
    train_datas = new float[rows * (cols - 1)];
    train_labels = new float[rows];
    for (int i = 0; std::getline(in_file, item) && i < rows; i++) {
        std::size_t begin = 0;
        for (int j = 0; j < cols; j++) {
            std::size_t pos = item.find_first_of(',', begin);
            if (pos == string::npos) {
                if (j != cols - 1) {
                    return false;
                }
                std::string tmp_string = string(item.begin() + begin, item.end());
                train_labels[i] = (float) std::atof(tmp_string.c_str());
                break;
            }
            std::string tmp_string = string(item.begin() + begin, item.begin() + pos);
            train_datas[i * (cols - 1) + j] = (float) std::atof(tmp_string.c_str());
            // cout << train_datas[i * (cols - 1) + j] << " ";
            begin = pos + 1;
        }
        // cout << train_labels[i] << endl;
    }
    XGDMatrixCreateFromMat(train_datas, rows, cols - 1, -1, &h_train);

    XGDMatrixSetFloatInfo(h_train, "label", train_labels, rows);

    return true;
}

bool XGBoost::predict() {
    bst_ulong out_len;
    XGBoosterPredict(h_booster_, h_test_[0], 0, 0, &out_len, &predict_labels);
    return true;
}

void XGBoost::precision_and_recall() {
    bst_ulong test_labels_len;
    const float *test_labels;
    XGDMatrixGetFloatInfo(h_test_[0], "label", &test_labels_len, &test_labels);
    std::map<int, std::vector<std::size_t>> test_labels2index;
    std::map<int, std::vector<std::size_t>> predict_labels2index;
    std::map<std::size_t, int> test_index2labels;
    std::map<std::size_t, int> predict_index2labels;
    for (std::size_t i = 0; i < test_labels_len; i++) {
        test_labels2index[round(test_labels[i])].push_back(i);
        test_index2labels[i] = round(test_labels[i]);
        predict_labels2index[round(predict_labels[i])].push_back(i);
        predict_index2labels[i] = round(predict_labels[i]);
    }
    for (const auto key_values : test_labels2index) {
        int tp = 0, fp = 0, fn = 0;
        for (const auto &index : key_values.second) {
            if (predict_index2labels[index] == key_values.first) {
                tp++;
            } else {
                fn++;
            }
        }
        for (const auto &index : predict_labels2index[key_values.first]) {
            if (test_labels[index] != key_values.first) {
                fp++;
            }
        }
        cout
                << "label: " << key_values.first
                << " precision: "
                << (float) tp / (tp + fp)
                << " recall: "
                << (float) tp / (tp + fn) << endl;
    }
};

int XGBoost::round(float r) {
    return (r > 0.0) ? (int) floor(r + 0.5) : (int) ceil(r - 0.5);
}
