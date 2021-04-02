from __future__ import print_function
import logging
import os
import numpy as np
import torch
import random
import datetime
import csv
from mmcv.runner import get_dist_info
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import cm
import time
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd

# import matplotlib.pyplot as plt
# plt.rc('font',family='Times New Roman')

CIFAR10_classes = ["airplane", "automobile", "bird",  "cat",  "deer",
                   "dog",      "frog",       "horse", "ship", "truck"]
def _norm(matrix, axis=-1):
    return np.sum(matrix ** 2, axis=axis) ** 0.5

def classifier_analysis(weights, samples_per_cls, save_dir='./'):
    # draw_post_norm(weights)
    # draw_post_CDT(weights, samples_per_cls)
    draw_norm(weights, save_dir)

    num_classes = weights.shape[0]
    fig, axes =plt.subplots(2, 5, figsize=(30, 8))
    for i in range(num_classes):
        sns.histplot(weights[i], kde=False, ax=axes[i // 5, i % 5])
    _savefig('./weight_values.jpg')
    all_weight_mean = np.mean(np.abs(weights), axis=0)
    all_weight_var = np.var(weights, axis=0)
    index = np.argsort(-all_weight_mean)
    fig, axes =plt.subplots(1, 3, figsize=(25, 8))
    sns.barplot(np.arange(weights.shape[1]), all_weight_mean[index], ax=axes[0]).set_title('mean')
    sns.barplot(np.arange(weights.shape[1]),all_weight_var[index], ax=axes[1]).set_title('var')
    _savefig('./av_weight_all.jpg')

    # fig, axes =plt.subplots(1, 1, figsize=(8, 8))
    # avg_weight = np.mean(weights, axis=0,keepdims=True)
    # cos = avg_weight * weights / _norm(weights) / _norm(avg_weight)
    # sns.barplot(np.arange(weights.shape[1]), cos)
    # _savefig('./weight_cos_to_avg.jpg')


def attack_step_analysis(labels, success_steps, mode, pgd_iters=20):
    print('* * * Evaluating attacking success rate! * * *')
    labels = np.asarray(labels, dtype=np.uint8)
    if len(success_steps.shape) == 2: # which is an early bug
        success_steps = np.hstack(success_steps)
    num_classes = np.max(labels) + 1
    # fig, axes =plt.subplots(2, 5, figsize=(30, 8))
    fig, axes =plt.subplots(figsize=(30, 8))
    for i in range(num_classes):
        outs_cla = success_steps[(labels == i)*(success_steps > 0)]
        success_rate = 1 - np.sum((labels == i)*(success_steps == 0)) / np.sum(labels == i)
        mean_steps = np.mean(outs_cla) / pgd_iters
        var_steps = np.var(outs_cla) / pgd_iters
        # sns.histplot(outs_cla, kde=False, ax=axes[i // 5, i % 5], discrete=True)
        # axes[i // 5, i % 5].title.set_text('success rate {:.3f}'.format(
        #     1 - np.sum((labels == i)*(success_steps == 0)) / np.sum(labels == i)))
    _savefig('./{}_success_step.jpg'.format(mode))
    print('results saved at ./{}_success_step.jpg'.format(mode))


def tsne(labels, clean_features, pgd_features, samples_per_cls, weights, mode, per_cls=True):

    num_classes = weights.shape[0]
    labels = np.asarray(labels, dtype=np.uint8)
    clean_features = pgd_features
    feat_cols = [ 'feat_dim_'+str(i) for i in range(clean_features.shape[1])]

    df_clean = pd.DataFrame(clean_features, columns=feat_cols)
    df_clean['y'] = labels
    df_clean['label'] = df_clean['y'].apply(lambda i: str(i))
    df_pgd = pd.DataFrame(pgd_features, columns=feat_cols)
    df_pgd['y'] = labels
    df_pgd['label'] = df_clean['y'].apply(lambda i: str(i))

    print('Size of the dataframe: {}'.format(df_clean.shape))
    if mode == 'train':
        n_min, n_max = 100, 1000
        select_index = []
        assert n_min <= min(samples_per_cls)
        ratio = (n_max - n_min) / (max(samples_per_cls) - n_min)
        for i in range(num_classes):
            num = int(ratio * samples_per_cls[i])
            class_index = np.arange(len(labels))[labels == i]
            index = np.random.permutation(class_index)[:num]
            select_index.extend(index.tolist())
            print(len(index), end=' ')
        print()
        df_subset = df_clean.loc[select_index,:].copy()
    else:
        N = len(labels) // 2
        np.random.seed(0)
        rndperm = np.random.permutation(df_clean.shape[0])
        df_subset = df_clean.loc[rndperm[:N],:].copy()
        df_pgd_subset = df_pgd.loc[rndperm[:N],:].copy()

    data_subset = df_subset[feat_cols].values

    pca_50 = PCA(n_components=50)
    pca_result_50 = pca_50.fit_transform(data_subset)
    print('Cumulative explained variation for 50 principal components: {}'.format(np.sum(pca_50.explained_variance_ratio_)))

    time_start = time.time()
    tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=300)
    tsne_pca_results = tsne.fit_transform(pca_result_50)
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
    df_subset['tsne-pca50-one'] = tsne_pca_results[:,0]
    df_subset['tsne-pca50-two'] = tsne_pca_results[:,1]
    plt.figure(figsize=(8,6))
    ax3 = plt.subplot(1, 1, 1)
    sns.scatterplot(
        x="tsne-pca50-one", y="tsne-pca50-two",
        hue="y",
        palette=sns.color_palette("muted", 10),
        data=df_subset,
        legend="full",
        alpha=0.3,
        ax=ax3
    )
    plt.subplots_adjust(left=0.1, right=0.7)
    ax3.set_xlabel("")
    ax3.set_ylabel("")
    # ax3.legend(fontsize=20)
    plt.legend(fontsize=20, bbox_to_anchor=(1.05,0), loc=3, borderaxespad=0)
    _savefig('./{}_pca50_tsne.jpg'.format(mode), dpi=300)
    if per_cls:
        fig, axes =plt.subplots(2, 5, figsize=(25, 8))
        cmap = sns.color_palette("muted", 10)
        # import ipdb; ipdb.set_trace()
        for cls in range(num_classes):
            df_this_set = df_subset[df_subset["y"] == cls]
            sns.scatterplot(
                x="tsne-pca50-one", y="tsne-pca50-two",
                hue="y",
                # palette=sns.color_palette("muted", 10),
                palette=[cmap[cls]],
                data=df_this_set,
                legend="full",
                alpha=0.3,
                ax=axes[cls // 5, cls % 5]
            )
            axes[cls // 5, cls % 5].set_xlabel("")
            axes[cls // 5, cls % 5].set_ylabel("")
            axes[cls // 5, cls % 5].legend(loc='upper left', fontsize=10)
            axes[cls // 5, cls % 5].get_legend().remove()
            axes[cls // 5, cls % 5].set_title("class {}: {}".format(cls,CIFAR10_classes[cls]), fontsize=25)
    _savefig('./{}_pca50_tsne_subsets.jpg'.format(mode), dpi=300)



def tsne_for_pairs(labels, pairing, clean_features, pgd_features, samples_per_cls, weights, mode):
    print('* * * t-SNE visualization of pair attacking * * *')
    num_classes = weights.shape[0]
    labels = np.asarray(labels, dtype=np.uint8)

    pairs = []
    for i, j in enumerate(pairing):
        if (j,i) not in pairs:
            pairs.append((i,j))

    def _draw_tsne(df_subset, data_clean, data_pgd, ax_clean, ax_pgd):
        pca_50 = PCA(n_components=50)
        all_data = np.vstack([data_clean, data_pgd])
        pca_result_50 = pca_50.fit_transform(all_data)
        print('Cumulative explained variation for 50 principal components: {}'.format(np.sum(pca_50.explained_variance_ratio_)))
        time_start = time.time()
        tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=300)
        tsne_pca_results = tsne.fit_transform(pca_result_50)
        print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

        split = tsne_pca_results.shape[0] // 2
        df_subset['tsne-pca50-one'] = tsne_pca_results[:split,0]
        df_subset['tsne-pca50-two'] = tsne_pca_results[:split,1]
        sns.scatterplot(
            x="tsne-pca50-one", y="tsne-pca50-two",
            hue="y",
            palette=sns.color_palette("hls", 2),
            data=df_subset,
            legend="full",
            alpha=0.3,
            ax=ax_clean
        )
        df_subset['tsne-pca50-one'] = tsne_pca_results[split:,0]
        df_subset['tsne-pca50-two'] = tsne_pca_results[split:,1]
        sns.scatterplot(
            x="tsne-pca50-one", y="tsne-pca50-two",
            hue="y",
            palette=sns.color_palette("hls", 2),
            data=df_subset,
            legend="full",
            alpha=0.3,
            ax=ax_pgd
        )

    feat_cols = [ 'feat_dim_'+str(i) for i in range(clean_features.shape[1])]

    df_clean = pd.DataFrame(clean_features, columns=feat_cols)
    df_clean['y'] = labels
    df_clean['label'] = df_clean['y'].apply(lambda i: str(i))

    df_pgd = pd.DataFrame(pgd_features, columns=feat_cols)
    df_pgd['y'] = labels
    df_pgd['label'] = df_clean['y'].apply(lambda i: str(i))

    plt.figure(figsize=(60 ,10))
    if mode == 'train':
        n_min, n_max = 100, 1000
        assert n_min <= min(samples_per_cls)
        ratio = (n_max - n_min) / (max(samples_per_cls) - n_min)
        for p, pair in enumerate(pairs):
            select_index = []
            for i in pair:
                num = int(ratio * samples_per_cls[i])
                class_index = labels == i
                index = np.random.permutation(class_index)[:num]
                select_index.extend(index.tolist())
                print(len(index), end=' ')
            print()
            for i_df, df in enumerate([df_clean, df_pgd]):
                df_subset = df.loc[select_index,:].copy()
                data_subset = df_subset[feat_cols].values
                ax = plt.subplot(2, len(pairs), 1 + p + i_df * len(pairs))
                _draw_tsne(df_subset, data_subset, ax)

    else:
        N = len(labels) // 10
        np.random.seed(0)
        for p, pair in enumerate(pairs):
            select_index = []
            for i in pair:
                class_index = np.arange(len(labels))[labels == i]
                index = np.random.permutation(class_index)[:N]
                select_index.extend(index.tolist())
                print(len(index), end=' ')
            print()

            df_subset_clean = df_clean.loc[select_index,:].copy()
            data_clean = df_subset_clean[feat_cols].values
            df_subset_pgd = df_pgd.loc[select_index,:].copy()
            data_pgd= df_subset_pgd[feat_cols].values
            ax_clean = plt.subplot(2, len(pairs), 1 + p)
            ax_pgd = plt.subplot(2, len(pairs), 1 + p + len(pairs))
            _draw_tsne(df_subset_clean, data_clean, data_pgd, ax_clean, ax_pgd)

    _savefig('./{}_pca50_tsne.jpg'.format(mode), dpi=200)

def draw_norm(weights, save_dir='./'):
    print('* * * Drawing the weight norm distribution * * *')
    cls_weight_norm = np.sum(weights ** 2, axis=-1)**0.5

    fig = plt.figure(figsize=(7,5))
    ax1 = fig.add_subplot(1,1,1)
    ax1.set_xlabel('class index')
    ax1.set_ylabel('weight L2 norm')
    ax1.set_title('Classifier Weight L2-norm')
    ax1.barplot(np.arange(weights.shape[0]), cls_weight_norm)
    plt.legend()
    _savefig(os.path.join(save_dir, 'weight_norm.jpg'), dpi=200)

def draw_post_norm(weights):
    print('* * * Drawing the weight proportion after normalization * * *')
    cls_weight_norm = np.sum(weights ** 2, axis=-1)**0.5
    tau_set = [0, 1, 2, 3, 4, 5]

    fig = plt.figure(figsize=(7,5))
    ax1 = fig.add_subplot(1,1,1)
    ax1.set_xlabel('class index')
    ax1.set_ylabel('proportion')
    ax1.set_title('Classifier weight L2-norm proportion after tau-normalization')
    colormap = plt.cm.viridis
    colors = [colormap(i) for i in np.linspace(0, 1,len(tau_set))]

    for i, tau in enumerate(tau_set):
        cls_weight_after = cls_weight_norm / (cls_weight_norm**tau)
        cls_weight_after = cls_weight_after / np.max(cls_weight_after)
        ax1.plot(np.arange(weights.shape[0]),cls_weight_after, c=colors[i], marker='s', markersize=5, label='tau-{}'.format(tau))
    plt.legend()
    _savefig('./new_images/postnorm.jpg', dpi=200)

def draw_post_CDT(weights, samples_per_cls):
    print('* * * Drawing the weight proportion after class-dependent re-scaling * * *')
    cls_weight_norm = np.sum(weights ** 2, axis=-1)**0.5
    tau_set = [0, 0.1, 0.2, 0.3, 0.5, 1]
    samples_per_cls = np.asarray(samples_per_cls)

    fig = plt.figure(figsize=(7,5))
    ax1 = fig.add_subplot(1,1,1)
    ax1.set_xlabel('class index')
    ax1.set_ylabel('proportion')
    ax1.set_title('Classifier weight L2-norm proportion after tau-sample-number re-scaling')
    colormap = plt.cm.viridis
    colors = [colormap(i) for i in np.linspace(0, 1,len(tau_set))]

    for i, tau in enumerate(tau_set):
        cls_weight_after = cls_weight_norm / (samples_per_cls**tau)
        cls_weight_after = cls_weight_after / np.max(cls_weight_after)
        ax1.plot(np.arange(weights.shape[0]),cls_weight_after, c=colors[i], marker='s', markersize=5, label='tau-{}'.format(tau))
    plt.legend()
    _savefig('./new_images/post_CDT.jpg', dpi=200)

def feature_analysis(labels, clean_features, pgd_features, weights, clean_outs, pgd_outs, w_norm=0, save_dir='./', per_cls=False):

    labels = np.asarray(labels, dtype=np.uint8)
    num_classes = np.max(labels) + 1
    feature_dim = clean_features.shape[-1]

    clean_norms = []
    pgd_norms = []
    norm_props = []
    if per_cls:
        assert num_classes == 10
        fig, axes =plt.subplots(2, 5, figsize=(25, 8))
        sample_index = np.random.randint(0, feature_dim, 30)
    clean_correct = clean_outs.argmax(1) == labels
    pgd_correct = pgd_outs.argmax(1) == labels

    # all clean norms
    fig, axes = plt.subplots(1, 1, figsize=(10, 5))
    all_clean_norms = np.sum(clean_features[clean_correct > 0] ** 2, axis=-1)**0.5
    all_clean_robust_norms = np.sum(clean_features[pgd_correct > 0] ** 2, axis=-1)**0.5
    all_clean_nonrob_norms = np.sum(clean_features[clean_correct * (1 - pgd_correct) > 0] ** 2, axis=-1)**0.5
    sns.histplot(all_clean_norms, kde=True, color='skyblue', alpha=0.3 , label='correct clean', element='step')
    sns.histplot(all_clean_robust_norms, kde=True, color='g', alpha=0.3 , label='robust clean', element='step')
    sns.histplot(all_clean_nonrob_norms, kde=True, color='red', alpha=0.3, label='non-robust clean', element='step')
    plt.legend()
    _savefig('./norm_clean.jpg')

    # scaling
    fs = 30
    fs_2 = 30
    fig, axes =plt.subplots(1, 1, figsize=(10, 7))
    all_robust_norms = np.sum(pgd_features[pgd_correct > 0] ** 2, axis=-1)**0.5 / np.sum(clean_features[pgd_correct > 0] ** 2, axis=-1)**0.5
    all_attacked_norms = np.sum(pgd_features[clean_correct * (1 - pgd_correct) > 0] ** 2, axis=-1)**0.5 / np.sum(clean_features[clean_correct * (1 - pgd_correct) > 0] ** 2, axis=-1)**0.5
    sns.histplot(all_robust_norms, kde=True, color='steelblue', alpha=0.3 , label='robust', element='step') # lightseagreen
    sns.histplot(all_attacked_norms, kde=True, color='firebrick', alpha=0.3, label='attacked', element='step') # crimson
    # plt.title('Feature Norm Scaling Ratio after Attack',fontsize=fs)
    plt.legend(fontsize=fs)
    # plt.xlabel('Scaling Ratio', fontsize=fs)

    # plt.ylabel('Count', fontsize=fs)
    axes.set(xlim=(0.45, 1.55)) #axes.set(xlim=(0.4, 1.6))
    plt.yticks(fontsize=fs_2)
    plt.xticks(fontsize=fs_2)
    _savefig('./norm_scaling.jpg')

    # clean and pgd features
    fig, axes =plt.subplots(1, 1, figsize=(10, 5))
    all_clean_norms = np.sum(clean_features[clean_correct > 0] ** 2, axis=-1)**0.5
    pgd_robust_norms = np.sum(pgd_features[pgd_correct > 0] ** 2, axis=-1)**0.5
    pgd_attacked_norms = np.sum(pgd_features[clean_correct * (1 - pgd_correct) > 0] ** 2, axis=-1)**0.5
    sns.histplot(all_clean_norms, kde=True, color='skyblue', alpha=0.3 , label='correct clean', element='step')
    sns.histplot(pgd_robust_norms, kde=True, color='cyan', alpha=0.3 , label='pgd robust', element='step')
    sns.histplot(pgd_attacked_norms, kde=True, color='red', alpha=0.3, label='pgd attacked', element='step')

    _savefig('./norm_both.jpg')

    for i in range(num_classes):
        bad_idx = (labels == i) * (1 - clean_correct)
        attacked_idx = (labels == i) * clean_correct * (1 - pgd_correct)
        robust_idx = (labels == i) * pgd_correct# # (1 - clean_correct) #  # * (1 - pgd_correct)#
        clean_per_cls = clean_features[(labels == i)]
        # att_pgd_per_cls = pgd_features[attacked_idx > 0]
        rob_pgd_per_cls = pgd_features[labels == i]
        clean_norms.append(np.mean(np.sum(clean_per_cls ** 2, axis=-1)**0.5))
        pgd_norms.append(np.mean(np.sum(rob_pgd_per_cls ** 2, axis=-1)**0.5))
        # norm_props.append(np.mean(np.sum(pgd_per_cls ** 2, axis=-1)**0.5 / np.sum(clean_per_cls ** 2, axis=-1)**0.5))
        norm_props.append(np.mean(np.sum(rob_pgd_per_cls ** 2, axis=-1)**0.5 / np.sum(clean_per_cls ** 2, axis=-1)**0.5))
        if per_cls:
            var_feats_cla = np.var(clean_per_cls, axis=0)
            mean_feats_cla = np.mean(clean_per_cls, axis=0)
            sns.barplot(x=np.arange(len(sample_index)), y=mean_feats_cla[sample_index], ax=axes[i // 5, i % 5])
            # sns.barplot(x=np.arange(len(sample_index)), y=var_feats_cla[sample_index], ax=axes[i // 5, i % 5])
            # sns.scatterplot(x=weights[i], y=np.mean(np.abs(weights[i] * (outs_cla-pgd_outs_cla)), axis=0), sizes=0.11, ax=axes[i // 5, i % 5])
            # sns.scatterplot(x=np.arange(feature_dim), y=weights[i], color='crimson', sizes=0.1, ax=axes[i // 5, i % 5])
            # axes[i // 5, i % 5].set_ylim(0,0.5)
    _savefig('./feature_var.jpg')

    weight_norms = np.sum(weights ** 2, axis=-1)**0.5
    weight_norms /= np.power(weight_norms, w_norm)
    idx = np.argsort(-weight_norms)

    fig, axes =plt.subplots(1, 1, figsize=(8, 7))
    # sns.color_palette("husl", 10)
    # sns.set_theme(palette=sns.color_palette("ch:s=.25,rot=-.25", as_cmap=True))
    sns.barplot(np.arange(num_classes), np.asarray(weight_norms), palette=reversed(sns.cubehelix_palette(num_classes, start=.5, rot=-.75)))#.set_title('Classifier Weight L2-norm', fontsize=fs)
    plt.xticks(np.linspace(0,num_classes-1,10))
    # plt.xlabel('Class Index', fontsize=fs)
    # plt.ylabel('Norm', fontsize=fs)
    axes.set(ylim=(min(weight_norms)*3/4,max(weight_norms)+0.1))
    sns.set_palette("husl")
    plt.yticks(fontsize=fs_2)
    plt.xticks(fontsize=fs_2)
    _savefig(os.path.join(save_dir, 'tuned_norms.jpg'))
    exit()

    figsize = (16,12)
    fig, axes = plt.subplots(2, 2 , figsize=figsize)
    sns.barplot(np.arange(num_classes), clean_norms, ax=axes[0,0]).set_title('Clean Feature L2-norm')
    sns.barplot(np.arange(num_classes),pgd_norms, ax=axes[0,1]).set_title('Attacked Feature L2-norm')

    sns.barplot(np.arange(num_classes),np.asarray(norm_props), ax=axes[1,0]).set_title('L2-norm Proportion after Attack')
    sns.barplot(np.arange(num_classes), np.asarray(weight_norms), ax=axes[1,1]).set_title('Classifier Weight L2-norm')

    axes[1,0].set(ylim=(min(norm_props)-0.01,max(norm_props)+0.01))
    axes[1,1].set(ylim=(min(weight_norms)*2/3,max(weight_norms)+0.1))
    _savefig(os.path.join(save_dir, 'norms.jpg'))
    exit()
    #
    # fig, axes =plt.subplots(2, 5, figsize=(25, 8))
    # for i in range(num_classes):
    #     outs_cla = pgd_features[labels == i]
    #     var_feats_cla = np.var(outs_cla, axis=0)
    #     sns.histplot(var_feats_cla, kde=False, ax=axes[i // 5, i % 5])
    #     axes[i // 5, i % 5].set_xlim(0,0.1)
    # _savefig('./pgd_feature_var.jpg')



def output_analysis(labels, clean_outs, attack_out_dict):

    labels = np.asarray(labels, dtype=np.uint8)
    num_classes = clean_outs.shape[-1]

    clean_pred = np.argmax(clean_outs, axis=-1)
    print(clean_pred[:10])
    gt_onehot = _onehot(labels, num_classes)
    clean_pred_onehot = _onehot(clean_pred, num_classes)
    print(clean_pred_onehot[:2])
    clean_recall = np.sum(gt_onehot * clean_pred_onehot, axis=0) / np.sum(gt_onehot, axis=0)
    print('clean recall')
    print(clean_recall)
    for key, data in attack_out_dict.items():
        attack_pred = np.argmax(data, axis=-1)
        attack_pred_onehot = _onehot(attack_pred, num_classes)
        attack_recall = np.sum(gt_onehot * attack_pred_onehot, axis=0) / np.sum(gt_onehot, axis=0)
        stay_right = np.sum(gt_onehot * attack_pred_onehot * clean_pred_onehot, axis=0) / np.sum(gt_onehot, axis=0)
        right_to_false = np.sum(gt_onehot * (1 - attack_pred_onehot) * clean_pred_onehot, axis=0) / np.sum(gt_onehot, axis=0)
        false_to_right = np.sum(gt_onehot * (1 - clean_pred_onehot) * attack_pred_onehot, axis=0) / np.sum(gt_onehot, axis=0)
        print(key)
        print(stay_right)
        print(right_to_false)
        print(false_to_right)

    # fig, axes =plt.subplots(2, 5, figsize=(25, 8))
    # for i in range(num_classes):
    #     clean_outs_cla = clean_outs[labels == i]
    #     sns.histplot(clean_outs_cla[:,i], kde=False, ax=axes[i // 5, i % 5])
    # _savefig('./clean_out_logits.jpg')
    #
    # fig, axes = plt.subplots(2, 5, figsize=(25, 8))
    # for i in range(num_classes):
    #     clean_outs_cla = _softmax(clean_outs[labels == i])
    #     sns.histplot(clean_outs_cla[:,i], kde=False, ax=axes[i // 5, i % 5])
    # _savefig('./clean_out_softmaxs.jpg')
    #
    # fig, axes = plt.subplots(2, 5, figsize=(25, 8))
    # for i in range(num_classes):
    #     outs_cla = _softmax(attack_out_dict['FGSM'][labels == i])
    #     sns.histplot(outs_cla[:,i], kde=False, ax=axes[i // 5, i % 5])
    # _savefig('./fgsm_out_softmaxs.jpg')
    #
    # fig, axes = plt.subplots(2, 5, figsize=(25, 8))
    # for i in range(num_classes):
    #     outs_cla = _softmax(attack_out_dict['PGD-20'][labels == i])
    #     sns.histplot(outs_cla[:,i], kde=False, ax=axes[i // 5, i % 5])
    # _savefig('./pgd_out_softmaxs.jpg')
    #
    # fig, axes = plt.subplots(2, 5, figsize=(25, 8))
    # for i in range(num_classes):
    #     outs_cla = attack_out_dict['FGSM'][labels == i]
    #     sns.histplot(outs_cla[:,i], kde=False, ax=axes[i // 5, i % 5])
    # _savefig('./fgsm_out_logits.jpg')
    #
    # fig, axes = plt.subplots(2, 5, figsize=(25, 8))
    # for i in range(num_classes):
    #     outs_cla = attack_out_dict['PGD-20'][labels == i]
    #     sns.histplot(outs_cla[:,i], kde=False, ax=axes[i // 5, i % 5])
    # _savefig('./pgd_out_logits.jpg')
    #
    # fig, axes = plt.subplots(2, 5, figsize=(25, 8))
    # for i in range(num_classes):
    #     outs_cla = _softmax(clean_outs[labels != i])
    #     sns.histplot(outs_cla[:,i], kde=False, ax=axes[i // 5, i % 5], cbar_kws=dict(alpha=0.25))
    #     outs_cla = _softmax(attack_out_dict['PGD-20'][labels != i])
    #     sns.histplot(outs_cla[:,i], kde=False, ax=axes[i // 5, i % 5], color='crimson', cbar_kws=dict(alpha=0.25))
    # _savefig('./pgd_out_other_softmaxs.jpg')
    #
    # fig, axes = plt.subplots(2, 5, figsize=(25, 8))
    # for i in range(num_classes):
    #     outs_cla = clean_outs[labels != i]
    #     sns.histplot(outs_cla[:,i], kde=False, ax=axes[i // 5, i % 5], cbar_kws=dict(alpha=0.25))
    #     outs_cla = attack_out_dict['PGD-20'][labels != i]
    #     sns.histplot(outs_cla[:,i], kde=False, ax=axes[i // 5, i % 5], color='crimson', cbar_kws=dict(alpha=0.25))
    # _savefig('./pgd_out_other_logits.jpg')

class CountMeter(object):
    def __init__(self, num_classes, non_zero=True, save_raw=False):
        self.num_classes = num_classes
        self.non_zero = non_zero
        self.save_raw = save_raw
        self.reset()

    def reset(self):
        self.n = 0
        # count non-zero
        self.n_per_class = np.zeros(self.num_classes)
        self.sum_values = np.zeros(self.num_classes)
        self.avg_values = np.zeros(self.num_classes)
        if self.save_raw:
            self.raw_values = None

    def update(self, data, target=None):
        self.n += data.shape[0]

        # data (n,), target=(n,)
        if len(data.shape) == 1:
            assert target is not None
            self.sum_values[target] += data
            for i in range(self.num_classes):
                self.n_per_class[i] += (target == i).sum()

        # data (n, dim), target=(n, dim) / None
        else:
            self.sum_values += np.sum(data, dim=0)
            if target is None:
                self.n_per_class += np.sum(data>0, dim=0)
            else:
                self.n_per_class += np.sum(target, dim=0)

        if self.non_zero:
            self.avg_values = self.sum_values / self.n_per_class
        else:
            self.avg_values = self.sum_values / self.n

        if self.save_raw:
            if self.raw_values is None:
                self.raw_values = data
            else:
                self.raw_values = np.vstack((self.raw_values, data))

    def save_data(self):
        raise NotImplementedError

def _savefig(name='./image.jpg', dpi=300):
    plt.savefig(name, dpi=dpi)
    print('Image saved at {}'.format(name))

def _softmax(logits, dim=-1):
    exp = np.exp(logits)
    sum = np.sum(exp, axis=dim, keepdims=True)
    return exp / sum

def _onehot(labels, num_classes):
    labels = np.asarray(labels, dtype=np.uint8)
    out = np.zeros((len(labels), num_classes), dtype=np.uint8)
    idx = np.arange(len(labels))
    out[idx, labels] = 1
    return out


if __name__ == '__main__':
    pass
