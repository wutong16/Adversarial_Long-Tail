from __future__ import print_function
from utils.data_utils import save_csv
import mmcv
import numpy as np
from autoattack import AutoAttack
from attacks.pgd_attack import eval_metrics


def eval_auto_attack(model, device, cfgs, logger, test_loader, individual=False, print_freq=20, mode='test', train_val=False):
    """
    evaluate model by Auto Attack
    """
    logger.info("Evaluating Auto Attack!")
    model.eval()
    attacks_to_run = ['apgd-ce', 'apgd-t']#, 'apgd-dlr', 'fab-t']
    adversary = AutoAttack(model, norm='Linf', eps=cfgs.test_epsilon, version='standard', verbose=False)
    adversary.attacks_to_run = attacks_to_run
    # fab-t and square-attack won't contribute much according to our experiments,
    # so we only use the first two for AA evaluation as also stated in our paper.
    total_batches = len(test_loader)
    all_labels = []
    all_outputs = dict()

    for i, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        outputs = dict()

        if individual:
            # set to what we care
            adversary.attacks_to_run = attacks_to_run
            x_adv_dict = adversary.run_standard_evaluation_individual(data, target, bs=len(data))
            # set back to standard
            adversary.attacks_to_run = ['apgd-ce', 'apgd-t', 'fab-t', 'square']
            for attack, x_adv in x_adv_dict.items():
                outputs.update({ attack : model(x_adv).detach().cpu().numpy()})
        else:
            x_adv = adversary.run_standard_evaluation(data, target, bs=len(data))
            outputs.update({ 'standard' : model(x_adv).detach().cpu().numpy()})

        all_labels.extend(target.cpu().detach().numpy().tolist())

        for key in outputs.keys():
            if key not in all_outputs.keys():
                all_outputs.update({key: []})
                all_outputs[key].append(outputs[key])
            else:
                all_outputs[key].append(outputs[key])

        if i == 5 or (i+1) % print_freq == 0:
            logger.info("[{}/{}] batches finished".format(i+1, total_batches))
            for key, data in all_outputs.items():
                if 'features' in key or 'success' in key:
                    continue
                data = np.vstack(data)
                accuracy, precision, recall, f_score, true_sum = eval_metrics(data, all_labels)
                logger.info("metric: {} | accuracy: {:.4f} | precision: {:.4f} | recall: {:.4f} | f_score: {:.4f} | ".format(
                    key, accuracy, np.mean(precision), np.mean(recall), np.mean(f_score)))
            if train_val:
                return

    # save metrics into .csv
    csv_name = '{}_all_results.csv'.format(cfgs.dataset)
    save_csv(csv_name, [[cfgs.model_path], [cfgs.remark]], devide=False)
    for key, data in all_outputs.items():
        save_data = [[' '], [key]]
        if 'success' in key:
            data = np.hstack(data)
        else:
            data = np.vstack(data)
        all_outputs[key] = data
        if 'features' in key or 'success' in key:
            continue
        accuracy, precision, recall, f_score, true_sum = eval_metrics(data, all_labels)
        # use group evaluation for CIFAR100 and ImageNet
        if recall.shape[0] >= 100:
            g_recall = np.reshape(recall, (10, -1)).sum(-1)
            g_precision = np.reshape(precision, (10, -1)).sum(-1)
            g_f_score = np.reshape(f_score, (10, -1)).sum(-1)
            save_data.extend([
                ['accuracy', accuracy],
                ['g_recall'], g_recall.tolist(),
                ['g_precision'], g_precision.tolist(),
                ['g_f_score'], g_f_score.tolist()
            ])
        # save original per-class metric
        save_data.extend([
            ['accuracy', accuracy],
            ['recall'], recall.tolist(),
            ['precision'], precision.tolist(),
            ['f_score'], f_score.tolist()
        ])
        save_csv(csv_name, save_data, devide=False)

        logger.info("metric: {} | accuracy: {:.4f} | precision: {:.4f} | recall: {:.4f} | f_score: {:.4f} | ".format(
            key, accuracy, np.mean(precision), np.mean(recall), np.mean(f_score)))
        for name, value in zip(['precision', 'recall', 'f_score'], [precision, recall, f_score]):
            print(name, end=' | ')
            for v in value:
                print("{:.2f}".format(v*100), end=' ')
            print()
    logger.info("[Remarks] {} | End of evaluation,model path {}".format(cfgs.remark, cfgs.model_path))

    for name, param in model.classifier.state_dict().items():
        if 'weight' in name:
            print('save classifier weight from module <{}> !'.format(name))
            all_outputs.update(CLASSIFIER_weight=model.classifier.state_dict()[name].clone().detach().cpu().numpy())

    all_outputs.update(LABLES=all_labels)

    mmcv.dump(all_outputs, cfgs.model_path + '.AA.{}.pkl'.format(mode))
    print('Data saved at {}'.format(cfgs.model_path + 'AA.{}.pkl'.format(mode)))

    return
