import torch.nn as nn
from models.backbones.wideresnet import WideResNet
from models.classifiers.basic_classifiers import FC_Classifier, Cos_Classifier, \
    Dot_Classifier, PostNorm_Classifier, TDESim_Classifier,\
    PostProc_Classifier, CDT_Classifier, CosPlus_Classifier, Cos_Center_Classifier
from models.classifiers.other_classifier import Mix_Classifier, MC_Classifier, BN_Classifier
from methods.losses_inclass import Losses


class Networks(nn.Module):
    def __init__(self, cfgs, num_classes, samples_per_cls=None):
        super(Networks, self).__init__()
        self.num_classes = num_classes
        self.samples_per_cls = samples_per_cls
        self.backbone_with_fc = cfgs.classifier is None
        self.backbone = self.build_backbone(cfgs)
        self.epoch = 0

        if not self.backbone_with_fc:
            self.classifier = self.build_classifier(cfgs)
        if hasattr(self.classifier, 'loss'):
            self.loss = self.classifier.loss
            print('use the loss by the classifier')
        else:
            self.loss = Losses(samples_per_cls, num_classes, cfgs.loss_opt)

        if hasattr(self.classifier, 'adv_loss'):
            self.adv_loss = self.classifier.adv_loss
            print('use the adv loss by the classifier for the inner maximum')
        else:
            self.adv_loss = Losses(samples_per_cls, num_classes, cfgs.adv_loss_opt)

        self.nat_loss = Losses(samples_per_cls, num_classes, cfgs.nat_loss_opt)

    def build_backbone(self, cfgs):
        name = cfgs.backbone
        print('>> Build backbone {}'.format(name))
        backbone_opt = getattr(cfgs, 'backbone_opt', dict())
        for k, v in backbone_opt.items():
            print('{} : {}'.format(k,v))

        if name == 'WideResNet':
            net = WideResNet(num_classes=self.num_classes, use_fc=self.backbone_with_fc, **backbone_opt)
        else:
            raise NameError

        return net

    def build_classifier(self, cfgs):
        name = cfgs.classifier
        classifier_opt = getattr(cfgs, 'classifier_opt', dict())
        for k, v in classifier_opt.items():
            print('{} : {}'.format(k,v))
        print('>> Build classifier {}'.format(name))

        if 'FC' in name:
            if cfgs.loss_opt  is not None:
                focal_init = "focal" in cfgs.loss_opt
            else:
                focal_init = False
            net = FC_Classifier(self.num_classes, samples_per_cls=self.samples_per_cls, focal_init=focal_init)
        elif 'Cos' == name:
            net = Cos_Classifier(self.num_classes, **classifier_opt)
        elif 'Dot' in name:
            net = Dot_Classifier(self.num_classes)
        elif 'PostNorm' in name:
            net = PostNorm_Classifier(self.num_classes, **classifier_opt)
        elif 'CDT' in name:
            net = CDT_Classifier(self.num_classes, samples_per_cls=self.samples_per_cls, **classifier_opt)
        elif 'TDESim' in name:
            net = TDESim_Classifier(self.num_classes, samples_per_cls=self.samples_per_cls, **classifier_opt)
        elif 'CosPlus' in name:
            net = CosPlus_Classifier(self.num_classes, samples_per_cls=self.samples_per_cls, **classifier_opt)
        elif 'PostProc' in name:
            net = PostProc_Classifier(self.num_classes, samples_per_cls=self.samples_per_cls, **classifier_opt)
        # elif 'Cos_Center' in name:
        #     net = Cos_Center_Classifier(self.num_classes)
        # elif 'MC' in name:
        #     net = MC_Classifier(self.num_classes, samples_per_cls=self.samples_per_cls, num_centroids=1)
        # elif 'Mix' in name:
        #     net = Mix_Classifier(self.num_classes, alpha=cfgs.alpha, samples_per_cls=self.samples_per_cls)
        # elif 'BN' in name:
        #     net = BN_Classifier(self.num_classes, samples_per_cls=self.samples_per_cls, **classifier_opt)
        else:
            raise NameError

        return net

    def forward(self, x):
        out = self.backbone(x)
        if not self.backbone_with_fc:
            out = self.classifier(out)
        return out

    def on_epoch(self):
        if hasattr(self.classifier, 'on_epoch'):
            self.classifier.on_epoch()
            print('Classifier operation on epoch')
        if hasattr(self.backbone, 'on_epoch'):
            self.backbone.on_epoch()
            print('Backbone operation on epoch')