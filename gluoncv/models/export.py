import gluoncv
import mxnet as mx


models = [
            'cifar_resnet20_v1',
            'cifar_resnet56_v1',
            'cifar_resnet110_v1',
            'cifar_resnet20_v2',
            'cifar_resnet56_v2',
            'cifar_resnet110_v2',
            'cifar_wideresnet16_10',
            #'cifar_wideresnet28_10',
            #'cifar_wideresnet40_8',
            #'cifar_resnext29_16x64d'
        ]

def export(model_name):
    h, w = 32, 32
    net = gluoncv.model_zoo.get_model(model_name, pretrained=True)
    net.hybridize()
    net.forward(mx.nd.zeros((1, 3, h, w)))
    net.export(model_name)


if __name__ == '__main__':
    for model in models:
        print(model)
        export(model)
