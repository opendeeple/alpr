from nn.backbones import ResNet18

backbones = {
  'resnet18': ResNet18
}

def build(backbone='resnet18', **kwargs):
  class_type = backbones.get(backbone)
  return class_type(**kwargs)
