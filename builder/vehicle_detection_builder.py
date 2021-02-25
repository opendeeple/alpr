from nn import VehicleDetection
from builder import backbone_builder

def build():
  backbone = backbone_builder.build(backbone='resnet18')
  vehicle_detection = VehicleDetection(
    backbone=backbone
  )
  return backbone_builder
