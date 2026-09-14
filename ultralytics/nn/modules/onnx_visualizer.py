import onnx

model = onnx.load("/mnt/nfs/home/dbenitom/pruebas/vincxr/history/100epochas_linearprobing_augmentsfinales_v1/yolo-dino-exp635/weights/best.onnx")
print(onnx.helper.printable_graph(model.graph))