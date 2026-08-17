from onnxruntime.quantization import quantize_dynamic, QuantType

quantize_dynamic(
    model_input="ml_models/plant_disease_unet.onnx",
    model_output="ml_models/plant_disease_unet_quantized.onnx",
    weight_type=QuantType.QUInt8
)

print("Done. Compare file sizes in ml_models/ to see the reduction.")