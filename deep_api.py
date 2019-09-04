import json
import numpy as np
from preprocess import preprocess_gray_images, preprocess_bgr, preprocess_unet
from grpc import insecure_channel
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from tensorflow.compat.v1 import make_tensor_proto

_DEFAULT_HOST = "localhost"
_DEFAULT_PORT = "8500"

# docker run -p 8500:8500 --name ocr-mobile-card-service -dit --restart always -t ocr-mobile-card:latest-avx2 --model_config_file=/models/models.conf

class DeepPostMan:
    def __init__(self, host=_DEFAULT_HOST, port=_DEFAULT_PORT, warm_up_model_server=True, *args, **kwargs):
        self.protocol = "GRPC"
        self.host = host
        self.port = port
        for attr_name in kwargs.keys():
            setattr(self, attr_name, kwargs[attr_name])
        
        # Khởi tạo các GRPC Object
        self.grpc_channel = insecure_channel("{}:{}".format(self.host, self.port))
        self.stub = prediction_service_pb2_grpc.PredictionServiceStub(self.grpc_channel)

        # Xử lý warm up model server. Sau khi khởi động model server, lần đầu tiên request sẽ rất chậm, 
        # do đó có thể ảnh hưởng tới xử lý, cần warm up trước để có thể đảm bảo request với tốc độ nhanh từ lần đầu tiên

        if warm_up_model_server:
            warmup_data = np.zeros((1,28,28,3), dtype=np.float32)
            self.predict_digits(warmup_data)

            warmup_data = np.zeros((1,224,64,3), dtype=np.float32)
            self.denoise_images(warmup_data)

    def __del__(self):
        self.grpc_channel.close()

    def predict_digits(self, digit_images, timeout_per_request=10):
        # digit_images = np.stack(digit_images)
        # assert digit_images.ndim == 4, "digit_images must has shape [Batch_size, height, width, color-channel]"
        preprocessed_bgr_images = [preprocess_bgr(img) for img in digit_images.copy()]

        # Function preprocess_gray_images return pair [background, no_background]
        preprocessed_gray_bg_images, preprocessed_gray_nobg_images = preprocess_gray_images(digit_images)

        batch_size = len(digit_images)

        request_data_list = [
            # Xử lý ảnh màu
            make_tensor_proto(np.array(preprocessed_bgr_images).astype(np.float32), shape=[batch_size,28, 28, 3]), 
            # Xử lý ảnh trắng đen
            make_tensor_proto(np.array(preprocessed_gray_bg_images).astype(np.float32), shape=[batch_size,28, 28, 1]),
            # Xử lý ảnh trắng đen, sử dụng OTSU threshold
            make_tensor_proto(np.array(preprocessed_gray_nobg_images).astype(np.float32), shape=[batch_size,28, 28, 1])
        ]

        # Future là loại response không lấy kết quả ngay, mà thực hiện submit tất cả các request, sau đó mới quay lại lấy kết quả sau
        future_responses = list()
        request = predict_pb2.PredictRequest()
        request.model_spec.signature_name = 'serving_default'
        # Submit request ảnh màu
        request.model_spec.name = 'digit-rgb'
        request.inputs['input_images'].CopyFrom(request_data_list[0])
        future_resp_tmp = self.stub.Predict.future(request, timeout_per_request)
        future_responses.append(future_resp_tmp)

        # Submit request ảnh gray có background
        request.model_spec.name = 'digit-gray-bg'
        request.inputs['input_images'].CopyFrom(request_data_list[1])
        future_resp_tmp = self.stub.Predict.future(request, timeout_per_request)
        future_responses.append(future_resp_tmp)

        # Submit request ảnh gray tiền xử lý otsu threshold
        request.model_spec.name = 'digit-gray-otsu'
        request.inputs['input_images'].CopyFrom(request_data_list[2])
        future_resp_tmp = self.stub.Predict.future(request, timeout_per_request)
        future_responses.append(future_resp_tmp)

        # Chờ lấy kết quả
        response_objects = [r.result() for r in future_responses]

        # Decode kết quả trả về
        final_result = list()
        for result_obj in response_objects:
            # Decode tensor shape
            tf_shape_grpc_obj_list = result_obj.outputs['predictions'].tensor_shape.dim
            resp_tensor_shape = [tmp.size for tmp in tf_shape_grpc_obj_list]
            
            # Decode response result
            model_result_flatten = np.array(result_obj.outputs['predictions'].float_val._values)
            model_result = np.reshape(model_result_flatten, resp_tensor_shape)
            final_result.append(model_result)
        final_result = np.stack(final_result)

        return final_result
    
    def denoise_images(self, images, timeout_per_request=10):
        processed_images = [preprocess_unet(img) for img in images]
        request_data_list = [make_tensor_proto(np.array([pr_img]).astype(np.float32), shape=[1,224, 64, 3]) for pr_img in processed_images]
        future_responses = list()
        request = predict_pb2.PredictRequest()
        request.model_spec.signature_name = 'serving_default'
        request.model_spec.name = 'unet'
        for request_data in request_data_list:
            request.inputs['input_images'].CopyFrom(request_data)
            future_responses.append(self.stub.Predict.future(request, timeout_per_request))
        
        # Chờ lấy kết quả
        response_objects = [r.result() for r in future_responses]
        # Decode kết quả trả về
        final_result = list()
        for result_obj in response_objects:
            # Decode tensor shape
            tf_shape_grpc_obj_list = result_obj.outputs['predictions'].tensor_shape.dim
            resp_tensor_shape = [tmp.size for tmp in tf_shape_grpc_obj_list]
            
            # Decode response result
            model_result_flatten = np.array(result_obj.outputs['predictions'].float_val._values)
            model_result = np.reshape(model_result_flatten, resp_tensor_shape)
            final_result.append(np.squeeze(model_result, axis=0))
        final_result = np.stack(final_result)
        return final_result

if __name__ == "__main__":
    import time
    postman = DeepPostMan()
    img_list = ["1.png", "4.png"]
    st = time.time()
    # for i in range(10):
    predictions = postman.predict_digits(img_list*30)
    
    print("Wait time {}s".format((time.time()-st)))
    print("Prediction\n{}".format(predictions.shape))
    