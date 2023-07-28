import argparse
from functools import partial
import os
import sys
import cv2
from PIL import Image
import numpy as np
from attrdict import AttrDict
from time import time
import tritonclient.grpc as grpcclient
import tritonclient.grpc.model_config_pb2 as mc
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException
from tritonclient.utils import triton_to_np_dtype
import random

import torch

import torchvision
#c_c_c=0
ffflist=[]
if sys.version_info >= (3, 0):
    import queue
else:
    import Queue as queue
cls = ['face', 'mask-1', 'mask-2', 'mask-3', 'eyewear', 'person', 'helmet', 'weapons', 'mas1']
class UserData:

    def __init__(self):
        self._completed_requests = queue.Queue()


# Callback function used for async_stream_infer()
def completion_callback(user_data, result, error):
    # passing error raise and handling out
    user_data._completed_requests.put((result, error))


FLAGS = None

def parse_model(model_metadata, model_config):
    """
    Check the configuration of a model to make sure it meets the
    requirements for an image classification network (as expected by
    this client)
    """
    if len(model_metadata.inputs) != 1:
        raise Exception("expecting 1 input, got {}".format(
            len(model_metadata.inputs)))
    if len(model_metadata.outputs) != 1:
        raise Exception("expecting 1 output, got {}".format(
            len(model_metadata.outputs)))

    if len(model_config.input) != 1:
        raise Exception(
            "expecting 1 input in model configuration, got {}".format(
                len(model_config.input)))

    input_metadata = model_metadata.inputs[0]
    input_config = model_config.input[0]
    output_metadata1 = model_metadata.outputs[0]
    names=[output_metadata1.name]
    print(input_metadata)
    if output_metadata1.datatype != "FP32":
        raise Exception("expecting output datatype to be FP32, model '" +
                        model_metadata.name + "' output type is " +
                        output_metadata.datatype)

    # Output is expected to be a vector. But allow any number of
    # dimensions as long as all but 1 is size 1 (e.g. { 10 }, { 1, 10
    # }, { 10, 1, 1 } are all ok). Ignore the batch dimension if there
    # is one.
    output_batch_dim = (model_config.max_batch_size > 0)
    non_one_cnt = 0

    # Model input must have 3 dims, either CHW or HWC (not counting
    # the batch dimension), either CHW or HWC
    input_batch_dim = (model_config.max_batch_size > 0)
    expected_input_dims = 3 + (1 if input_batch_dim else 0)

    #if input_config.format == mc.ModelInput.FORMAT_NHWC:
    h = input_metadata.shape[1 if input_batch_dim else 3]
    w = input_metadata.shape[2 if input_batch_dim else 2]
    c = input_metadata.shape[3 if input_batch_dim else 1]
    print("h,c,w",h,c,w)
    #else:
    #    c = input_metadata.shape[1 if input_batch_dim else 0]
    #    h = input_metadata.shape[2 if input_batch_dim else 1]
    #    w = input_metadata.shape[3 if input_batch_dim else 2]

    return (model_config.max_batch_size, input_metadata.name,
            names,c, h, w, input_config.format,
            input_metadata.datatype)


def preprocess(img, format, dtype, c, h, w, scaling, protocol):
    """
    Pre-process an image to meet the size, type and format
    requirements specified by the parameters.
    """
    print("h,c,w",h,c,w)
    if c == 1:
        sample_img = img.convert('L')
    else:
        sample_img = img.convert('RGB')
    
    resized_img = sample_img.resize((h, w), Image.BILINEAR)
    resized = np.array(resized_img)
    if resized.ndim == 2:
        resized = resized[:, :, np.newaxis]
    print("resized",resized.shape)
    npdtype = triton_to_np_dtype(dtype)
    typed = resized.astype(npdtype)
    typed /= 255.0
    typed = np.ascontiguousarray(typed)
    print(typed.shape)
    if scaling == 'INCEPTION':
        scaled = (typed / 127.5) - 1
    elif scaling == 'VGG':
        if c == 1:
            scaled = typed - np.asarray((128,), dtype=npdtype)
        else:
            scaled = typed - np.asarray((123, 117, 104), dtype=npdtype)
    else:
        scaled = typed

    # Swap to CHW if necessary
    ordered = np.transpose(scaled, (2, 0, 1))
    ordered = np.expand_dims(ordered, axis=0)
    #ordered = ordered.numpy.reshape(1,3,320,320)
    #print("h,c,w",h,c,w)
    print("ordered",ordered.shape)
    return ordered


def non_max_suppression(prediction, conf_thres=0.1, iou_thres=0.6, merge=False, classes=None, agnostic=False):
    """Performs Non-Maximum Suppression (NMS) on inference results
    Returns:
        detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """
    prediction=torch.from_numpy(prediction)
    if prediction.dtype is torch.float16:
        prediction = prediction.float()  # to FP32
    nc = prediction[0].shape[1] - 8  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)

    output = [None] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue
        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        n = x.shape[0]  # number of boxes
        if not n:
            continue

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.boxes.nms(boxes, scores, iou_thres)
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            try:  # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
                weights = iou * scores[None]  # box weights
                x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
                if redundant:
                    i = i[iou.sum(1) > 1]  # require redundancy
            except:  # possible CUDA error https://github.com/ultralytics/yolov3/issues/1139
                pass
        
        output[xi] = x[i]
    return output

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def postprocess(results, inputs, output_name):
    """
    Post-process results to show classifications.
    """
    aa=results.as_numpy(output_name[0])
    correct_nms = non_max_suppression(aa, conf_thres=0.5, iou_thres=0.6, merge=False, classes=None, agnostic=False)
    correct_nms = [item.numpy() for item in correct_nms if item is not None]

    return correct_nms  



def requestGenerator(batched_image_data, input_name, output_name, dtype, FLAGS):
    protocol = FLAGS.protocol.lower()

    if protocol == "grpc":
        client = grpcclient
    else:
        client = httpclient

    # Set the input data
    inputs = [client.InferInput(input_name, batched_image_data.shape, dtype)]
    inputs[0].set_data_from_numpy(batched_image_data)

    outputs = [
        client.InferRequestedOutput(output_name[0])
    ]

    yield inputs, outputs, FLAGS.model_name, FLAGS.model_version


def convert_http_metadata_config(_metadata, _config):
    _model_metadata = AttrDict(_metadata)
    _model_config = AttrDict(_config)

    return _model_metadata, _model_config


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v',
                        '--verbose',
                        action="store_true",
                        required=False,
                        default=False,
                        help='Enable verbose output')
    parser.add_argument('-a',
                        '--async',
                        dest="async_set",
                        action="store_true",
                        required=False,
                        default=False,
                        help='Use asynchronous inference API')
    parser.add_argument('--streaming',
                        action="store_true",
                        required=False,
                        default=False,
                        help='Use streaming inference API. ' +
                        'The flag is only available with gRPC protocol.')
    parser.add_argument('-m',
                        '--model-name',
                        type=str,
                        required=True,
                        help='Name of model')
    parser.add_argument('-x',
                        '--model-version',
                        type=str,
                        required=False,
                        default="",
                        help='Version of model. Default is to use latest version.')
    parser.add_argument('-b',
                        '--batch-size',
                        type=int,
                        required=False,
                        default=1,
                        help='Batch size. Default is 1.')
    parser.add_argument('-c',
                        '--classes',
                        type=int,
                        required=False,
                        default=3,
                        help='Number of class results to report. Default is 80.')
    parser.add_argument('-s',
                        '--scaling',
                        type=str,
                        choices=['NONE', 'INCEPTION', 'VGG'],
                        required=False,
                        default='NONE',
                        help='Type of scaling to apply to image pixels. Default is NONE.')
    parser.add_argument('-u',
                        '--url',
                        type=str,
                        required=False,
                        default='localhost:8000',
                        help='Inference server URL. Default is localhost:8000.')
    parser.add_argument('-i',
                        '--protocol',
                        type=str,
                        required=False,
                        default='HTTP',
                        help='Protocol (HTTP/gRPC) used to communicate with ' +
                        'the inference service. Default is HTTP.')
    parser.add_argument('image_filename',
                        type=str,
                        nargs='?',
                        default=None,
                        help='Input image / Input folder.')
    FLAGS = parser.parse_args()
    
    sstime= time()

    if FLAGS.streaming and FLAGS.protocol.lower() != "grpc":
        raise Exception("Streaming is only allowed with gRPC protocol")

    try:
        if FLAGS.protocol.lower() == "grpc":
            # Create gRPC client for communicating with the server
            triton_client = grpcclient.InferenceServerClient(
                url=FLAGS.url, verbose=FLAGS.verbose)
        else:
            # Specify large enough concurrency to handle the
            # the number of requests.
            concurrency = 2 if FLAGS.async_set else 1
            triton_client = httpclient.InferenceServerClient(
                url=FLAGS.url, verbose=FLAGS.verbose, concurrency=concurrency)
    except Exception as e:
        print("client creation failed: " + str(e))
        sys.exit(1)

    # Make sure the model matches our requirements, and get some
    # properties of the model that we need for preprocessing
    try:
        model_metadata = triton_client.get_model_metadata(
            model_name=FLAGS.model_name, model_version=FLAGS.model_version)
    except InferenceServerException as e:
        print("failed to retrieve the metadata: " + str(e))
        sys.exit(1)

    try:
        model_config = triton_client.get_model_config(
            model_name=FLAGS.model_name, model_version=FLAGS.model_version)
    except InferenceServerException as e:
        print("failed to retrieve the config: " + str(e))
        sys.exit(1)

    if FLAGS.protocol.lower() == "grpc":
        model_config = model_config.config
    else:
        model_metadata, model_config = convert_http_metadata_config(
            model_metadata, model_config)

    max_batch_size, input_name, output_name, c, h, w, format, dtype = parse_model(
        model_metadata, model_config)
    
    filenames = []
    if os.path.isdir(FLAGS.image_filename):
        osl= os.listdir(FLAGS.image_filename)
        for f in osl:
            if os.path.isfile(os.path.join(FLAGS.image_filename, f)):
                filenames.append(os.path.join(FLAGS.image_filename, f))      
    else:
        filenames = [
            FLAGS.image_filename,
        ]

    filenamess= filenames[:50]
    filenamess.sort()
    ttime=[]
    batchtiming=[]
    loadtime= time()- sstime
    tor1= 0
    tor2= FLAGS.batch_size
    if len(filenamess)%FLAGS.batch_size==0:
        iterfiles= len(filenamess)//FLAGS.batch_size
    else:
        iterfiles= len(filenamess)//FLAGS.batch_size +1

    result_list= []
    for filebatch in range(int(iterfiles)):
        batchtime= time()
        filenames= filenamess[tor1:tor2]
        if len(filenamess)- tor2 < FLAGS.batch_size:
            tor1+= FLAGS.batch_size
            tor2= len(filenamess)
        else:
            tor1+=FLAGS.batch_size
            tor2+=FLAGS.batch_size

        # Preprocess the images into input data according to model
        # requirements
        image_data = []
        for filename in filenames:
            img = Image.open(filename)
            image_data.append(
                preprocess(img, format, dtype, c, h, w, FLAGS.scaling,
                           FLAGS.protocol.lower()))

        # Send requests of FLAGS.batch_size images. If the number of
        # images isn't an exact multiple of FLAGS.batch_size then just
        # start over with the first images until the batch is filled.
        requests = []
        responses = []
        result_filenames = []
        request_ids = []
        image_idx = 0
        last_request = False
        user_data = UserData()

        # Holds the handles to the ongoing HTTP async requests.
        async_requests = []

        sent_count = 0

        if FLAGS.streaming:
            triton_client.start_stream(partial(completion_callback, user_data))
        
        while not last_request:
            
            input_filenames = []
            repeated_image_data = []

            for idx in range(FLAGS.batch_size):
                input_filenames.append(filenames[image_idx])
                repeated_image_data.append(image_data[image_idx])
                image_idx = (image_idx + 1) % len(image_data)
                if image_idx == 0:
                    last_request = True

            if max_batch_size > 0:
                batched_image_data = np.stack(repeated_image_data, axis=0)
            else:
                batched_image_data = repeated_image_data[0]
            
            ffps=[]
            # Send request
            try:
                for inputs, outputs, model_name, model_version in requestGenerator(
                        batched_image_data, input_name, output_name, dtype, FLAGS):
                    sent_count += 1
                    stime= time()
                    if FLAGS.streaming:
                        triton_client.async_stream_infer(
                            FLAGS.model_name,
                            inputs,
                            request_id=str(sent_count),
                            model_version=FLAGS.model_version,
                            outputs=outputs)
                    elif FLAGS.async_set:
                        if FLAGS.protocol.lower() == "grpc":
                            triton_client.async_infer(
                                FLAGS.model_name,
                                inputs,
                                partial(completion_callback, user_data),
                                request_id=str(sent_count),
                                model_version=FLAGS.model_version,
                                outputs=outputs)
                        else:
                            async_requests.append(
                                triton_client.async_infer(
                                    FLAGS.model_name,
                                    inputs,
                                    request_id=str(sent_count),
                                    model_version=FLAGS.model_version,
                                    outputs=outputs))
                    else:
                        responses.append(
                            triton_client.infer(FLAGS.model_name,
                                                inputs,
                                                request_id=str(sent_count),
                                                model_version=FLAGS.model_version,
                                                outputs=outputs))
                    #etime= time()
                    ttime.append(time() -stime)
                    ffps.append(concurrency*FLAGS.batch_size/(time()-stime))
            except InferenceServerException as e:
                print("inference failed: " + str(e))
                if FLAGS.streaming:
                    triton_client.stop_stream()
                sys.exit(1)
        #timetaken= time()-stime
        if FLAGS.streaming:
            triton_client.stop_stream()

        if FLAGS.protocol.lower() == "grpc":
            if FLAGS.streaming or FLAGS.async_set:
                processed_count = 0
                while processed_count < sent_count:
                    (results, error) = user_data._completed_requests.get()
                    processed_count += 1
                    if error is not None:
                        print("inference failed: " + str(error))
                        sys.exit(1)
                    responses.append(results)
        else:
            if FLAGS.async_set:
                # Collect results from the ongoing async requests
                # for HTTP Async requests.
                for async_request in async_requests:
                    responses.append(async_request.get_result())
        
        for response in responses:
            if FLAGS.protocol.lower() == "grpc":
                this_id = response.get_response().id
            else:
                this_id = response.get_response()["id"]
            ret= postprocess(response, inputs, output_name)
            result_list.append(ret)
        batchtiming.append((time()-batchtime))


    timetaken= time()-sstime            #total time taken for the script to run
    fps = FLAGS.batch_size* len(ttime)/(sum(ttime))  #+ loadtime)   #fps is total frames per sum of time taken for each request
    abs_fps= FLAGS.batch_size* len(ttime)/sum(batchtiming)                              #   and loading time(including preprocessing)
    

    print(len(result_list))
    print(result_list)
    with open("temp.txt","a") as fp:
        for res in range(len(result_list)):
            print("Result Iteration: {}\n".format(res),file=fp)
            print("{}\n\n\n".format(result_list[res]),file=fp)
    print ("\nTime Taken: {}\n Number of Frames including repeated: {}\n FPS: {}".format(timetaken, FLAGS.batch_size* len(ttime), fps))
    print("Absolute FPS: ", abs_fps)
    print(len(ttime))
