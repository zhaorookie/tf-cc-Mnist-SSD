#include <stdio.h>
#include <string.h>
#include <assert.h>
#include "opencv2/opencv.hpp"
#include "my_interface.h"

#define BATH_SIZE 1

int main()
{   
    //申请输入输出内存
    tensor_params_array_t in_tensor_params_ar = {0};
    tensor_params_array_t out_tensor_params_ar = {0};
    tensor_array_t *input_tensor_array = NULL;
    tensor_array_t *output_tensor_array = NULL;

    //输入Tensor数组参数设置
    in_tensor_params_ar.nArraySize = 1;
    //serving_default是通过 saved_model_cli show --dir saved_model/1 --all查看后获得，下同
    //查看时出现 -1 ，表示这个位置的值是任意的
    strcpy(in_tensor_params_ar.pcSignatureDef, "serving_default");
    in_tensor_params_ar.pTensorParamArray = (tensor_params_t *)malloc(
        in_tensor_params_ar.nArraySize * sizeof(tensor_params_t));
    tensor_params_t *cur_in_tensor_params = &(in_tensor_params_ar.pTensorParamArray[0]);
    cur_in_tensor_params->nDims = 4; //(-1, 28, 28, 1)
    cur_in_tensor_params->type = DT_FLOAT;
    cur_in_tensor_params->pShape[0] = BATH_SIZE;
    cur_in_tensor_params->pShape[1] = 28;   // H
    cur_in_tensor_params->pShape[2] = 28;   // W
    cur_in_tensor_params->pShape[3] = 1;    // channel
    strcpy(cur_in_tensor_params->aTensorName, "conv2d_input");

    //输出Tensor数组参数设置，同样通过saved_model_cli show可得
    out_tensor_params_ar.nArraySize = 1;  //这里根据自己想要查看的信息种类进行设置，比如我只想看分类这一种信息，则设置1
    strcpy(out_tensor_params_ar.pcSignatureDef, "serving_default");
    out_tensor_params_ar.pTensorParamArray = (tensor_params_t *)malloc(
        out_tensor_params_ar.nArraySize * sizeof(tensor_params_t));
    tensor_params_t *cur_out_tensor_params = &(out_tensor_params_ar.pTensorParamArray[0]);
    cur_out_tensor_params->type = DT_FLOAT;
    cur_out_tensor_params->nDims = 2;  //(-1, 10)
    cur_out_tensor_params->pShape[0] = BATH_SIZE;
    cur_out_tensor_params->pShape[1] = 10;
    strcpy(cur_out_tensor_params->aTensorName, "dense_1");

    //调用API申请Tensor数组内存
    if (SUCCESS != init_tensors(&in_tensor_params_ar, &out_tensor_params_ar,
                                &input_tensor_array, &output_tensor_array))
    {
        printf("Open tensor memory error\n");
    }

    //设置模型加载参数
    model_params_t tModelParams = {0};
    model_handle_t tModelHandel = {0};
    tModelParams.cpu_or_gpu = 1;

    strcpy(tModelParams.visibleCard, "0");
    //strcpy(tModelParam.visibleCard, "0,1");
    tModelParams.gpu_id = 0;

    tModelParams.gpu_memory_faction = 0.5;

    // tModelParams.bIsCipher = true;
    // strcpy(tModelParams.model_path, "models/tf_mnist_enc/1");

    tModelParams.bIsCipher = false;
    strcpy(tModelParams.model_path, "models/mnist_export_tf2/saved_model/1");

    // 同样通过saved_model_cli show可得
    strcpy(tModelParams.paModelTagSet, "serve");

    //调用API装载模型
    if (SUCCESS != load_model(&tModelParams, input_tensor_array, output_tensor_array, &tModelHandel) != SUCCESS)
    {
        printf("Load Model error!!!\n");
    }

    //填充input tensor array的数据
    tensor_t *cur_input_tensor =&(input_tensor_array->pTensorArray[0]);
    tensor_params_t* cur_input_tensor_info = cur_input_tensor->pTensorInfo;
    cv::Mat mnistImage = cv::imread("test_data/mnist_img_0.png", cv::IMREAD_GRAYSCALE);
    int img_size = mnistImage.rows * mnistImage.cols * mnistImage.channels();

    std::cout << "Cur tensor value length: " <<cur_input_tensor_info->nLength <<std::endl;
    assert(img_size == cur_input_tensor_info->nLength/sizeof(float));

    for(int i =0; i < img_size; i++){
        float *pfValue =(float*)(cur_input_tensor->pValue);
        pfValue[i] = mnistImage.ptr<unsigned char>(0)[i];
    }

    printf("Call api to inferencing.....\n");
    inference_tensors(&tModelHandel);
    printf("End inference!!!\n");

    //打印推理结果
    tensor_t* cur_output_tensor = &(output_tensor_array->pTensorArray[0]);
    for (int i = 0; i < cur_output_tensor->pTensorInfo->nElementSize; i++)
    {
        float* value =(float*) cur_output_tensor->pValue;
        printf("%e\n", value[i]);
    }

    //释放申请的Tensor数组内存
    deinit_tensors(input_tensor_array, output_tensor_array);

    release_model(&tModelHandel);

    free(in_tensor_params_ar.pTensorParamArray);
    free(out_tensor_params_ar.pTensorParamArray);

}