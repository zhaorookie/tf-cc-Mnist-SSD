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
    cur_in_tensor_params->type = DT_UINT8;
    cur_in_tensor_params->pShape[0] = BATH_SIZE;
    cur_in_tensor_params->pShape[1] = 636;     // H，这里的636是待测试图片的高
    cur_in_tensor_params->pShape[2] = 1024;    // W，这里的1025是待测试图片的宽
    cur_in_tensor_params->pShape[3] = 3;       // channel
    strcpy(cur_in_tensor_params->aTensorName, "inputs");

    //输出Tensor数组参数设置，同样通过saved_model_cli show可得
    out_tensor_params_ar.nArraySize = 3; //这里根据自己想要查看的信息种类进行设置，比如我只想看框的数量、分类、框的得分三种信息，则设置3
    strcpy(out_tensor_params_ar.pcSignatureDef, "serving_default");
    out_tensor_params_ar.pTensorParamArray = (tensor_params_t *)malloc(
        out_tensor_params_ar.nArraySize * sizeof(tensor_params_t));

    //通过saved_model_cli show可知第一种信息是 detection_boxes
    tensor_params_t *cur_out_tensor_params1 = &(out_tensor_params_ar.pTensorParamArray[0]);
    cur_out_tensor_params1->type = DT_FLOAT;
    cur_out_tensor_params1->nDims = 3;  //(-1, 100, 4)
    cur_out_tensor_params1->pShape[0] = BATH_SIZE;
    cur_out_tensor_params1->pShape[1] = 100;
    cur_out_tensor_params1->pShape[2] = 4;
    strcpy(cur_out_tensor_params1->aTensorName, "detection_boxes");

    //通过saved_model_cli show可知第二种信息是 detection_classes
    tensor_params_t *cur_out_tensor_params2 = &(out_tensor_params_ar.pTensorParamArray[1]);
    cur_out_tensor_params2->type = DT_FLOAT;
    cur_out_tensor_params2->nDims = 2;  //(-1, 100)
    cur_out_tensor_params2->pShape[0] = BATH_SIZE;
    cur_out_tensor_params2->pShape[1] = 100;
    strcpy(cur_out_tensor_params2->aTensorName, "detection_classes");

    //通过saved_model_cli show可知第四种信息是 detection_scores
    tensor_params_t *cur_out_tensor_params3 = &(out_tensor_params_ar.pTensorParamArray[2]);
    cur_out_tensor_params3->type = DT_FLOAT;
    cur_out_tensor_params3->nDims = 2;  //(-1)
    cur_out_tensor_params3->pShape[0] = BATH_SIZE;
    cur_out_tensor_params3->pShape[1] = 100;
    strcpy(cur_out_tensor_params3->aTensorName, "detection_scores");

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

    tModelParams.gpu_memory_faction = 0.9;

    // tModelParams.bIsCipher = true;
    // strcpy(tModelParams.model_path, "models/tf_mnist_enc/1");

    tModelParams.bIsCipher = false;
    // strcpy(tModelParams.model_path, "models/mnist_export_tf2/saved_model/1");
    strcpy(tModelParams.model_path, "models/objDet_export_tf2/saved_model/1");

    // 同样通过saved_model_cli show可得
    strcpy(tModelParams.paModelTagSet, "serve");

    //调用API装载模型
    if (SUCCESS != load_model(&tModelParams, input_tensor_array, output_tensor_array, &tModelHandel))
    {
        printf("Load Model error!!!\n");
    }

    //填充input tensor array的数据
    tensor_t *cur_input_tensor =&(input_tensor_array->pTensorArray[0]);
    tensor_params_t* cur_input_tensor_info = cur_input_tensor->pTensorInfo;

    cv::Mat bgrImg, rgbImg;
    bgrImg = cv::imread("test_data/image1.jpg");
    cv::cvtColor(bgrImg, rgbImg, cv::COLOR_BGR2RGB);

    int img_size = rgbImg.rows * rgbImg.cols * rgbImg.channels();

    std::cout << "Cur tensor value length: " <<cur_input_tensor_info->nLength <<std::endl;
    assert(img_size == cur_input_tensor_info->nLength);

    memcpy(cur_input_tensor -> pValue, rgbImg.ptr<unsigned char>(0), img_size);

    printf("Call api to inferencing.....\n");
    inference_tensors(&tModelHandel);
    printf("End inference!!!\n");

    //打印推理结果，这里根据上面输出的三种信息(框的数量、分类、框的得分)获得
    tensor_t* cur_output_tensor_boxes = &(output_tensor_array->pTensorArray[0]);
    tensor_t* cur_output_tensor_classes = &(output_tensor_array->pTensorArray[1]);
    tensor_t* cur_output_tensor_scores = &(output_tensor_array->pTensorArray[2]);

    //转换类型
    float* fBoxes = (float*)cur_output_tensor_boxes -> pValue;
    float* fCls = (float*)cur_output_tensor_classes -> pValue;
    float* fScore = (float*)cur_output_tensor_scores -> pValue;

    for (int i = 0; i < BATH_SIZE; i++)
    {
        for (int j = 0; j < 100; j++)  // 100是分类的数量
        {
            std::cout << "box[" << j << "]:" 
                      << fBoxes[4 * j + 0] << " "
                      << fBoxes[4 * j + 1] << " "
                      << fBoxes[4 * j + 2] << " "
                      << fBoxes[4 * j + 3] << " " << std::endl;
            std::cout << "class: " << fCls[j] << "; score:" << fScore[j] << std::endl << std::endl;
        }
    }


    //释放申请的Tensor数组内存
    deinit_tensors(input_tensor_array, output_tensor_array);

    release_model(&tModelHandel);

    free(in_tensor_params_ar.pTensorParamArray);
    free(out_tensor_params_ar.pTensorParamArray);

}