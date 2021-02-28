#ifndef __MY_INTERFACE_H_
#define __MY_INTERFACE_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include "common.h"
   /**
 * 功能： 申请输入/输出tensor array的内存
 * 参数：
 *     input_tensors_params（in） ： 输入的tensor参数结构体；
 *     output_tensors_params(in) : 输出的tensor参数结构体；
 *     input_tensors（out) :        申请的输入tensor数组；
 *     output_tensors（out) :       申请的输出tensor数组；
 **/
   result_t init_tensors(tensor_params_array_t *input_tensors_params, tensor_params_array_t *output_tensors_params,
                         tensor_array_t **input_tensors, tensor_array_t **output_tensors);

   /**
 * 功能： 释放申请的tensor array的内存
 * 参数
 *     input_tensors（in) :        申请的输入tensor数组指针；
 *     output_tensors（in) :       申请的输出tensor数组指针；
 **/
   result_t deinit_tensors(tensor_array_t *input_tensors, tensor_array_t *output_tensors);

   /**
 * 功能： 根据模型参数装载tensorflow模型
 * 参数：
 *     model_param（in) : 模型的输入参数
 *     input_tensors（in) :        输入tensor数组指针；
 *     output_tensors（in) :       输出tensor数组指针；
 *     model_handle（out) :装载好的模型句柄 
 **/
   result_t load_model(model_params_t *load_model_param,
                       tensor_array_t *input_tensors,
                       tensor_array_t *output_tensors,
                       model_handle_t *load_model_handle);

   /**
 * 功能： 释放模型的内存
 * 参数：
 *      model_handle（in):要释放的模型句柄
**/
   result_t release_model(model_handle_t *load_model_handle);

   /**
 *  功能：进行推理，推理后的结果放到output_tensors中
 *  参数：
 *       model_handle(in) 模型句柄
 **/
   result_t inference_tensors(model_handle_t *load_model_handle);

#ifdef __cplusplus
}
#endif

#endif