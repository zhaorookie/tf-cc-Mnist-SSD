#ifndef _MY_COMMON_H_
#define _MY_COMMON_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdio.h>

    typedef signed char s8;
    typedef unsigned char u8;
    typedef signed short s16;
    typedef unsigned short u16;
    typedef signed int s32;
    typedef unsigned int u32;
    typedef signed char BOOL;

#define TRUE 1
#define FALSE 0

#ifdef DEBUG_ON
#define MY_DEBUG(...)                                                                          \
    do                                                                                         \
    {                                                                                          \
        fprintf(stdout, "[DEBUG]  %s    %s  (Line  %d) : ", __FILE__, __FUNCTION__, __LINE__); \
        fprintf(stdout, __VA_ARGS__);                                                          \
    } while (0)
#else
#define MY_DEBUG(...)
#endif

#define MY_ERROR(...)                                                                          \
    do                                                                                         \
    {                                                                                          \
        fprintf(stderr, "[ERROR]  %s    %s  (Line  %d) : ", __FILE__, __FUNCTION__, __LINE__); \
        fprintf(stderr, __VA_ARGS__);                                                          \
    } while (0)

#define MY_CHECK_NULL(a, errcode)     \
    do                                \
    {                                 \
        if (NULL == (a))              \
        {                             \
            MY_ERROR("NULL DATA \n"); \
            return errcode;           \
        }                             \
    } while (0)

    typedef enum
    {
        SUCCESS = 0, //成功
        FAILED,      //失败
        PARAM_NULL,  //参数为空
        PARAM_SET_ERROR,
        FILE_NOT_EXIST,       //文件不存在
        MEMORY_MALLOC_FAILED, //内存分配失败
        MODEL_LOAD_FAILED,    //模型加载失败
        TENSOR_ALLOC_FAILED,  //tensor内存分配失败
    } result_t;

    typedef struct
    {
        int cpu_or_gpu;           //模型加载再cpu：０；　　gpu: 1
        char visibleCard[32];     //设置哪些ＧＰＵ卡是可见的
        int gpu_id;               //虚拟的gpu id
        float gpu_memory_faction; //设置ＧＰＵ显存的比例
        char model_path[256];     //模型的路径名
        char paModelTagSet[256];  //模型的 tagset
        BOOL bIsCipher;            //模型文件是否加密
    } model_params_t;

    typedef struct
    {
        void *model_handle;         //模型句柄
    } model_handle_t;

    typedef enum
    {
        DT_INVALID = 0,
        DT_FLOAT = 1,
        DT_DOUBLE = 2,
        DT_INT32 = 3,
        DT_UINT8 = 4,
        DT_INT16 = 5,
        DT_INT8 = 6,
        DT_STRING = 7,
        DT_COMPLEX64 = 8,
        DT_INT64 = 9,
        DT_BOOL = 10,
        DT_QINT8 = 11,
        DT_QUINT8 = 12,
        DT_QINT32 = 13,
        DT_BFLOAT16 = 14,
        DT_QINT16 = 15,
        DT_QUINT16 = 16,
        DT_UINT16 = 17,
        DT_COMPLEX128 = 18,
        DT_HALF = 19,
        DT_RESOURCE = 20,
        DT_VARIANT = 21,
        DT_UINT32 = 22,
        DT_UINT64 = 23,
        DT_FLOAT_REF = 101,
        DT_DOUBLE_REF = 102,
        DT_INT32_REF = 103,
        DT_UINT8_REF = 104,
        DT_INT16_REF = 105,
        DT_INT8_REF = 106,
        DT_STRING_REF = 107,
        DT_COMPLEX64_REF = 108,
        DT_INT64_REF = 109,
        DT_BOOL_REF = 110,
        DT_QINT8_REF = 111,
        DT_QUINT8_REF = 112,
        DT_QINT32_REF = 113,
        DT_BFLOAT16_REF = 114,
        DT_QINT16_REF = 115,
        DT_QUINT16_REF = 116,
        DT_UINT16_REF = 117,
        DT_COMPLEX128_REF = 118,
        DT_HALF_REF = 119,
        DT_RESOURCE_REF = 120,
        DT_VARIANT_REF = 121,
        DT_UINT32_REF = 122,
        DT_UINT64_REF = 123,
    } tensor_types_t;

    //Tensor参数的数据结构
    typedef struct
    {
        tensor_types_t type;   //Tensor的类型
        char aTensorName[256]; //Tensor的名字
        int nDims;             //Tensor的rank
        int pShape[4];         //shape
        int nElementSize;      //多少个元素
        int nLength;           //多少个字节长度
    } tensor_params_t;

    //定义Tensor的数据结构
    typedef struct
    {
        tensor_params_t *pTensorInfo;
        void *pValue;
    } tensor_t;

    typedef struct
    {
        int nArraySize;
        tensor_params_t *pTensorParamArray;
        char pcSignatureDef[256]; //函数签名
    } tensor_params_array_t;

    typedef struct
    {
        int nArraySize;
        tensor_t *pTensorArray;
        char pcSignatureDef[256]; //函数签名
    } tensor_array_t;

#ifdef __cplusplus
}
#endif

#endif