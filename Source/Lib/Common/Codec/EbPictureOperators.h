/*
* Copyright(c) 2019 Intel Corporation
* SPDX - License - Identifier: BSD - 2 - Clause - Patent
*/

#ifndef EbPictureOperators_h
#define EbPictureOperators_h

#include "EbPictureOperators_C.h"
#include "EbPictureOperators_SSE2.h"
#include "EbPictureOperators_SSE4_1.h"
#include "EbPictureOperators_AVX2.h"
#include "EbPictureOperators_AVX512.h"
#include "EbHmCode.h"
#include "EbDefinitions.h"
#include "EbPictureBufferDesc.h"
#include "EbSequenceControlSet.h"
#ifdef __cplusplus
extern "C" {
#endif

    extern void picture_addition(
        uint8_t  *pred_ptr,
        uint32_t  pred_stride,
        int16_t  *residual_ptr,
        uint32_t  residual_stride,
        uint8_t  *recon_ptr,
        uint32_t  recon_stride,
        uint32_t  width,
        uint32_t  height,
        EbAsm     asm_type);

#if PREDICT_NSQ_SHAPE
    extern EbErrorType picture_copy8_bit(
        EbPictureBufferDesc  *src,
        uint32_t                src_luma_origin_index,
        uint32_t                src_chroma_origin_index,
        EbPictureBufferDesc  *dst,
        uint32_t                dst_luma_origin_index,
        uint32_t                dst_chroma_origin_index,
        uint32_t                area_width,
        uint32_t                area_height,
        uint32_t                chroma_area_width,
        uint32_t                chroma_area_height,
        uint32_t                component_mask,
        EbAsm                   asm_type);
#endif

    extern EbErrorType picture_full_distortion32_bits(
        EbPictureBufferDesc  *coeff,
        uint32_t                coeff_luma_origin_index,
        uint32_t                coeff_chroma_origin_index,
        EbPictureBufferDesc  *recon_coeff,
        uint32_t                recon_coeff_luma_origin_index,
        uint32_t                recon_coeff_chroma_origin_index,
        uint32_t                bwidth,
        uint32_t                bheight,
        uint32_t                bwidth_uv,
        uint32_t                bheight_uv,
        uint64_t                y_distortion[DIST_CALC_TOTAL],
        uint64_t                cb_distortion[DIST_CALC_TOTAL],
        uint64_t                cr_distortion[DIST_CALC_TOTAL],
        uint32_t                y_count_non_zero_coeffs,
        uint32_t                cb_count_non_zero_coeffs,
        uint32_t                cr_count_non_zero_coeffs,
        COMPONENT_TYPE          component_type);

    extern uint64_t compute_nx_m_satd_sad_lcu(
        uint8_t  *src,        // input parameter, source samples Ptr
        uint32_t  src_stride,  // input parameter, source stride
        uint32_t  width,      // input parameter, block width (N)
        uint32_t  height);     // input parameter, block height (M)

    //Residual Data

    void compressed_pack_lcu(
        uint8_t  *in8_bit_buffer,
        uint32_t  in8_stride,
        uint8_t  *inn_bit_buffer,
        uint32_t  inn_stride,
        uint16_t *out16_bit_buffer,
        uint32_t  out_stride,
        uint32_t  width,
        uint32_t  height);

    void conv2b_to_c_pack_lcu(
        const uint8_t *inn_bit_buffer,
        uint32_t       inn_stride,
        uint8_t       *in_compn_bit_buffer,
        uint32_t       out_stride,
        uint8_t       *local_cache,
        uint32_t       width,
        uint32_t       height);

    void pack2d_src(
        uint8_t  *in8_bit_buffer,
        uint32_t  in8_stride,
        uint8_t  *inn_bit_buffer,
        uint32_t  inn_stride,
        uint16_t *out16_bit_buffer,
        uint32_t  out_stride,
        uint32_t  width,
        uint32_t  height,
        EbAsm     asm_type);

    void un_pack2d(
        uint16_t *in16_bit_buffer,
        uint32_t  in_stride,
        uint8_t  *out8_bit_buffer,
        uint32_t  out8_stride,
        uint8_t  *outn_bit_buffer,
        uint32_t  outn_stride,
        uint32_t  width,
        uint32_t  height,
        EbAsm     asm_type);

    void extract_8bit_data(
        uint16_t *in16_bit_buffer,
        uint32_t  in_stride,
        uint8_t  *out8_bit_buffer,
        uint32_t  out8_stride,
        uint32_t  width,
        uint32_t  height);

    void unpack_l0l1_avg(
        uint16_t *ref16_l0,
        uint32_t  ref_l0_stride,
        uint16_t *ref16_l1,
        uint32_t  ref_l1_stride,
        uint8_t  *dst_ptr,
        uint32_t  dst_stride,
        uint32_t  width,
        uint32_t  height);

    void extract8_bitdata_safe_sub(
        uint16_t   *in16_bit_buffer,
        uint32_t    in_stride,
        uint8_t    *out8_bit_buffer,
        uint32_t    out8_stride,
        uint32_t    width,
        uint32_t    height,
        EbBool      sub_pred);

    void unpack_l0l1_avg_safe_sub(
        uint16_t *ref16_l0,
        uint32_t  ref_l0_stride,
        uint16_t *ref16_l1,
        uint32_t  ref_l1_stride,
        uint8_t  *dst_ptr,
        uint32_t  dst_stride,
        uint32_t  width,
        uint32_t  height,
        EbBool    sub_pred);

    void memcpy16bit(
        uint16_t *out_ptr,
        uint16_t *in_ptr,
        uint64_t  num_of_elements);

    void memcpy32bit(
        uint32_t *out_ptr,
        uint32_t *in_ptr,
        uint64_t  num_of_elements);

    static INLINE void memset16bit(
        uint16_t                     * in_ptr,
        uint16_t                       value,
        uint64_t                       num_of_elements)
    {
        uint64_t i;

        for (i = 0; i < num_of_elements; i++)
            in_ptr[i] = value;
    }

    static INLINE void memset32bit(
        uint32_t                     * in_ptr,
        uint32_t                       value,
        uint64_t                       num_of_elements)
    {
        uint64_t i;

        for (i = 0; i < num_of_elements; i++)
            in_ptr[i] = value;
    }

    static void picture_addition_void_func() {}
    static void pic_zero_out_coef_void_func() {}

    int32_t sum_residual_c(
        int16_t  *in_ptr,
        uint32_t  size,
        uint32_t  stride_in);

    void memset16bit_block(
        int16_t  *in_ptr,
        uint32_t  stride_in,
        uint32_t  size,
        int16_t   value);

    void full_distortion_kernel_cbf_zero32_bits_c(
        int32_t  *coeff,
        uint32_t  coeff_stride,
        int32_t  *recon_coeff,
        uint32_t  recon_coeff_stride,
        uint64_t  distortion_result[DIST_CALC_TOTAL],
        uint32_t  area_width,
        uint32_t  area_height);

    void full_distortion_kernel32_bits_c(
        int32_t  *coeff,
        uint32_t  coeff_stride,
        int32_t  *recon_coeff,
        uint32_t  recon_coeff_stride,
        uint64_t  distortion_result[DIST_CALC_TOTAL],
        uint32_t  area_width,
        uint32_t  area_height);

    uint64_t full_distortion_kernel16_bits(
        uint8_t  *input,
        uint32_t  input_offset,
        uint32_t  input_stride,
        uint8_t  *pred,
        uint32_t  pred_offset,
        uint32_t  pred_stride,
        uint32_t  area_width,
        uint32_t  area_height);

    /***************************************
    * Function Types
    ***************************************/
    typedef void(*EbAddKernelType)(
        uint8_t  *pred_ptr,
        uint32_t  pred_stride,
        int16_t  *residual_ptr,
        uint32_t  residual_stride,
        uint8_t  *recon_ptr,
        uint32_t  recon_stride,
        uint32_t  width,
        uint32_t  height);

    typedef void(*EbZeroCoeffType)(
        int16_t *coeff_buffer,
        uint32_t coeff_stride,
        uint32_t coeff_origin_index,
        uint32_t area_width,
        uint32_t area_height);

    /***************************************
    * Function Tables
    ***************************************/
    static EbAddKernelType FUNC_TABLE addition_kernel_func_ptr_array[ASM_TYPE_TOTAL][9] = {
        // NON_AVX2
        {
            /*0 4x4   */    picture_addition_kernel4x4_sse_intrin,
            /*1 8x8   */    picture_addition_kernel8x8_sse2_intrin,
            /*2 16x16 */    picture_addition_kernel16x16_sse2_intrin,
            /*3       */    (EbAddKernelType)picture_addition_void_func,
            /*4 32x32 */    picture_addition_kernel32x32_sse2_intrin,
            /*5       */    (EbAddKernelType)picture_addition_void_func,
            /*6       */    (EbAddKernelType)picture_addition_void_func,
            /*7       */    (EbAddKernelType)picture_addition_void_func,
            /*8 64x64 */    picture_addition_kernel64x64_sse2_intrin,
        },
        // AVX2
        {
            /*0 4x4   */    picture_addition_kernel4x4_sse_intrin,
            /*1 8x8   */    picture_addition_kernel8x8_sse2_intrin,
            /*2 16x16 */    picture_addition_kernel16x16_sse2_intrin,
            /*3       */    (EbAddKernelType)picture_addition_void_func,
            /*4 32x32 */    picture_addition_kernel32x32_sse2_intrin,
            /*5       */    (EbAddKernelType)picture_addition_void_func,
            /*6       */    (EbAddKernelType)picture_addition_void_func,
            /*7       */    (EbAddKernelType)picture_addition_void_func,
            /*8 64x64 */    picture_addition_kernel64x64_sse2_intrin,
        },
    };

    typedef void(*EB_RESDKERNELSUBSAMPLED_TYPE)(
        uint8_t  *input,
        uint32_t  input_stride,
        uint8_t  *pred,
        uint32_t  pred_stride,
        int16_t  *residual,
        uint32_t  residual_stride,
        uint32_t  area_width,
        uint32_t  area_height,
        uint8_t   last_line
        );

    void residual_kernel16bit(
        uint16_t *input,
        uint32_t  input_stride,
        uint16_t *pred,
        uint32_t  pred_stride,
        int16_t  *residual,
        uint32_t  residual_stride,
        uint32_t  area_width,
        uint32_t  area_height);

    void residual_kernel_c(
        uint8_t  *input,
        uint32_t  input_stride,
        uint8_t  *pred,
        uint32_t  pred_stride,
        int16_t  *residual,
        uint32_t  residual_stride,
        uint32_t  area_width,
        uint32_t  area_height);

    void residual_kernel_subsampled(
        uint8_t   *input,
        uint32_t   input_stride,
        uint8_t   *pred,
        uint32_t   pred_stride,
        int16_t  *residual,
        uint32_t   residual_stride,
        uint32_t   area_width,
        uint32_t   area_height,
        uint8_t    last_line);

    static EbZeroCoeffType FUNC_TABLE pic_zero_out_coef_func_ptr_array[ASM_TYPE_TOTAL][5] = {
        // NON_AVX2
        {
            /*0 4x4   */     zero_out_coeff4x4_sse,
            /*1 8x8   */     zero_out_coeff8x8_sse2,
            /*2 16x16 */     zero_out_coeff16x16_sse2,
            /*3       */     (EbZeroCoeffType)pic_zero_out_coef_void_func,
            /*4 32x32 */     zero_out_coeff32x32_sse2
        },
        // AVX2
        {
            /*0 4x4   */     zero_out_coeff4x4_sse,
            /*1 8x8   */     zero_out_coeff8x8_sse2,
            /*2 16x16 */     zero_out_coeff16x16_sse2,
            /*3       */     (EbZeroCoeffType)pic_zero_out_coef_void_func,
            /*4 32x32 */     zero_out_coeff32x32_sse2
        },
    };

    void picture_addition_kernel16_bit(
        uint16_t *pred_ptr,
        uint32_t  pred_stride,
        int32_t  *residual_ptr,
        uint32_t  residual_stride,
        uint16_t *recon_ptr,
        uint32_t  recon_stride,
        uint32_t  width,
        uint32_t  height,
        int32_t   bd);

void pic_copy_kernel_8bit(
    EbByte                     src,
    uint32_t                   src_stride,
    EbByte                     dst,
    uint32_t                   dst_stride,
    uint32_t                   area_width,
    uint32_t                   area_height);

void pic_copy_kernel_16bit(
    uint16_t                  *src,
    uint32_t                   src_stride,
    uint16_t                  *dst,
    uint32_t                   dst_stride,
    uint32_t                   width,
    uint32_t                   height);

EbErrorType picture_copy(
    EbPictureBufferDesc       *src,
    uint32_t                   src_luma_origin_index,
    uint32_t                   src_chroma_origin_index,
    EbPictureBufferDesc       *dst,
    uint32_t                   dst_luma_origin_index,
    uint32_t                   dst_chroma_origin_index,
    uint32_t                   area_width,
    uint32_t                   area_height,
    uint32_t                   chroma_area_width,
    uint32_t                   chroma_area_height,
    uint32_t                   component_mask,
    EbBool                     hbd,
    EbAsm                      asm_type);

#ifdef __cplusplus
}
#endif
#endif // EbPictureOperators_h
