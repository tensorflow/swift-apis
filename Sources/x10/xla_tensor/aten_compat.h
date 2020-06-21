/*
 * Copyright 2020 TensorFlow Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cassert>
#include <cstdint>
#include <cstring>
#include <ostream>
#include <string>
#include <vector>

#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "absl/types/variant.h"

// TODO(asuhan): remove

#define FORALL_ATEN_BASE_SYMBOLS(_)                         \
  _(aten, __and__)                                          \
  _(aten, __iand__)                                         \
  _(aten, __ilshift__)                                      \
  _(aten, __ior__)                                          \
  _(aten, __irshift__)                                      \
  _(aten, __ixor__)                                         \
  _(aten, __lshift__)                                       \
  _(aten, __or__)                                           \
  _(aten, __rshift__)                                       \
  _(aten, __xor__)                                          \
  _(aten, _abs)                                             \
  _(aten, _acos)                                            \
  _(aten, _addmv)                                           \
  _(aten, _addr)                                            \
  _(aten, _arange)                                          \
  _(aten, _argmax)                                          \
  _(aten, _argmin)                                          \
  _(aten, _asin)                                            \
  _(aten, _atan)                                            \
  _(aten, _baddbmm_mkl)                                     \
  _(aten, _cast_Byte)                                       \
  _(aten, _cast_Char)                                       \
  _(aten, _cast_Double)                                     \
  _(aten, _cast_Float)                                      \
  _(aten, _cast_Half)                                       \
  _(aten, _cast_Int)                                        \
  _(aten, _cast_Long)                                       \
  _(aten, _cast_Short)                                      \
  _(aten, _cat)                                             \
  _(aten, _ceil)                                            \
  _(aten, _convolution)                                     \
  _(aten, _convolution_double_backward)                     \
  _(aten, convolution_overrideable)                         \
  _(aten, convolution_backward_overrideable)                \
  _(aten, _convolution_nogroup)                             \
  _(aten, _copy_ignoring_overlaps)                          \
  _(aten, _cos)                                             \
  _(aten, _cosh)                                            \
  _(aten, _ctc_loss)                                        \
  _(aten, _ctc_loss_backward)                               \
  _(aten, _cudnn_ctc_loss)                                  \
  _(aten, _cudnn_init_dropout_state)                        \
  _(aten, _cudnn_rnn)                                       \
  _(aten, _cudnn_rnn_backward)                              \
  _(aten, _cudnn_rnn_flatten_weight)                        \
  _(aten, _cufft_clear_plan_cache)                          \
  _(aten, _cufft_get_plan_cache_max_size)                   \
  _(aten, _cufft_get_plan_cache_size)                       \
  _(aten, _cufft_set_plan_cache_max_size)                   \
  _(aten, _cumprod)                                         \
  _(aten, _cumsum)                                          \
  _(aten, _denseDims)                                       \
  _(aten, _dimI)                                            \
  _(aten, _dimV)                                            \
  _(aten, _dim_arange)                                      \
  _(aten, _dirichlet_grad)                                  \
  _(aten, _dot)                                             \
  _(aten, _embedding_bag)                                   \
  _(aten, _embedding_bag_backward)                          \
  _(aten, _embedding_bag_dense_backward)                    \
  _(aten, _embedding_bag_sparse_backward)                   \
  _(aten, _erf)                                             \
  _(aten, _erfc)                                            \
  _(aten, _exp)                                             \
  _(aten, _expm1)                                           \
  _(aten, _fft_with_size)                                   \
  _(aten, _fill)                                            \
  _(aten, _floor)                                           \
  _(aten, _fused_dropout)                                   \
  _(aten, _ger)                                             \
  _(aten, _indexCopy)                                       \
  _(aten, _indices)                                         \
  _(aten, _linspace)                                        \
  _(aten, _local_scalar)                                    \
  _(aten, _local_scalar_dense)                              \
  _(aten, _log)                                             \
  _(aten, _log10)                                           \
  _(aten, _log1p)                                           \
  _(aten, _log2)                                            \
  _(aten, _logspace)                                        \
  _(aten, _lu_with_info)                                    \
  _(aten, _masked_scale)                                    \
  _(aten, _mm)                                              \
  _(aten, _mv)                                              \
  _(aten, _nnz)                                             \
  _(aten, _pack_padded_sequence)                            \
  _(aten, _pack_padded_sequence_backward)                   \
  _(aten, _pad_packed_sequence)                             \
  _(aten, _pdist_backward)                                  \
  _(aten, _pdist_forward)                                   \
  _(aten, _prod)                                            \
  _(aten, _prodall)                                         \
  _(aten, _range)                                           \
  _(aten, _reshape_from_tensor)                             \
  _(aten, _round)                                           \
  _(aten, _rsqrt)                                           \
  _(aten, _s_where)                                         \
  _(aten, _shape_as_tensor)                                 \
  _(aten, _sigmoid)                                         \
  _(aten, _sigmoid_backward)                                \
  _(aten, _sigmoid_forward)                                 \
  _(aten, _sin)                                             \
  _(aten, _sinh)                                            \
  _(aten, _sparseDims)                                      \
  _(aten, _sparse_add)                                      \
  _(aten, _sparse_addmm)                                    \
  _(aten, _sparse_coo_tensor_with_dims)                     \
  _(aten, _sparse_coo_tensor_with_dims_and_tensors)         \
  _(aten, _sparse_coo_tensor_unsafe)                        \
  _(aten, _sparse_dense_add)                                \
  _(aten, _sparse_div_scalar)                               \
  _(aten, _sparse_div_zerodim)                              \
  _(aten, _sparse_mul)                                      \
  _(aten, _sparse_mul_scalar)                               \
  _(aten, _sparse_mul_zerodim)                              \
  _(aten, _sparse_sum)                                      \
  _(aten, _sqrt)                                            \
  _(aten, _square)                                          \
  _(aten, _standard_gamma)                                  \
  _(aten, _standard_gamma_grad)                             \
  _(aten, _sum)                                             \
  _(aten, _sum_cuda)                                        \
  _(aten, _sumall)                                          \
  _(aten, _tan)                                             \
  _(aten, _tanh)                                            \
  _(aten, _tanh_backward)                                   \
  _(aten, _tanh_forward)                                    \
  _(aten, _th_baddbmm)                                      \
  _(aten, _th_bmm)                                          \
  _(aten, _th_clamp)                                        \
  _(aten, _th_clamp_max)                                    \
  _(aten, _th_clamp_min)                                    \
  _(aten, _th_get_device)                                   \
  _(aten, _th_kthvalue)                                     \
  _(aten, _th_max)                                          \
  _(aten, _th_median)                                       \
  _(aten, _th_min)                                          \
  _(aten, _th_mode)                                         \
  _(aten, _th_prod)                                         \
  _(aten, _th_sigmoid)                                      \
  _(aten, _th_std)                                          \
  _(aten, _th_sum)                                          \
  _(aten, _th_tanh)                                         \
  _(aten, _th_var)                                          \
  _(aten, _thnn_fused_gru_cell)                             \
  _(aten, _thnn_fused_gru_cell_backward)                    \
  _(aten, _thnn_fused_lstm_cell)                            \
  _(aten, _thnn_fused_lstm_cell_backward)                   \
  _(aten, _trilinear)                                       \
  _(aten, _trunc)                                           \
  _(aten, _unique)                                          \
  _(aten, _unique_dim)                                      \
  _(aten, _unsafe_view)                                     \
  _(aten, _values)                                          \
  _(aten, _weight_norm)                                     \
  _(aten, _weight_norm_cuda_interface)                      \
  _(aten, _weight_norm_cuda_interface_backward)             \
  _(aten, _weight_norm_differentiable_backward)             \
  _(aten, abs)                                              \
  _(aten, acos)                                             \
  _(aten, acosh)                                            \
  _(aten, adaptive_avg_pool1d)                              \
  _(aten, adaptive_avg_pool2d)                              \
  _(aten, adaptive_avg_pool2d_backward)                     \
  _(aten, adaptive_avg_pool2d_forward)                      \
  _(aten, adaptive_avg_pool3d)                              \
  _(aten, adaptive_avg_pool3d_backward)                     \
  _(aten, adaptive_avg_pool3d_forward)                      \
  _(aten, adaptive_max_pool1d)                              \
  _(aten, adaptive_max_pool2d)                              \
  _(aten, adaptive_max_pool2d_backward)                     \
  _(aten, adaptive_max_pool2d_forward)                      \
  _(aten, adaptive_max_pool3d)                              \
  _(aten, adaptive_max_pool3d_backward)                     \
  _(aten, adaptive_max_pool3d_forward)                      \
  _(aten, add)                                              \
  _(aten, add_)                                             \
  _(aten, addbmm)                                           \
  _(aten, addcdiv)                                          \
  _(aten, addcmul)                                          \
  _(aten, addmm)                                            \
  _(aten, addmv)                                            \
  _(aten, addr)                                             \
  _(aten, affine_grid_generator)                            \
  _(aten, affine_grid_generator_backward)                   \
  _(aten, alias)                                            \
  _(aten, all)                                              \
  _(aten, allclose)                                         \
  _(aten, alpha_dropout)                                    \
  _(aten, any)                                              \
  _(aten, arange)                                           \
  _(aten, argmax)                                           \
  _(aten, argmin)                                           \
  _(aten, as_strided)                                       \
  _(aten, as_tensor)                                        \
  _(aten, asin)                                             \
  _(aten, asinh)                                            \
  _(aten, atan)                                             \
  _(aten, atan2)                                            \
  _(aten, atanh)                                            \
  _(aten, avg_pool1d)                                       \
  _(aten, avg_pool2d)                                       \
  _(aten, avg_pool2d_backward)                              \
  _(aten, avg_pool2d_forward)                               \
  _(aten, avg_pool3d)                                       \
  _(aten, avg_pool3d_backward)                              \
  _(aten, avg_pool3d_forward)                               \
  _(aten, baddbmm)                                          \
  _(aten, bartlett_window)                                  \
  _(aten, batch_norm)                                       \
  _(aten, bernoulli)                                        \
  _(aten, bilinear)                                         \
  _(aten, binary_cross_entropy)                             \
  _(aten, binary_cross_entropy_backward)                    \
  _(aten, binary_cross_entropy_forward)                     \
  _(aten, binary_cross_entropy_with_logits)                 \
  _(aten, binary_cross_entropy_with_logits_backward)        \
  _(aten, binary_cross_entropy_with_logits_target_backward) \
  _(aten, bincount)                                         \
  _(aten, blackman_window)                                  \
  _(aten, bmm)                                              \
  _(aten, broadcast_tensors)                                \
  _(aten, cartesian_prod)                                   \
  _(aten, cat)                                              \
  _(aten, cauchy)                                           \
  _(aten, ceil)                                             \
  _(aten, celu)                                             \
  _(aten, chain_matmul)                                     \
  _(aten, cholesky)                                         \
  _(aten, cholesky_inverse)                                 \
  _(aten, cholesky_solve)                                   \
  _(aten, chunk)                                            \
  _(aten, clamp)                                            \
  _(aten, clamp_max)                                        \
  _(aten, clamp_min)                                        \
  _(aten, clone)                                            \
  _(aten, coalesce)                                         \
  _(aten, combinations)                                     \
  _(aten, constant_pad_nd)                                  \
  _(aten, contiguous)                                       \
  _(aten, conv1d)                                           \
  _(aten, conv2d)                                           \
  _(aten, conv3d)                                           \
  _(aten, conv_tbc)                                         \
  _(aten, conv_tbc_backward)                                \
  _(aten, conv_transpose1d)                                 \
  _(aten, convolution)                                      \
  _(aten, copy_sparse_to_sparse)                            \
  _(aten, cos)                                              \
  _(aten, cosh)                                             \
  _(aten, cosine_embedding_loss)                            \
  _(aten, cosine_similarity)                                \
  _(aten, cross)                                            \
  _(aten, std_mean)                                         \
  _(aten, var_mean)                                         \
  _(aten, ctc_loss)                                         \
  _(aten, cudnn_affine_grid_generator)                      \
  _(aten, cudnn_affine_grid_generator_backward)             \
  _(aten, cudnn_batch_norm)                                 \
  _(aten, cudnn_batch_norm_backward)                        \
  _(aten, cudnn_convolution)                                \
  _(aten, cudnn_convolution_backward)                       \
  _(aten, cudnn_convolution_backward_bias)                  \
  _(aten, cudnn_convolution_backward_input)                 \
  _(aten, cudnn_convolution_backward_weight)                \
  _(aten, cudnn_convolution_transpose)                      \
  _(aten, cudnn_convolution_transpose_backward)             \
  _(aten, cudnn_convolution_transpose_backward_bias)        \
  _(aten, cudnn_convolution_transpose_backward_input)       \
  _(aten, cudnn_convolution_transpose_backward_weight)      \
  _(aten, cudnn_grid_sampler)                               \
  _(aten, cudnn_grid_sampler_backward)                      \
  _(aten, cudnn_is_acceptable)                              \
  _(aten, cumprod)                                          \
  _(aten, cumsum)                                           \
  _(aten, data_ptr)                                         \
  _(aten, det)                                              \
  _(aten, detach)                                           \
  _(aten, diag)                                             \
  _(aten, diag_embed)                                       \
  _(aten, diagflat)                                         \
  _(aten, diagonal)                                         \
  _(aten, fill_diagonal_)                                   \
  _(aten, digamma)                                          \
  _(aten, dim)                                              \
  _(aten, dist)                                             \
  _(aten, div)                                              \
  _(aten, div_)                                             \
  _(aten, dot)                                              \
  _(aten, dropout)                                          \
  _(aten, eig)                                              \
  _(aten, einsum)                                           \
  _(aten, elu)                                              \
  _(aten, elu_backward)                                     \
  _(aten, elu_forward)                                      \
  _(aten, embedding)                                        \
  _(aten, embedding_backward)                               \
  _(aten, embedding_bag)                                    \
  _(aten, embedding_dense_backward)                         \
  _(aten, embedding_renorm)                                 \
  _(aten, embedding_sparse_backward)                        \
  _(aten, empty)                                            \
  _(aten, empty_like)                                       \
  _(aten, empty_strided)                                    \
  _(aten, eq)                                               \
  _(aten, equal)                                            \
  _(aten, erf)                                              \
  _(aten, erfc)                                             \
  _(aten, erfinv)                                           \
  _(aten, exp)                                              \
  _(aten, expand)                                           \
  _(aten, expand_as)                                        \
  _(aten, expm1)                                            \
  _(aten, exponential)                                      \
  _(aten, eye)                                              \
  _(aten, feature_alpha_dropout)                            \
  _(aten, feature_dropout)                                  \
  _(aten, fft)                                              \
  _(aten, fill)                                             \
  _(aten, flatten)                                          \
  _(aten, flip)                                             \
  _(aten, floor)                                            \
  _(aten, fmod)                                             \
  _(aten, frac)                                             \
  _(aten, fractional_max_pool2d)                            \
  _(aten, fractional_max_pool2d_backward)                   \
  _(aten, fractional_max_pool2d_forward)                    \
  _(aten, frobenius_norm)                                   \
  _(aten, full)                                             \
  _(aten, full_like)                                        \
  _(aten, gather)                                           \
  _(aten, ge)                                               \
  _(aten, gelu)                                             \
  _(aten, geometric)                                        \
  _(aten, geqrf)                                            \
  _(aten, ger)                                              \
  _(aten, get_device)                                       \
  _(aten, glu)                                              \
  _(aten, glu_backward)                                     \
  _(aten, glu_forward)                                      \
  _(aten, grid_sampler)                                     \
  _(aten, grid_sampler_2d)                                  \
  _(aten, grid_sampler_2d_backward)                         \
  _(aten, grid_sampler_3d)                                  \
  _(aten, grid_sampler_3d_backward)                         \
  _(aten, group_norm)                                       \
  _(aten, gru)                                              \
  _(aten, gru_cell)                                         \
  _(aten, gt)                                               \
  _(aten, hamming_window)                                   \
  _(aten, hann_window)                                      \
  _(aten, hardshrink)                                       \
  _(aten, hardshrink_backward)                              \
  _(aten, hardsigmoid)                                      \
  _(aten, hardsigmoid_backward)                             \
  _(aten, hardtanh)                                         \
  _(aten, hardtanh_backward)                                \
  _(aten, hardtanh_forward)                                 \
  _(aten, hinge_embedding_loss)                             \
  _(aten, histc)                                            \
  _(aten, hspmm)                                            \
  _(aten, ifft)                                             \
  _(aten, index)                                            \
  _(aten, index_add)                                        \
  _(aten, index_copy)                                       \
  _(aten, index_fill)                                       \
  _(aten, index_put)                                        \
  _(aten, index_select)                                     \
  _(aten, indices)                                          \
  _(aten, instance_norm)                                    \
  _(aten, inverse)                                          \
  _(aten, irfft)                                            \
  _(aten, is_coalesced)                                     \
  _(aten, is_complex)                                       \
  _(aten, is_contiguous)                                    \
  _(aten, is_cuda)                                          \
  _(aten, is_distributed)                                   \
  _(aten, is_floating_point)                                \
  _(aten, is_nonzero)                                       \
  _(aten, is_same_size)                                     \
  _(aten, is_set_to)                                        \
  _(aten, is_signed)                                        \
  _(aten, is_sparse)                                        \
  _(aten, isclose)                                          \
  _(aten, kl_div)                                           \
  _(aten, kl_div_backward)                                  \
  _(aten, kthvalue)                                         \
  _(aten, l1_loss)                                          \
  _(aten, l1_loss_backward)                                 \
  _(aten, l1_loss_forward)                                  \
  _(aten, layer_norm)                                       \
  _(aten, le)                                               \
  _(aten, leaky_relu)                                       \
  _(aten, leaky_relu_backward)                              \
  _(aten, leaky_relu_forward)                               \
  _(aten, lerp)                                             \
  _(aten, lgamma)                                           \
  _(aten, linear)                                           \
  _(aten, linspace)                                         \
  _(aten, log)                                              \
  _(aten, log10)                                            \
  _(aten, log1p)                                            \
  _(aten, log2)                                             \
  _(aten, log_normal)                                       \
  _(aten, log_sigmoid)                                      \
  _(aten, log_sigmoid_backward)                             \
  _(aten, log_sigmoid_forward)                              \
  _(aten, log_softmax)                                      \
  _(aten, _log_softmax)                                     \
  _(aten, _log_softmax_backward_data)                       \
  _(aten, logdet)                                           \
  _(aten, logical_and)                                      \
  _(aten, logical_or)                                       \
  _(aten, logspace)                                         \
  _(aten, logsumexp)                                        \
  _(aten, lstm)                                             \
  _(aten, lstm_cell)                                        \
  _(aten, lstsq)                                            \
  _(aten, lt)                                               \
  _(aten, lu_solve)                                         \
  _(aten, margin_ranking_loss)                              \
  _(aten, masked_fill)                                      \
  _(aten, masked_scatter)                                   \
  _(aten, masked_select)                                    \
  _(aten, matmul)                                           \
  _(aten, matrix_power)                                     \
  _(aten, matrix_rank)                                      \
  _(aten, max)                                              \
  _(aten, max_pool1d)                                       \
  _(aten, max_pool1d_with_indices)                          \
  _(aten, max_pool2d)                                       \
  _(aten, max_pool2d_with_indices)                          \
  _(aten, max_pool2d_with_indices_backward)                 \
  _(aten, max_pool2d_with_indices_forward)                  \
  _(aten, max_pool3d)                                       \
  _(aten, max_pool3d_with_indices)                          \
  _(aten, max_pool3d_with_indices_backward)                 \
  _(aten, max_pool3d_with_indices_forward)                  \
  _(aten, max_unpool2d)                                     \
  _(aten, max_unpool2d_backward)                            \
  _(aten, max_unpool2d_forward)                             \
  _(aten, max_unpool3d)                                     \
  _(aten, max_unpool3d_backward)                            \
  _(aten, max_unpool3d_forward)                             \
  _(aten, max_values)                                       \
  _(aten, mean)                                             \
  _(aten, median)                                           \
  _(aten, meshgrid)                                         \
  _(aten, min)                                              \
  _(aten, min_values)                                       \
  _(aten, miopen_batch_norm)                                \
  _(aten, miopen_batch_norm_backward)                       \
  _(aten, miopen_convolution)                               \
  _(aten, miopen_convolution_backward)                      \
  _(aten, miopen_convolution_backward_bias)                 \
  _(aten, miopen_convolution_backward_input)                \
  _(aten, miopen_convolution_backward_weight)               \
  _(aten, miopen_convolution_transpose)                     \
  _(aten, miopen_convolution_transpose_backward)            \
  _(aten, miopen_convolution_transpose_backward_input)      \
  _(aten, miopen_convolution_transpose_backward_weight)     \
  _(aten, miopen_depthwise_convolution)                     \
  _(aten, miopen_depthwise_convolution_backward)            \
  _(aten, miopen_depthwise_convolution_backward_input)      \
  _(aten, miopen_depthwise_convolution_backward_weight)     \
  _(aten, miopen_rnn)                                       \
  _(aten, miopen_rnn_backward)                              \
  _(aten, mkldnn_convolution)                               \
  _(aten, mkldnn_convolution_backward)                      \
  _(aten, mkldnn_convolution_backward_input)                \
  _(aten, mkldnn_convolution_backward_weights)              \
  _(aten, mm)                                               \
  _(aten, mode)                                             \
  _(aten, mse_loss)                                         \
  _(aten, mse_loss_backward)                                \
  _(aten, mse_loss_forward)                                 \
  _(aten, mul)                                              \
  _(aten, mul_)                                             \
  _(aten, multi_margin_loss)                                \
  _(aten, multi_margin_loss_backward)                       \
  _(aten, multi_margin_loss_forward)                        \
  _(aten, multilabel_margin_loss)                           \
  _(aten, multilabel_margin_loss_backward)                  \
  _(aten, multilabel_margin_loss_forward)                   \
  _(aten, multinomial)                                      \
  _(aten, mv)                                               \
  _(aten, mvlgamma)                                         \
  _(aten, narrow)                                           \
  _(aten, narrow_copy)                                      \
  _(aten, native_batch_norm)                                \
  _(aten, native_batch_norm_backward)                       \
  _(aten, native_clone)                                     \
  _(aten, native_get_device)                                \
  _(aten, native_norm)                                      \
  _(aten, native_pow)                                       \
  _(aten, native_resize_as)                                 \
  _(aten, native_tensor)                                    \
  _(aten, native_zero)                                      \
  _(aten, ne)                                               \
  _(aten, neg)                                              \
  _(aten, bitwise_not)                                      \
  _(aten, bitwise_xor)                                      \
  _(aten, nll_loss)                                         \
  _(aten, nll_loss2d)                                       \
  _(aten, nll_loss2d_backward)                              \
  _(aten, nll_loss2d_forward)                               \
  _(aten, nll_loss_backward)                                \
  _(aten, nll_loss_forward)                                 \
  _(aten, nonzero)                                          \
  _(aten, norm)                                             \
  _(aten, norm_except_dim)                                  \
  _(aten, normal)                                           \
  _(aten, nuclear_norm)                                     \
  _(aten, numel)                                            \
  _(aten, ones)                                             \
  _(aten, ones_like)                                        \
  _(aten, orgqr)                                            \
  _(aten, ormqr)                                            \
  _(aten, pairwise_distance)                                \
  _(aten, pdist)                                            \
  _(aten, cdist)                                            \
  _(aten, permute)                                          \
  _(aten, pin_memory)                                       \
  _(aten, pinverse)                                         \
  _(aten, pixel_shuffle)                                    \
  _(aten, poisson)                                          \
  _(aten, polygamma)                                        \
  _(aten, pow)                                              \
  _(aten, prelu)                                            \
  _(aten, prelu_backward)                                   \
  _(aten, prod)                                             \
  _(aten, put)                                              \
  _(aten, qr)                                               \
  _(aten, rand)                                             \
  _(aten, rand_like)                                        \
  _(aten, randint)                                          \
  _(aten, randint_like)                                     \
  _(aten, randn)                                            \
  _(aten, randn_like)                                       \
  _(aten, random)                                           \
  _(aten, randperm)                                         \
  _(aten, range)                                            \
  _(aten, reciprocal)                                       \
  _(aten, reflection_pad1d)                                 \
  _(aten, reflection_pad1d_backward)                        \
  _(aten, reflection_pad1d_forward)                         \
  _(aten, reflection_pad2d)                                 \
  _(aten, reflection_pad2d_backward)                        \
  _(aten, reflection_pad2d_forward)                         \
  _(aten, relu)                                             \
  _(aten, remainder)                                        \
  _(aten, renorm)                                           \
  _(aten, repeat)                                           \
  _(aten, replication_pad1d)                                \
  _(aten, replication_pad1d_backward)                       \
  _(aten, replication_pad1d_forward)                        \
  _(aten, replication_pad2d)                                \
  _(aten, replication_pad2d_backward)                       \
  _(aten, replication_pad2d_forward)                        \
  _(aten, replication_pad3d)                                \
  _(aten, replication_pad3d_backward)                       \
  _(aten, replication_pad3d_forward)                        \
  _(aten, reshape)                                          \
  _(aten, reshape_as)                                       \
  _(aten, resize)                                           \
  _(aten, resize_)                                          \
  _(aten, resize_as)                                        \
  _(aten, resize_as_)                                       \
  _(aten, rfft)                                             \
  _(aten, rnn_relu)                                         \
  _(aten, rnn_relu_cell)                                    \
  _(aten, rnn_tanh)                                         \
  _(aten, rnn_tanh_cell)                                    \
  _(aten, rot90)                                            \
  _(aten, round)                                            \
  _(aten, round_to_even)                                    \
  _(aten, rrelu)                                            \
  _(aten, rrelu_with_noise)                                 \
  _(aten, rrelu_with_noise_backward)                        \
  _(aten, rrelu_with_noise_forward)                         \
  _(aten, rsqrt)                                            \
  _(aten, scatter)                                          \
  _(aten, scatter_add)                                      \
  _(aten, select)                                           \
  _(aten, selu)                                             \
  _(aten, set)                                              \
  _(aten, sigmoid)                                          \
  _(aten, sign)                                             \
  _(aten, sin)                                              \
  _(aten, sinh)                                             \
  _(aten, size)                                             \
  _(aten, sizes)                                            \
  _(aten, slice)                                            \
  _(aten, slogdet)                                          \
  _(aten, smm)                                              \
  _(aten, smooth_l1_loss)                                   \
  _(aten, smooth_l1_loss_backward)                          \
  _(aten, smooth_l1_loss_forward)                           \
  _(aten, soft_margin_loss)                                 \
  _(aten, soft_margin_loss_backward)                        \
  _(aten, soft_margin_loss_forward)                         \
  _(aten, softmax)                                          \
  _(aten, _softmax)                                         \
  _(aten, _softmax_backward_data)                           \
  _(aten, softplus)                                         \
  _(aten, softplus_backward)                                \
  _(aten, softplus_forward)                                 \
  _(aten, softshrink)                                       \
  _(aten, softshrink_backward)                              \
  _(aten, softshrink_forward)                               \
  _(aten, solve)                                            \
  _(aten, sort)                                             \
  _(aten, sparse_coo_tensor)                                \
  _(aten, sparse_mask)                                      \
  _(aten, sparse_resize)                                    \
  _(aten, sparse_resize_and_clear)                          \
  _(aten, split)                                            \
  _(aten, split_with_sizes)                                 \
  _(aten, sqrt)                                             \
  _(aten, square)                                           \
  _(aten, squeeze)                                          \
  _(aten, sspaddmm)                                         \
  _(aten, stack)                                            \
  _(aten, std)                                              \
  _(aten, stft)                                             \
  _(aten, storage_offset)                                   \
  _(aten, stride)                                           \
  _(aten, strides)                                          \
  _(aten, sub)                                              \
  _(aten, sub_)                                             \
  _(aten, rsub)                                             \
  _(aten, sum)                                              \
  _(aten, sum_to_size)                                      \
  _(aten, svd)                                              \
  _(aten, symeig)                                           \
  _(aten, t)                                                \
  _(aten, take)                                             \
  _(aten, tan)                                              \
  _(aten, tanh)                                             \
  _(aten, tensor)                                           \
  _(aten, tensordot)                                        \
  _(aten, tf_convolution)                                   \
  _(aten, tf_conv_backprop_filter)                          \
  _(aten, tf_conv_backprop_input)                           \
  _(aten, tf_mirror_pad)                                    \
  _(aten, tf_mirror_pad_backward)                           \
  _(aten, tf_one_hot)                                       \
  _(aten, tf_stateless_random_normal)                       \
  _(aten, tf_stateless_random_uniform)                      \
  _(aten, tf_unsorted_segment_sum)                          \
  _(aten, th_addmm)                                         \
  _(aten, th_clone)                                         \
  _(aten, th_norm)                                          \
  _(aten, th_pow)                                           \
  _(aten, th_resize_as)                                     \
  _(aten, th_tensor)                                        \
  _(aten, th_zero)                                          \
  _(aten, thnn_conv2d)                                      \
  _(aten, thnn_conv2d_backward)                             \
  _(aten, thnn_conv2d_forward)                              \
  _(aten, slow_conv3d)                                      \
  _(aten, slow_conv3d_backward)                             \
  _(aten, slow_conv3d_forward)                              \
  _(aten, thnn_conv_depthwise2d)                            \
  _(aten, thnn_conv_depthwise2d_backward)                   \
  _(aten, thnn_conv_depthwise2d_forward)                    \
  _(aten, slow_conv_dilated2d)                              \
  _(aten, slow_conv_dilated2d_backward)                     \
  _(aten, slow_conv_dilated3d)                              \
  _(aten, slow_conv_dilated3d_backward)                     \
  _(aten, slow_conv_transpose2d)                            \
  _(aten, slow_conv_transpose2d_backward)                   \
  _(aten, slow_conv_transpose3d)                            \
  _(aten, slow_conv_transpose3d_backward)                   \
  _(aten, threshold)                                        \
  _(aten, threshold_backward)                               \
  _(aten, to)                                               \
  _(aten, to_sparse)                                        \
  _(aten, to_dense)                                         \
  _(aten, topk)                                             \
  _(aten, trace)                                            \
  _(aten, transpose)                                        \
  _(aten, triangular_solve)                                 \
  _(aten, tril)                                             \
  _(aten, triplet_margin_loss)                              \
  _(aten, triu)                                             \
  _(aten, trunc)                                            \
  _(aten, type_as)                                          \
  _(aten, unbind)                                           \
  _(aten, unfold)                                           \
  _(aten, uniform)                                          \
  _(aten, unsqueeze)                                        \
  _(aten, upsample_bilinear2d)                              \
  _(aten, upsample_bilinear2d_backward)                     \
  _(aten, upsample_bilinear2d_forward)                      \
  _(aten, upsample_bicubic2d)                               \
  _(aten, upsample_bicubic2d_backward)                      \
  _(aten, upsample_bicubic2d_forward)                       \
  _(aten, upsample_linear1d)                                \
  _(aten, upsample_linear1d_backward)                       \
  _(aten, upsample_linear1d_forward)                        \
  _(aten, upsample_nearest1d)                               \
  _(aten, upsample_nearest1d_backward)                      \
  _(aten, upsample_nearest1d_forward)                       \
  _(aten, upsample_nearest2d)                               \
  _(aten, upsample_nearest2d_backward)                      \
  _(aten, upsample_nearest2d_forward)                       \
  _(aten, upsample_nearest3d)                               \
  _(aten, upsample_nearest3d_backward)                      \
  _(aten, upsample_nearest3d_forward)                       \
  _(aten, upsample_trilinear3d)                             \
  _(aten, upsample_trilinear3d_backward)                    \
  _(aten, upsample_trilinear3d_forward)                     \
  _(aten, values)                                           \
  _(aten, var)                                              \
  _(aten, view)                                             \
  _(aten, view_as)                                          \
  _(aten, where)                                            \
  _(aten, zero)                                             \
  _(aten, zeros)                                            \
  _(aten, zeros_like)                                       \
  _(aten, xla_avg_pool)                                     \
  _(aten, xla_avg_pool_grad)                                \
  _(aten, xla_dynamic_update_slice)                         \
  _(aten, xla_dynamic_slice)                                \
  _(aten, xla_max_pool)                                     \
  _(aten, xla_max_pool_grad)                                \
  _(aten, xla_pad)                                          \
  _(aten, xla_rem)                                          \
  _(aten, xla_replica_id)                                   \
  _(aten, xla_slice)                                        \
  _(aten, xla_truncated_normal)                             \
  _(aten, xla_is_finite)                                    \
  _(aten, xla_is_inf)                                       \
  _(aten, xla_is_nan)

#define FORALL_XLA_SYMBOLS(_, __)  \
  __(xla, all_to_all)              \
  _(xla, as_strided_view_update)   \
  _(xla, cast)                     \
  _(xla, collective_permute)       \
  _(xla, cross_replica_sum)        \
  _(xla, device_data)              \
  _(xla, diagonal_view_update)     \
  _(xla, generic_slice)            \
  _(xla, get_dimensions_size)      \
  _(xla, moving_average)           \
  _(xla, nms)                      \
  _(xla, not_supported)            \
  _(xla, replication_pad)          \
  _(xla, replication_pad_backward) \
  _(xla, select)                   \
  _(xla, tensor_data)              \
  _(xla, token)                    \
  _(xla, unselect)                 \
  _(xla, update_slice)

namespace at {

namespace aten {

enum SymbolKind : uint32_t {
#define DEFINE_KEY(ns, s) s,
  FORALL_ATEN_BASE_SYMBOLS(DEFINE_KEY)
#undef DEFINE_KEY
  // Just for enum math. Never use directly.
  END_Symbol
};

}  // namespace aten

namespace prim {

enum SymbolKind : uint32_t {
  Constant = aten::SymbolKind::END_Symbol,
  END_Symbol
};

}  // namespace prim
}  // namespace at

namespace swift_xla {
namespace xla_symbols {

enum SymbolKind : uint32_t {
#define DEFINE_KEY(ns, s) s,
#define DEFINE_FIRST_KEY(ns, s) s = ::at::prim::SymbolKind::END_Symbol,
  FORALL_XLA_SYMBOLS(DEFINE_KEY, DEFINE_FIRST_KEY)
#undef DEFINE_KEY
#undef DEFINE_FIRST_KEY
  // Just for enum math. Never use directly.
  END_Symbol
};

}  // namespace xla_symbols
}  // namespace swift_xla

namespace at {

using BFloat16 = int16_t;
using Half = uint16_t;

#define LIST_SCALAR_TYPES(_)     \
  _(Bool, Bool, bool)            \
  _(Float, Float, float)         \
  _(BFloat16, BFloat16, int16_t) \
  _(Half, Half, int16_t)         \
  _(Double, Double, double)      \
  _(UInt8, Byte, uint8_t)        \
  _(Int8, Char, int8_t)          \
  _(Int16, Short, int16_t)       \
  _(Int32, Int, int32_t)         \
  _(Int64, Long, int64_t)

enum class ScalarType : int8_t {
#define DEFINE_ENUM_CASE(name, aten_name, type) aten_name,
  LIST_SCALAR_TYPES(DEFINE_ENUM_CASE)
#undef DEFINE_ENUM_CASE
};

inline bool isIntegralType(ScalarType t, bool includeBool) {
  bool isIntegral =
      (t == ScalarType::Byte || t == ScalarType::Char || t == ScalarType::Int ||
       t == ScalarType::Long || t == ScalarType::Short);

  return includeBool ? isIntegral || (t == ScalarType::Bool) : isIntegral;
}

inline const char* toString(ScalarType t) { return "unknown"; }

inline std::ostream& operator<<(std::ostream& stream,
                                at::ScalarType scalar_type) {
  return stream << toString(scalar_type);
}

class Scalar {
 public:
  Scalar(int8_t v) : value_(int64_t(v)) {}    // NOLINT
  Scalar(uint8_t v) : value_(int64_t(v)) {}   // NOLINT
  Scalar(int16_t v) : value_(int64_t(v)) {}   // NOLINT
  Scalar(uint16_t v) : value_(int64_t(v)) {}  // NOLINT
  Scalar(int32_t v) : value_(int64_t(v)) {}   // NOLINT
  Scalar(uint32_t v) : value_(int64_t(v)) {}  // NOLINT
  Scalar(int64_t v) : value_(int64_t(v)) {}   // NOLINT
  Scalar(uint64_t v) : value_(int64_t(v)) {}  // NOLINT
  Scalar(float v) : value_(double(v)) {}      // NOLINT
  Scalar(double v) : value_(v) {}             // NOLINT

  bool isFloatingPoint() const { return value_.index() == 1; }

  bool isIntegral() const { return value_.index() == 0; }

  float toFloat() const { return to<float>(); }

  double toDouble() const { return to<double>(); }

  uint8_t toByte() const { return to<uint8_t>(); }

  int8_t toChar() const { return to<int8_t>(); }

  int16_t toShort() const { return to<int16_t>(); }

  int32_t toInt() const { return to<int32_t>(); }

  int64_t toLong() const { return to<int64_t>(); }

  template <class T>
  T to() const {
    // TODO(parkers): Check casts for safety.
    return absl::visit([](auto v) { return static_cast<T>(v); }, value_);
  }

  Scalar operator-() const { return *this; }

 private:
  absl::variant<int64_t, double> value_;
};

namespace internal {
template <typename T>
ScalarType GetScalarType() = delete;

template <>
inline ScalarType GetScalarType<bool>() {
  return ScalarType::Bool;
}
template <>
inline ScalarType GetScalarType<float>() {
  return ScalarType::Float;
}
template <>
inline ScalarType GetScalarType<double>() {
  return ScalarType::Double;
}
template <>
inline ScalarType GetScalarType<uint8_t>() {
  return ScalarType::Byte;
}
template <>
inline ScalarType GetScalarType<int8_t>() {
  return ScalarType::Char;
}
template <>
inline ScalarType GetScalarType<int16_t>() {
  return ScalarType::Short;
}
template <>
inline ScalarType GetScalarType<uint16_t>() {
  return ScalarType::Short;
}
template <>
inline ScalarType GetScalarType<int32_t>() {
  return ScalarType::Int;
}
template <>
inline ScalarType GetScalarType<int64_t>() {
  return ScalarType::Long;
}

inline size_t GetSizeof(ScalarType type) {
  switch (type) {
#define SIZEOF_CASE(name, aten_name, DType) \
  case ScalarType::aten_name:               \
    return sizeof(DType);
    LIST_SCALAR_TYPES(SIZEOF_CASE)
#undef SIZEOF_CASE
  }
}

}  // namespace internal

// Type-erased [T] where T is one of the supported integral or
// floating_point types.
class AnyScalarBuffer {
 public:
  AnyScalarBuffer(const AnyScalarBuffer&) = delete;
  AnyScalarBuffer(AnyScalarBuffer&&) = delete;
  AnyScalarBuffer& operator=(const AnyScalarBuffer&) = delete;
  AnyScalarBuffer& operator=(AnyScalarBuffer&&) = delete;
  virtual ~AnyScalarBuffer() {}

  template <typename T>
  static std::unique_ptr<AnyScalarBuffer> make(std::unique_ptr<T[]> data,
                                               size_t len);

  static std::unique_ptr<AnyScalarBuffer> empty(ScalarType type) {
    return std::unique_ptr<AnyScalarBuffer>(new AnyScalarBuffer(type));
  }

  template <typename T>
  const T* data() {
    using NonConstT = typename std::remove_const<T>::type;
    assert(type_ == internal::GetScalarType<NonConstT>());
    return reinterpret_cast<const T*>(base_);
  }
  const void* raw_data() const { return base_; }
  size_t size() const { return size_; }

  size_t raw_size() const {
    return internal::GetSizeof(scalar_type()) * size();
  }

  ScalarType scalar_type() const { return type_; }

  std::unique_ptr<AnyScalarBuffer> dup() {
    switch (type_) {
#define DUP_CASE(name, aten_name, DType)                \
  case ScalarType::aten_name: {                         \
    std::unique_ptr<DType[]> data;                      \
    if (size_) {                                        \
      data.reset(new DType[size_]);                     \
      memcpy(data.get(), base_, sizeof(DType) * size_); \
    }                                                   \
    return make(std::move(data), size_);                \
  }
      LIST_SCALAR_TYPES(DUP_CASE)
#undef DUP_CASE
    }
  }

 protected:
  explicit AnyScalarBuffer(ScalarType type)
      : base_(nullptr), size_(0), type_(type) {}

  void set_base(const void* base) { base_ = base; }
  void set_size(size_t size) { size_ = size; }

 private:
  const void* base_;
  size_t size_;
  ScalarType type_;
};

// Implementation of Scalar buffer backed by a unique_ptr.
template <typename T>
class OwnedAnyScalarBuffer : public AnyScalarBuffer {
 public:
  OwnedAnyScalarBuffer(std::unique_ptr<T[]> data, size_t len)
      : AnyScalarBuffer(internal::GetScalarType<T>()), data_(std::move(data)) {
    set_base(data_.get());
    set_size(len);
  }

  std::unique_ptr<T[]> data_;
};

// Implementation of Scalar buffer backed by a vector.
template <typename T>
class VectorScalarBuffer : public AnyScalarBuffer {
 public:
  explicit VectorScalarBuffer(std::vector<T> data)
      : AnyScalarBuffer(internal::GetScalarType<T>()), data_(std::move(data)) {
    set_base(data_.data());
    set_size(data_.size());
  }

 private:
  std::vector<T> data_;
};

// Implementation of Scalar buffer backed by a non-owned data buffer.
template <typename T>
class NonOwnedAnyScalarBuffer : public AnyScalarBuffer {
 public:
  NonOwnedAnyScalarBuffer(const T* data, size_t len)
      : AnyScalarBuffer(internal::GetScalarType<T>()) {
    set_base(data);
    set_size(len);
  }
};

template <typename T>
std::unique_ptr<AnyScalarBuffer> AnyScalarBuffer::make(
    std::unique_ptr<T[]> data, size_t len) {
  return std::make_unique<OwnedAnyScalarBuffer<T>>(std::move(data), len);
}

inline size_t GetLenFromShape(const std::vector<int64_t>& shape) {
  size_t result = 1;
  for (int64_t i : shape) {
    result *= i;
  }
  return result;
}

class Tensor {
 public:
  Tensor(std::vector<float> data, std::vector<int64_t> shape)
      : data_(std::make_shared<VectorScalarBuffer<float>>(std::move(data))),
        shape_(shape) {}

  template <typename T>
  Tensor(std::unique_ptr<T[]> data, std::vector<int64_t> shape)
      : data_(std::make_shared<OwnedAnyScalarBuffer<T>>(
            std::move(data), GetLenFromShape(shape))),
        shape_(shape) {}

  Tensor(std::unique_ptr<AnyScalarBuffer> data, std::vector<int64_t> shape)
      : data_(std::move(data)), shape_(std::move(shape)) {}

  ScalarType scalar_type() const { return data_->scalar_type(); }

  bool equal(const Tensor& other) const {
    if (shape_ != other.shape_ || scalar_type() != other.scalar_type()) {
      return false;
    }
    switch (scalar_type()) {
#define DEFINE_COMPARE_CASE(name, aten_name, type) \
  case ScalarType::aten_name:                      \
    return data<type>() == other.data<type>();
      LIST_SCALAR_TYPES(DEFINE_COMPARE_CASE)
#undef DEFINE_COMPARE_CASE
    }
  }

  const std::vector<int64_t>& shape() const { return shape_; }

  template <typename T>
  absl::Span<const T> data() const {
    return absl::Span<const T>(data_->data<const T>(), data_->size());
  }

  size_t rank() const { return shape().size(); }

  Scalar item() const {
    switch (scalar_type()) {
#define DEFINE_ITEM_CASE(name, aten_name, type) \
  case ScalarType::aten_name:                   \
    return data<type>()[0];
      LIST_SCALAR_TYPES(DEFINE_ITEM_CASE)
#undef DEFINE_ITEM_CASE
    }
  }

  Tensor dup() const { return Tensor(*this); }

  const AnyScalarBuffer& buffer() const { return *data_; }

 private:
  // TODO(parkers): Support storage aliasing?
  std::shared_ptr<AnyScalarBuffer> data_;
  std::vector<int64_t> shape_;
};

namespace Reduction {

enum Reduction {
  None,  // Do not reduce
  Mean,  // (Possibly weighted) mean of losses
  Sum,   // Sum losses
  END
};
}  // namespace Reduction

}  // namespace at

namespace c10 {

using unique_t = uint32_t;

struct Symbol {
  constexpr operator unique_t() const { return value; }  // NOLINT

  Symbol(at::aten::SymbolKind value) : value(value) {}  // NOLINT

  Symbol(at::prim::SymbolKind value) : value(value) {}  // NOLINT

  Symbol(swift_xla::xla_symbols::SymbolKind value) : value(value) {}  // NOLINT

  Symbol() : value(at::prim::END_Symbol) {}

  const char* toQualString() const;

  static Symbol fromQualString(const std::string& s) { return Symbol(); }

 private:
  unique_t value;
};

template <class T>
using optional = absl::optional<T>;

}  // namespace c10

namespace swift_xla {

template <typename T, typename S>
T OptionalOr(const c10::optional<S>& value, T defval) {
  return value ? static_cast<T>(*value) : defval;
}

// Return at::ScalarType from at::Scalar
inline at::ScalarType GetScalarType(at::Scalar scalar) {
  return at::ScalarType::Double;
}

}  // namespace swift_xla
